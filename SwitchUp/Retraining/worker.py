"""Train and replay a small LoRA adapter on one SageMaker instance."""
from pathlib import Path
import hashlib
import json
import random
import sys


INPUT = Path('/opt/ml/input/data/payload')
OUTPUT = Path('/opt/ml/model')
if not INPUT.is_dir() or not Path('/opt/ml/input/config/resourceconfig.json').is_file():
    raise SystemExit('Run the weights pilot as a SageMaker training job.')

sys.path.insert(0, str(INPUT / 'switchit.zip'))

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from switchit.composer import evaluate_candidates
from switchit.promotions import validate_selection
from switchit.recipes import make_recipe
from switchit.runtime_adapters import RuntimeAdapter, RuntimeRegistry, model_request


TARGETS = ('q_proj', 'v_proj')


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scale = alpha / rank
        options = {'bias': False, 'device': base.weight.device, 'dtype': torch.float32}
        self.lora_A = torch.nn.Linear(base.in_features, rank, **options)
        self.lora_B = torch.nn.Linear(rank, base.out_features, **options)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            delta = self.lora_B(self.lora_A(inputs.float())) * self.scale
            output = output + delta.to(output.dtype)
        return output


def install_adapters(model, rank, alpha):
    model.requires_grad_(False)
    adapters = {}
    for name, layer in list(model.named_modules()):
        parent, _, leaf = name.rpartition('.')
        if leaf in TARGETS and isinstance(layer, torch.nn.Linear):
            adapter = LoRALinear(layer, rank, alpha)
            setattr(model.get_submodule(parent), leaf, adapter)
            adapters[name] = adapter
    if not adapters:
        raise ValueError('the model has no matching attention projections')
    return adapters


def _prompt_ids(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, add_special_tokens=False).input_ids


def train(model, tokenizer, rows, config):
    seed = config['seed']
    rng = random.Random(seed)
    encoded = []
    for row in rows:
        messages = [{'role': 'system', 'content': config['system_prompt']}, {'role': 'user', 'content': row['text']}]
        prefix = _prompt_ids(tokenizer, messages)
        completion = tokenizer(row['corrected_action'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        encoded.append((prefix + completion, [-100] * len(prefix) + completion))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0)
    model.train()
    epoch_losses = []
    for epoch in range(config['epochs']):
        rng.shuffle(encoded)
        losses = []
        for start in range(0, len(encoded), config['batch_size']):
            batch = encoded[start:start + config['batch_size']]
            width = max(len(ids) for ids, _ in batch)
            inputs = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
            attention = torch.zeros_like(inputs)
            labels = torch.full_like(inputs, -100)
            for index, (ids, targets) in enumerate(batch):
                inputs[index, :len(ids)] = torch.tensor(ids, device='cuda')
                labels[index, :len(ids)] = torch.tensor(targets, device='cuda')
                attention[index, :len(ids)] = 1
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, attention_mask=attention, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(sum(losses) / len(losses))
        print(f'epoch={epoch+1} mean_loss={epoch_losses[-1]:.4f}', flush=True)
    model.eval()
    return epoch_losses


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def save_adapter(adapters, config):
    tensors = {}
    destinations = {}
    for name, adapter in adapters.items():
        for suffix in ('lora_A', 'lora_B'):
            key = f'base_model.model.{name}.{suffix}.weight'
            destination = getattr(adapter, suffix).weight
            destinations[key] = destination
            tensors[key] = destination.detach().cpu().contiguous()
    if not any(torch.count_nonzero(value).item() for key, value in tensors.items() if '.lora_B.' in key):
        raise ValueError('adapter updates are all zero')
    adapter_path = OUTPUT / 'adapter_model.safetensors'
    save_file(tensors, adapter_path, metadata={'format': 'pt'})
    _write(OUTPUT / 'adapter_config.json', {
        'base_model_name_or_path': config['model'], 'revision': config['revision'],
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'target_modules': list(TARGETS),
        'r': config['rank'], 'lora_alpha': config['alpha'], 'lora_dropout': 0.0,
        'bias': 'none', 'fan_in_fan_out': False, 'inference_mode': True,
    })
    restored = load_file(adapter_path)
    if set(restored) != set(destinations):
        raise ValueError('adapter checkpoint is incomplete')
    with torch.no_grad():
        for key, weight in destinations.items():
            weight.copy_(restored[key].to(device=weight.device, dtype=weight.dtype))
    return hashlib.sha256(adapter_path.read_bytes()).hexdigest()


class ReplayRuntime(RuntimeAdapter):
    runtime_id = 'sagemaker-lora'
    step_kinds = frozenset({'model'})

    def __init__(self, model, tokenizer, adapters, recipe):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters = adapters
        self.use_adapter = recipe['config']['use_adapter']
        self.models = frozenset({recipe['model']})

    def execute(self, step, inputs, context):
        for adapter in self.adapters.values():
            adapter.enabled = self.use_adapter
        request = model_request(step['request'], {**context['recipe'], 'config': {}})
        ids = _prompt_ids(self.tokenizer, request['messages'])
        tokens = torch.tensor([ids], device='cuda')
        with torch.inference_mode():
            output = self.model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                                         max_new_tokens=8, pad_token_id=self.tokenizer.pad_token_id)
        generated = output[0, len(ids):]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        calls = [{'name': 'cancel_order', 'arguments': {'id': inputs['order_id']}}] if text == 'CANCEL' else []
        return {'text': text, 'tool_calls': calls, 'usage': {'input_tokens': len(ids),
                'output_tokens': len(generated), 'basis': 'observed'}}


def replay(model, tokenizer, adapters, workflow, suite, config):
    recipes = [make_recipe(name, ReplayRuntime.runtime_id, ReplayRuntime.version, workflow['revision'],
                           workflow['allowed_tools'], model=config['model'], system_prompt=config['system_prompt'],
                           config={'use_adapter': enabled})
               for name, enabled in (('base-weights', False), ('lora-weights', True))]
    runtimes = RuntimeRegistry()
    runtimes.register(ReplayRuntime.runtime_id, lambda recipe, case: ReplayRuntime(model, tokenizer, adapters, recipe))
    experiment = evaluate_candidates(workflow, suite, recipes, runtimes, max_candidates=2,
                                     max_trials=2 * len(suite['cases']), max_seconds=60)
    pairs = experiment['paired_replay']['candidates'][0]['cases']
    split_by_case = {case['case_id']: case['split'] for case in suite['cases']}
    repaired = [pair['case_id'] for pair in pairs if pair['baseline_verdict'] == 'fail' and pair['candidate_verdict'] == 'pass']
    regressed = [pair['case_id'] for pair in pairs if pair['baseline_verdict'] == 'pass' and pair['candidate_verdict'] != 'pass']
    complete = all(pair['baseline_verdict'] in ('pass', 'fail') and pair['candidate_verdict'] in ('pass', 'fail') for pair in pairs)
    eligible = complete and bool(repaired) and experiment['selected_recipe'] == recipes[1]['recipe_id']
    if eligible:
        validate_selection(experiment)
    holdout = [pair for pair in pairs if split_by_case[pair['case_id']] == 'holdout']
    report = {'candidate_passed_gate': eligible, 'baseline_and_candidate_completed': complete,
              'repaired_case_ids': repaired, 'regressed_case_ids': regressed,
              'heldout': {'cases': len(holdout), 'baseline_pass': sum(p['baseline_verdict'] == 'pass' for p in holdout),
                          'adapter_pass': sum(p['candidate_verdict'] == 'pass' for p in holdout)}}
    return experiment, report


def main():
    if not torch.cuda.is_available():
        raise ValueError('a CUDA instance is required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((INPUT / 'manifest.json').read_text())
    for name, expected in config['files'].items():
        actual = hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'pilot input changed: {name}')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    checkpoint = snapshot_download(config['model'], revision=config['revision'],
                                   allow_patterns=['*.json', '*.safetensors', 'merges.txt', '*.jinja'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=False).to('cuda')
    adapters = install_adapters(model, config['rank'], config['alpha'])
    frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
    rows = json.loads((INPUT / 'training.json').read_text())
    losses = train(model, tokenizer, rows, config)
    if frozen_versions != {name: p._version for name, p in model.named_parameters() if not p.requires_grad}:
        raise ValueError('the base model changed during adapter training')
    adapter_digest = save_adapter(adapters, config)
    workflow = json.loads((INPUT / 'workflow.json').read_text())
    suite = json.loads((INPUT / 'suite.json').read_text())
    experiment, report = replay(model, tokenizer, adapters, workflow, suite, config)
    report.update(model=config['model'], revision=config['revision'], adapter_sha256=adapter_digest,
                  training_rows=len(rows), epoch_losses=losses,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), experiment_id=experiment['experiment_id'])
    _write(OUTPUT / 'experiment.json', experiment)
    _write(OUTPUT / 'manifest.json', config)
    _write(OUTPUT / 'report.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
