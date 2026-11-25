"""Train one folded answer mapper and enforce the development gate."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.probe_worker import OUT, STARTED, notice, upload, write


def development_gate(row):
    n = row['questions']
    return (n == 30 and row['native_correct'] >= 15
            and row['correct'] >= row['native_correct'] - .03*n
            and row['correct'] > max(row['source_correct'], row['prefix_only_correct'])
            and row['native_agreement'] >= .8*n and row['mean_native_kl'] <= .5)


def development_rows(examples, seed):
    from kvverify.qa_training import group_questions

    rng = random.Random(seed)
    groups = group_questions(examples)
    if len(groups) != 15 or any(len(rows) < 2 for rows in groups.values()):
        raise ValueError('Expected 15 previously scored articles with two questions each')
    return [item for questions in groups.values() for item in rng.sample(questions, 2)]


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.activation_offload import selective_saved_tensors_cpu
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.answer_mapper import AnswerMapper, answer_loss
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.data import assert_disjoint, load_examples
    from kvverify.fast_affine import map_affine_fast
    from kvverify.hf_backend import DenseRuntime, legacy
    from kvverify.native_prefix import replace_prefix
    from kvverify.qa_training import DocumentQuestionSampler, split_document_prompt
    from kvverify.quality import option_token_ids, score_option_logits

    cfg = json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode'] != 'answer_train' or not torch.cuda.is_bf16_supported():
        raise ValueError('Answer mapper requires BF16-capable CUDA')
    for name, expected in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input drifted: '+name)
    train = load_examples(ROOT/'inputs/train.jsonl', 'calibration')
    development = development_rows(load_examples(ROOT/'inputs/development.jsonl', 'eval'), cfg['seed'])
    assert_disjoint(train, development)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    write('config.json', cfg)
    paths = []
    for model in cfg['models']:
        notice({'phase': 'downloading_model', 'candidate': model['repo']})
        paths.append(snapshot_download(model['repo'], revision=model['revision'], token=False,
                     allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json'], max_workers=4))
    notice({'phase': 'loading_models'})
    runtime = DenseRuntime.from_local(*paths, dtype='bfloat16')
    for model in (runtime.source, runtime.target):
        model.requires_grad_(False)
    base = AffineCacheMapper.load(cfg['input_mapper_dir'], runtime.identities['source']['sha256'],
                                  runtime.identities['target']['sha256'], runtime.tokenizer_sha256).to('cuda:0')
    if base.manifest['weights_sha256'] != cfg['initial_weights_sha256']:
        raise ValueError('Base adapter hash changed')
    mapper = AnswerMapper(base, cfg['rank'])
    trained_parameters = [p for p in mapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained_parameters, lr=cfg['learning_rate'], weight_decay=0.)
    prefix_count = cfg['native_prefix_tokens']
    limit = cfg['max_prompt_tokens']
    notice({'phase': 'tokenizing_training'})
    lengths = {item.id: len(runtime.tokenize(item)) for item in train}
    eligible = [item for item in train if 256 < lengths[item.id] <= limit]
    sampler = DocumentQuestionSampler(eligible, cfg['seed'])
    boundaries = {item.id: split_document_prompt(runtime, item) for item in development}
    if any(len(full)>limit or len(prefix)<=prefix_count for full,prefix,_ in boundaries.values()):
        raise ValueError('Development row outside the frozen context limit')
    write('protocol.json', {'train_documents': sampler.documents, 'train_questions_eligible': len(eligible),
          'validation_ids': [item.id for item in development], 'validation_role': 'previously-scored development',
          'native_prefix_tokens': prefix_count, 'rank': cfg['rank'],
          'trainable_parameters': sum(p.numel() for p in trained_parameters),
          'loss': 'Equal weights for forward KL on full vocabulary and conditional answer options',
          'student_attention': 'math SDPA with 64-query blocks',
          'teacher_and_evaluation_attention': 'automatic SDPA',
          'gold_in_training': False, 'locked_confirmation_bundled': False})

    def loss_on(item):
        full, prefix, suffix = split_document_prompt(runtime, item)
        options = option_token_ids(runtime.tokenizer, item.choices)
        return answer_loss(runtime, mapper, full, prefix, suffix, options, native_prefix_tokens=prefix_count)

    notice({'phase': 'backward_preflight', 'step': 0})
    preflight_item = max(eligible, key=lambda item: lengths[item.id])
    offload = selective_saved_tensors_cpu([runtime.source,runtime.target,base], pin_memory=False)
    with offload:
        loss, _ = loss_on(preflight_item)
        loss.backward()
    for parameter in trained_parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError('Mapper gradient preflight failed')
    if any(p.grad is not None for model in (runtime.source,runtime.target,base) for p in model.parameters()):
        raise ValueError('Frozen checkpoint received gradients')
    optimizer.zero_grad(set_to_none=True)
    del loss
    write('memory-preflight.json', {'status': 'passed', 'tokens': lengths[preflight_item.id],
                                    'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20})

    controls = {}
    history = []
    best = None
    deadline = STARTED + cfg['worker_timeout_seconds'] - 180

    @torch.no_grad()
    def validate(step):
        nonlocal best
        notice({'phase': 'validation_begin', 'step': step})
        folded = mapper.folded()
        rows = []
        for item in development:
            full, prefix, suffix = boundaries[item.id]
            tokens = option_token_ids(runtime.tokenizer, item.choices)
            source = runtime.forward(runtime.source, prefix).past_key_values
            head = runtime.forward(runtime.target, prefix[:prefix_count]).past_key_values
            if item.id not in controls:
                native = runtime.forward(runtime.target, full).logits[0, -1].float().cpu()
                source_only = runtime.forward(runtime.source, suffix,
                             runtime.DynamicCache.from_legacy_cache(legacy(source))).logits[0, -1]
                prefix_only = runtime.forward(runtime.target, suffix,
                              runtime.DynamicCache.from_legacy_cache(legacy(head))).logits[0, -1]
                controls[item.id] = (native, *(score_option_logits(logits, tokens, item.answer)
                                                for logits in (native, source_only, prefix_only)))
            native, native_score, source_score, prefix_score = controls[item.id]
            mapped = map_affine_fast(folded, source, runtime.source, runtime.target)
            with_head = replace_prefix(mapped, head, prefix_count)
            logits = runtime.forward(runtime.target, suffix, with_head).logits[0, -1].float().cpu()
            score = score_option_logits(logits, tokens, item.answer)
            distribution = probability_equivalence(torch, native, logits)
            rows.append({'id': item.id, 'document_id': item.document_id, 'native': native_score,
                        'source': source_score, 'prefix_only': prefix_score, 'candidate': score,
                        'native_agreement': score['decision_answer'] == native_score['decision_answer'],
                        'kl': distribution['native_to_candidate_kl_nats']})
        row = {'step': step, 'questions': len(rows), 'documents': len({r['document_id'] for r in rows}),
               'correct': sum(r['candidate']['decision_correct'] for r in rows),
               'native_correct': sum(r['native']['decision_correct'] for r in rows),
               'source_correct': sum(r['source']['decision_correct'] for r in rows),
               'prefix_only_correct': sum(r['prefix_only']['decision_correct'] for r in rows),
               'native_agreement': sum(r['native_agreement'] for r in rows),
               'mean_native_kl': sum(r['kl'] for r in rows)/len(rows)}
        row['passes_development_gate'] = development_gate(row)
        # Passable checkpoints come first, then gold accuracy, then fidelity.
        key = (row['passes_development_gate'], row['correct'], row['native_agreement'], -row['mean_native_kl'])
        if best is None or key > best['key']:
            best = {'key': key, 'row': row, 'state': {'down': mapper.down.detach().cpu().clone(),
                                                    'up': mapper.up.detach().cpu().clone()}}
            write('best-selection.json', row)
        history.append(row)
        write('validation-history.json', history)
        write(f'validation/step-{step:06d}.json', rows)
        notice({'phase': 'validation', **row, 'selected': best['row']['step']})

    validate(0)
    completed = 0
    sampled_docs = set()
    last_notice = time.monotonic()
    for step in range(1, cfg['steps']+1):
        if time.monotonic() >= deadline:
            break
        example = sampler.next()
        optimizer.zero_grad(set_to_none=True)
        with selective_saved_tensors_cpu([runtime.source, runtime.target, base], pin_memory=False):
            loss, metrics = loss_on(example)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trained_parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        completed = step
        sampled_docs.add(example.document_id)
        row = {'step': step, 'question_id': example.id, 'document_id': example.document_id, **metrics}
        with (OUT/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if time.monotonic() - last_notice >= 25:
            notice({'phase': 'training', 'total': cfg['steps'], 'unique_documents': len(sampled_docs),
                    'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20, **row})
            last_notice = time.monotonic()
        if step in cfg['validate_at']:
            validate(step)
    if history[-1]['step'] != completed:
        validate(completed)
    with torch.no_grad():
        for name, value in best['state'].items():
            getattr(mapper, name).copy_(value)
    selected = best['row']
    artifact = mapper.folded({'training_objective': 'first_answer_full_and_option_kl',
                 'rank': cfg['rank'], 'native_prefix_policy': str(prefix_count),
                 'trained_steps': completed, 'selected_at_step': selected['step'],
                 'initial_weights_sha256': cfg['initial_weights_sha256'],
                 'training_documents': sampler.documents,
                 'selection_documents': sorted({item.document_id for item in development})})
    saved = artifact.save(OUT/'best')
    report = {'status': 'completed' if completed == cfg['steps'] else 'time_budget_reached',
               'requested_updates': cfg['steps'], 'completed_updates': completed,
               'selected': selected, 'validation_history': history, 'weights_sha256': saved['weights_sha256'],
               'next_stage': 'candidate_timing' if selected['passes_development_gate'] else 'development_rejected',
               'locked_confirmation_accessed': False, 'product_goal_achieved': False,
               'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20}
    write('summary.json', report)
    write('worker-status.json', report)
    notice({'phase': 'completed_training', 'step': completed, 'selected': selected['step'],
            'correct': selected['correct'], 'native_agreement': selected['native_agreement'],
            'mean_native_kl': selected['mean_native_kl']})


if __name__ == '__main__':
    def interrupt(*_):
        raise TimeoutError('SageMaker worker deadline')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json', {'status': 'failed'})
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
