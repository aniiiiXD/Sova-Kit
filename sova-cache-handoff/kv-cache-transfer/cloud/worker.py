"""Bounded pretrained-model mechanism pilot; no serving endpoint or model SFT."""
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/opt/ml/model')
OUT.mkdir(parents=True, exist_ok=True)


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def upload():
    import boto3
    s3 = boto3.client('s3')
    for path in OUT.rglob('*'):
        if path.is_file():
            s3.upload_file(str(path), os.environ['KV_BUCKET'],
                           os.environ['KV_PREFIX']+'/live/'+path.relative_to(OUT).as_posix())


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.data import Example, assert_disjoint, load_examples, json_digest
    from kvverify.hf_backend import DenseRuntime, RidgeMapper, legacy
    from kvverify.runner import evaluate

    torch.set_num_threads(4)
    torch.manual_seed(19)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    from packaging.version import Version
    if Version(torch.__version__.split('+')[0]) < Version('2.1.1'):
        raise RuntimeError('Transformers SDPA requires Torch >=2.1.1; fail before checkpoint downloads')
    probe = torch.randn(1, 2, 4, 8, device='cuda:0', dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(probe, probe, probe, is_causal=True)
    torch.cuda.synchronize()
    del probe
    config = json.loads((ROOT/'cloud/config.json').read_text())
    write('environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'dtype': 'float16',
          'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
          'config': config, 'scope': '12-question exploratory mechanism pilot, not thesis acceptance'})
    paths = []
    for spec in config['models']:
        print('Downloading pinned public checkpoint:', spec, flush=True)
        paths.append(snapshot_download(spec['repo'], revision=spec['revision'], token=False,
            allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json'], max_workers=4))
    runtime = DenseRuntime.from_local(*paths, device='cuda:0', dtype='float16')
    rng = random.Random(19)
    train = [json.loads(x) for x in (ROOT/'inputs/train-documents.jsonl').read_text().splitlines()]
    dev = [json.loads(x) for x in (ROOT/'inputs/dev-documents.jsonl').read_text().splitlines()]
    decisions = [json.loads(x) for x in (ROOT/'inputs/dev-decisions.jsonl').read_text().splitlines()]
    rng.shuffle(train)
    calibration = [Example('cal:'+d['document_id'], 'contractnli:'+d['document_id'], d['text'],
                     'Calibration context only; labels are not used.', {'A': 'Unused', 'B': 'Unused'},
                     'A', 'calibration') for d in train[:12]]
    lookup = {d['document_id']: d for d in dev}
    rng.shuffle(decisions)
    labels = {'Entailment': 'A', 'Contradiction': 'B', 'NotMentioned': 'C'}
    choices = {'A': 'Entailment', 'B': 'Contradiction', 'C': 'NotMentioned'}
    counts = {key: 0 for key in labels}
    selected, used, rejected = [], set(), []
    for d in decisions:
        label, doc = d['gold_label'], lookup[d['document_id']]
        if counts[label] >= 4 or doc['document_id'] in used:
            continue
        example = Example(d['decision_id'], 'contractnli:'+doc['document_id'], doc['text'],
            'Classify whether the document entails, contradicts, or does not mention this statement: '+d['statement'],
            choices, labels[label], 'eval')
        length = len(runtime.tokenize(example))
        if length > 8192:
            rejected.append({'id': example.id, 'prompt_tokens': length, 'reason': 'pilot selection cap; no truncation'})
            continue
        selected.append(example)
        used.add(doc['document_id'])
        counts[label] += 1
        if len(selected) == 12:
            break
    if len(selected) != 12:
        raise RuntimeError('Could not select the predeclared 12 document-disjoint questions')
    assert_disjoint(calibration, selected)
    from dataclasses import asdict
    for name, examples in [('calibration', calibration), ('heldout', selected)]:
        path = OUT/(name+'.jsonl')
        path.write_text(''.join(json.dumps(asdict(e))+'\n' for e in examples))
        load_examples(path, 'calibration' if name == 'calibration' else 'eval')
    write('data-selection.json', {'seed': 19, 'counts': counts, 'excluded_from_selection': rejected,
          'source': 'Existing local ContractNLI official train_clean/dev; no official test used',
          'selection': 'First eligible distinct documents after seeded decision shuffle, four per gold label',
          'calibration_prefix_tokens': 1024, 'maximum_eval_prompt_tokens': 8192,
          'eval_lengths': [len(runtime.tokenize(e)) for e in selected],
          'limitations': 'Small label-balanced convenience pilot; not representative benchmark accuracy.'})
    upload()
    # Same-checkpoint transfer must work before testing a cross-checkpoint mapper.
    print('Checking real pretrained identity-cache and zero-cache controls', flush=True)
    control = DenseRuntime(runtime.target, runtime.target, runtime.tokenizer)
    ids = runtime.tokenize(selected[0])[:128]
    with torch.inference_mode():
        native = control.forward(control.target, ids).logits[0, -1].float()
        prefix = control.forward(control.target, ids[:-1]).past_key_values
        mapped = control.identity_mapper().to('cuda:0').apply(prefix, control.source, control.target)
        actual = control.forward(control.target, ids[-1:], mapped).logits[0, -1].float()
        zero = control.DynamicCache.from_legacy_cache(tuple((torch.zeros_like(k), torch.zeros_like(v))
                                                          for k, v in legacy(prefix)))
        corrupted = control.forward(control.target, ids[-1:], zero).logits[0, -1].float()
        identity_delta = float((native-actual).abs().max())
        zero_delta = float((native-corrupted).abs().max())
        passed = (torch.allclose(native, actual, rtol=.02, atol=.1)
                  and int(native.argmax()) == int(actual.argmax()) and zero_delta > 1e-4)
    write('controls.json', {'passed': bool(passed), 'tokens': len(ids),
          'identity_max_logit_delta': identity_delta, 'zero_cache_max_logit_delta': zero_delta,
          'identity_first_token_equal': int(native.argmax()) == int(actual.argmax()),
          'rtol': .02, 'atol': .1})
    upload()
    if not passed:
        raise RuntimeError('Pretrained cache controls failed; refusing performance evaluation')
    del control, native, prefix, mapped, actual, zero, corrupted
    torch.cuda.empty_cache()
    print('Fitting depth-aligned ridge mapper on 12 training documents', flush=True)
    started = time.perf_counter()
    mapper = runtime.fit(calibration, alpha=.01, prefix_tokens=1024)
    mapper.save(OUT/'mapper')
    runtime.mapper = RidgeMapper.load(OUT/'mapper', runtime.identities['source']['sha256'],
        runtime.identities['target']['sha256'], runtime.tokenizer_sha256).to('cuda:0')
    write('fit.json', {'seconds': time.perf_counter()-started,
                      'tokens': mapper.manifest['calibration_tokens']})
    upload()
    print('Evaluating 12 held-out questions, five arms, two repetitions', flush=True)
    result = evaluate(runtime, selected, OUT/'evaluation', repeats=2, warmups=1, max_new_tokens=4)
    write('worker-status.json', {'status': 'completed', 'acceptance': 'not_established',
          'summary': result, 'cost_savings': 'not_measured; compare only after collection'})
    upload()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json', {'status': 'failed', 'acceptance': 'not_established'})
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
