"""Train our mapper and evaluate the frozen development protocol on one GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/opt/ml/model')
CHECKPOINTS = Path('/opt/ml/checkpoints')
UPLOADED = {}
STARTED = time.monotonic()
CONFIG = {}


def write(name, data):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload():
    import boto3
    from botocore.config import Config
    client = boto3.client('s3', config=Config(connect_timeout=10, read_timeout=20,
                                            retries={'max_attempts': 2}))
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or path.name.endswith('.tmp'):
            continue
        digest = sha(path)
        if UPLOADED.get(str(path)) == digest:
            continue
        client.upload_file(str(path), os.environ['KV_BUCKET'],
            os.environ['KV_PREFIX']+'/live/'+path.relative_to(OUT).as_posix())
        UPLOADED[str(path)] = digest


def notice(progress):
    elapsed = time.monotonic()-STARTED
    memory = {}
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        memory = {'gpu_allocated_mib':torch.cuda.memory_allocated()/2**20,
                  'gpu_reserved_mib':torch.cuda.memory_reserved()/2**20}
    write('progress.json', {**progress, **memory, 'elapsed_worker_seconds': elapsed})
    path = progress.get('checkpoint_path') or progress.get('best_mapper')
    if path:
        source = Path(path)
        target = CHECKPOINTS/('best-final' if progress.get('best_mapper') else source.name)
        if not target.exists():
            shutil.copytree(source, target)
    print('PROGRESS '+json.dumps(progress), flush=True)
    upload()
    if CONFIG and elapsed > CONFIG['worker_timeout_seconds']-120:
        raise TimeoutError('Reserved final upload time reached; partial evidence retained')


def verify_inputs(config):
    from kvverify.data import load_examples, assert_disjoint
    manifest_path = ROOT/config['data_manifest']
    if sha(manifest_path) != config['data_manifest_sha256']:
        raise ValueError('Frozen dataset manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    for name, spec in manifest['files'].items():
        path = manifest_path.parent/name
        if not path.is_file() or sha(path) != spec['sha256']:
            raise ValueError('Frozen dataset file hash mismatch: '+name)
    splits = {name: load_examples(manifest_path.parent/manifest['splits'][name]['file'],
                manifest['splits'][name]['record_split']) for name in ('train','mapper_validation','dev')}
    assert_disjoint(splits['train'], splits['mapper_validation'])
    assert_disjoint(splits['train'], splits['dev'])
    assert_disjoint(splits['mapper_validation'], splits['dev'])
    for name, rows in splits.items():
        if len(rows) != manifest['splits'][name]['examples']:
            raise ValueError('Frozen split count mismatch: '+name)
    write('dataset-manifest.json', manifest)
    return splits


def identity_controls(runtime, example):
    import torch
    from kvverify.hf_backend import DenseRuntime, legacy
    control = DenseRuntime(runtime.target, runtime.target, runtime.tokenizer)
    ids = runtime.tokenize(example)[:128]
    with torch.inference_mode():
        expected = control.forward(control.target, ids).logits[0,-1].float()
        prefix = control.forward(control.source, ids[:-1]).past_key_values
        mapped = control.identity_mapper().to('cuda:0').apply(prefix, control.source, control.target)
        actual = control.forward(control.target, ids[-1:], mapped).logits[0,-1].float()
        zero = control.DynamicCache.from_legacy_cache(tuple((torch.zeros_like(k),torch.zeros_like(v))
                                                          for k,v in legacy(prefix)))
        corrupted = control.forward(control.target, ids[-1:], zero).logits[0,-1].float()
        okay = (torch.allclose(expected, actual, rtol=.02, atol=.1)
                and int(expected.argmax()) == int(actual.argmax())
                and float((expected-corrupted).abs().max()) > 1e-4)
        report = {'passed': bool(okay), 'tokens': len(ids), 'rtol': .02, 'atol': .1,
            'identity_max_logit_delta': float((expected-actual).abs().max()),
            'zero_cache_max_logit_delta': float((expected-corrupted).abs().max()),
            'first_token_equal': int(expected.argmax()) == int(actual.argmax())}
    write('controls.json', report)
    upload()
    if not okay:
        raise RuntimeError('Pretrained identity/cache controls failed')


def main():
    global CONFIG
    import torch
    from packaging.version import Version
    from huggingface_hub import snapshot_download
    from kvverify.data import check_heldout
    from kvverify.hf_backend import DenseRuntime, RidgeMapper
    from kvverify.train_mapper import NeuralMapper, train_mapper
    from kvverify.full_eval import run_quality, run_latency, run_long_latency

    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    CONFIG = json.loads(Path(os.environ.get('KV_CONFIG', ROOT/'cloud/full-config.json')).read_text())
    write('worker-config.json', CONFIG)
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG['seed'])
    if not torch.cuda.is_available() or Version(torch.__version__.split('+')[0]) < Version('2.1.1'):
        raise RuntimeError('Requires CUDA and Torch>=2.1.1; no fallback')
    probe = torch.randn(1,2,4,8,device='cuda:0',dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(probe,probe,probe,is_causal=True)
    torch.cuda.synchronize()
    del probe
    splits = verify_inputs(CONFIG)
    write('environment.json', {'torch':torch.__version__, 'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0), 'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,
        'dtype':CONFIG['dtype'], 'scope':'Development experiment, not final test or vendor replication'})
    notice({'phase':'downloading_models'})
    paths = [snapshot_download(spec['repo'], revision=spec['revision'],token=False,
                allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)
             for spec in CONFIG['models']]
    runtime = DenseRuntime.from_local(*paths,device='cuda:0',dtype=CONFIG['dtype'])
    runtime.prompt_token_limit = CONFIG['max_eval_tokens']
    write('model-identities.json', runtime.identities)
    identity_controls(runtime, splits['train'][0])
    torch.cuda.empty_cache()
    train_documents = list({e.document_id:e for e in splits['train']}.values())
    notice({'phase':'ridge_baseline_fit','documents':len(train_documents)})
    ridge = runtime.fit(train_documents,prefix_tokens=CONFIG['train_prefix_tokens'])
    ridge.save(OUT/'ridge')
    check_heldout(ridge.manifest,splits['dev'])
    if CONFIG['resume_dir']:
        candidates = list(Path(CONFIG['resume_dir']).rglob('manifest.json'))
        best = [p for p in candidates if p.parent.name in ('best-final','best')]
        if not best:
            raise ValueError('Resume requires a completed validation-selected best mapper; arbitrary last checkpoint is not selected automatically')
        if len(best) != 1:
            raise ValueError('Ambiguous resume prefix; select exactly one best mapper')
        neural = NeuralMapper.load(best[0].parent,runtime.identities['source']['sha256'],
            runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
        write('resume.json',{'mapper':str(best[0].parent),'mode':'Warm start; optimizer state is not resumed'})
        if CONFIG['stage'] != 'eval':
            raise ValueError('Current neural warm-start is eval-only; optimizer resumption is not implemented')
    else:
        if CONFIG['stage'] == 'eval':
            raise ValueError('Evaluation-only requires existing selected mapper')
        notice({'phase':'training_mapper','steps':CONFIG['train_steps']})
        available = CONFIG['worker_timeout_seconds']-(time.monotonic()-STARTED)-1800
        if available < 60:
            raise TimeoutError('Insufficient training budget after setup; no hidden extension')
        if CONFIG['val_prefix_tokens'] != CONFIG['train_prefix_tokens']:
            raise ValueError('This trainer requires equal train/validation window lengths')
        neural = train_mapper(runtime,splits['train'],splits['mapper_validation'],OUT/'training',
            steps=CONFIG['train_steps'],prefix_tokens=CONFIG['train_prefix_tokens'],tail_tokens=8,
            rank=CONFIG['rank'],checkpoint_every=50,callback=notice,
            max_seconds=min(CONFIG['train_max_seconds'],available),seed=CONFIG['seed'])
        # Evaluate the actual saved validation-selected artifact, not an unsaved
        # in-memory state that could differ from the deployed checkpoint.
        runtime.mapper = None
        del neural
        torch.cuda.empty_cache()
        neural = NeuralMapper.load(OUT/'training/best',runtime.identities['source']['sha256'],
            runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
    check_heldout(neural.manifest,splits['dev'])
    neural.eval()
    torch.cuda.empty_cache()
    if CONFIG['stage'] != 'train':
        # All development examples stay in the denominator, including over-limit cases.
        if len(splits['dev']) > CONFIG['max_eval_examples']:
            raise ValueError('Dev set exceeds frozen protocol; refusing a silent subset')
        notice({'phase':'development_quality','examples':len(splits['dev'])})
        quality = run_quality(runtime,splits['dev'],{'ridge':ridge.to('cuda:0'),'neural':neural},
            OUT/'quality',freegen_documents=CONFIG['freegen_documents'],
            freegen_max_new_tokens=CONFIG['freegen_max_new_tokens'],seed=CONFIG['seed'],callback=notice)
        write('quality-summary.json',quality)
        runtime.mapper = neural
        notice({'phase':'paired_latency'})
        latency = run_latency(runtime,splits['dev'],OUT/'latency',example_count=CONFIG['latency_examples'],
            repeats=CONFIG['latency_repeats'],seed=CONFIG['seed'],callback=notice)
        write('latency-summary.json',latency)
        remaining = CONFIG['worker_timeout_seconds']-(time.monotonic()-STARTED)-180
        if remaining > 300:
            notice({'phase':'synthetic_long_context_timing'})
            long_results = run_long_latency(runtime,OUT/'long-latency',repeats=10,
                max_seconds=min(900,remaining),callback=notice)
            write('long-latency-summary.json',long_results)
        else:
            write('long-latency-summary.json',{'status':'not_run_time_budget'})
    write('worker-status.json',{'status':'completed','stage':CONFIG['stage'],
        'elapsed_worker_seconds':time.monotonic()-STARTED,'acceptance':'not_established',
        'cost_savings':'not_measured; collect SageMaker billable seconds separately',
        'limitations':['Official dev is development evidence, not untouched final test.',
            'Both models and mappers are resident; memory is not native-only serving memory.',
            'Reference HF runtime, batch one; no production throughput claim.',
            'Synthetic long-context runs measure timing only.']})
    notice({'phase':'completed'})


def terminate(signum,frame):
    raise TimeoutError('Worker received termination signal; exporting partial evidence')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,terminate)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed','acceptance':'not_established',
            'elapsed_worker_seconds':time.monotonic()-STARTED})
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
