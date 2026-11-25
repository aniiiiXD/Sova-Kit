"""One selected-checkpoint diagnostic; no training or benchmark resampling."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=Path('/opt/ml/model')
UPLOADED={}
STARTED=time.monotonic()


def write(name,data):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def upload():
    import boto3
    from botocore.config import Config
    client=boto3.client('s3',config=Config(connect_timeout=10,read_timeout=20,retries={'max_attempts':2}))
    for p in sorted(OUT.rglob('*')):
        if not p.is_file():continue
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if UPLOADED.get(str(p))==sha:continue
        client.upload_file(str(p),os.environ['KV_BUCKET'],os.environ['KV_PREFIX']+'/live/'+p.relative_to(OUT).as_posix())
        UPLOADED[str(p)]=sha


def notice(progress):
    write('progress.json',{**progress,'elapsed_worker_seconds':time.monotonic()-STARTED})
    print('PROGRESS '+json.dumps(progress),flush=True);upload()


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import NeuralMapper
    from kvverify.boundary_probe import run_probe
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text());write('config.json',cfg)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    for name,expected in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen input hash mismatch: '+name)
    examples=load_examples(ROOT/'inputs/dev.jsonl','eval')
    notice({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4) for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype=cfg['dtype'])
    runtime.prompt_token_limit=cfg['max_prompt_tokens']
    runtime.mapper=NeuralMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],
        runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
    write('environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'identities':runtime.identities,'mapper':runtime.mapper.manifest})
    notice({'phase':'oracle_and_boundary_probe','documents':cfg['probe_examples']})
    result=run_probe(runtime,examples,OUT/'probe',tails=tuple(cfg['tails']),
                     max_examples=cfg['probe_examples'],seed=cfg['seed'],callback=notice)
    write('worker-status.json',{'status':'completed','acceptance':'not_established',
        'summary':result,'elapsed_worker_seconds':time.monotonic()-STARTED})
    notice({'phase':'completed'})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker time limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed','acceptance':'not_established'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
