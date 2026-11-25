"""Evaluation-only continuation of fixed incomplete reference triplets.

Requires a separately approved bounded job. This module creates no resources
and contains no training, checkpoint selection, parameter update or tuner.
"""
from contextlib import ExitStack
from dataclasses import asdict
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
from cloud.probe_worker import OUT,STARTED,write,upload,notice


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_input_scope(config,records,quality):
    from kvverify.data import json_digest
    if config is None:return
    field='example_hash' if quality else 'records_hash'
    values=[asdict(record) for record in records] if quality else records
    if config.get(field)!=json_digest(values):
        raise ValueError('Source evaluation was not on these exact frozen inputs')


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import NeuralMapper
    from kvverify.reference_eval import (load_reference_data,run_quality_reference,
        run_hellaswag_reference,QUALITY_PROTOCOL,HELLASWAG_PROTOCOL,_summary)
    from kvverify.eval_resume import (quality_items,hellaswag_items,plan_resume,
        merge_resume,verify_evaluation_config)
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    write('config.json',cfg)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no fallback')
    if (cfg['dtype']!='float32' or cfg['attention_backend']!='math' or cfg['query_block_size']!=256
            or cfg['eval_max_prompt_tokens']!=9000 or cfg['quality_freegen_tokens']!=16):
        raise ValueError('Continuation cannot alter the approved numerical/evaluation protocol')
    for name,sha in cfg['input_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe input path')
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen continuation input changed: '+name)
    quality,hella,_=load_reference_data(ROOT/'reference',cfg['input_sha256']['reference/manifest.json'])
    parent_root=ROOT/'inputs/parent'
    parent_worker_path=parent_root/'worker-config.json'
    if parent_worker_path.relative_to(ROOT).as_posix() not in cfg['input_sha256']:
        raise ValueError('Exact parent worker numerical configuration must be bundled and hashed')
    parent_worker=json.loads(parent_worker_path.read_text())
    for field in ('models','dtype','attention_backend','query_block_size','eval_max_prompt_tokens','quality_freegen_tokens','seed'):
        if parent_worker.get(field)!=cfg.get(field):
            raise ValueError('Parent numerical/evaluation setting changed: '+field)
    plans={}
    for name,items,protocol in [('quality',quality_items(quality),QUALITY_PROTOCOL),
                                ('hellaswag',hellaswag_items(hella),HELLASWAG_PROTOCOL)]:
        config_path=parent_root/name/'config.json'
        rows_path=parent_root/name/'rows.jsonl'
        if config_path.exists() and rows_path.exists():
            parent_config=json.loads(config_path.read_text());parent_rows=read_rows(rows_path)
        elif name in cfg.get('parent_not_started_benchmarks',[]) and not config_path.exists() and not rows_path.exists():
            parent_config=None;parent_rows=[]
        else:raise ValueError('Parent evidence missing without explicit not-started declaration: '+name)
        verify_input_scope(parent_config,quality if name=='quality' else hella,name=='quality')
        # Every existing parent artifact must be included in the frozen bundle hashes.
        for path in (config_path,rows_path):
            if path.exists() and path.relative_to(ROOT).as_posix() not in cfg['input_sha256']:
                raise ValueError('Unhashed parent evidence')
        plans[name]=(plan_resume(items,parent_rows,protocol,parent_config),parent_rows)
    write('resume-plan.json',{name:value[0] for name,value in plans.items()})
    notice({'phase':'loading_models','continuation_only':True})
    paths=[snapshot_download(model['repo'],revision=model['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4) for model in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype=cfg['dtype'])
    runtime.prompt_token_limit=cfg['eval_max_prompt_tokens']
    for model in(runtime.source,runtime.target):model.eval().requires_grad_(False)
    mapper=NeuralMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],
        runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
    if mapper.manifest['weights_sha256']!=cfg['mapper_weights_sha256']:
        raise ValueError('Selected checkpoint changed')
    runtime.mapper=mapper
    write('environment.json',{'identities':runtime.identities,'tokenizer_sha256':runtime.tokenizer_sha256,
        'mapper_weights_sha256':mapper.manifest['weights_sha256'],'torch':torch.__version__,
        'dtype':cfg['dtype'],'attention_backend':cfg['attention_backend'],'query_block_size':cfg['query_block_size']})
    summaries={}
    child_job=os.environ['KV_PREFIX'].rstrip('/').split('/')[-1]
    for name,records,evaluator,metrics in [
            ('quality',quality,run_quality_reference,('decision_correct','free_generation_strict_correct','free_generation_format_valid')),
            ('hellaswag',hella,run_hellaswag_reference,('acc','acc_norm'))]:
        plan,parent_rows=plans[name]
        verify_evaluation_config(plan['parent_config'],mapper_weights_sha256=cfg['mapper_weights_sha256'],
            runtime_identities=runtime.identities,quality=name=='quality')
        pending=set(plan['pending_ids'])
        subset=[r for r in records if (r.id if name=='quality' else r['benchmark_id']) in pending]
        child_rows=[];child_config=None
        if subset:
            remaining=cfg['worker_timeout_seconds']-(time.monotonic()-STARTED)-120
            if remaining>0:
                budget=min(cfg[name+'_max_seconds'],remaining)
                kwargs={'max_seconds':budget,'callback':notice,'seed':cfg['seed']}
                if name=='quality':kwargs['freegen_max_new_tokens']=cfg['quality_freegen_tokens']
                evaluator(runtime,subset,mapper,OUT/name,**kwargs)
                child_rows=read_rows(OUT/name/'rows.jsonl')
                child_config=json.loads((OUT/name/'config.json').read_text())
                verify_input_scope(child_config,subset,name=='quality')
                verify_evaluation_config(child_config,mapper_weights_sha256=cfg['mapper_weights_sha256'],
                    runtime_identities=runtime.identities,quality=name=='quality')
        merged=merge_resume(plan,parent_rows,child_rows,child_config,parent_job=cfg['parent_job_name'],
            child_job=child_job,mapper_weights_sha256=cfg['mapper_weights_sha256'])
        destination=OUT/'merged'/name;destination.mkdir(parents=True,exist_ok=False)
        (destination/'rows.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in merged))
        summary=_summary(merged,metrics)
        summary.update(parent_job=cfg['parent_job_name'],continuation_job=child_job,
            mapper_weights_sha256=cfg['mapper_weights_sha256'],frozen_items_sha256=plan['frozen_items_sha256'],
            parent_rows_sha256=plan['parent_rows_sha256'],child_rows_sha256=hashlib.sha256(json.dumps(child_rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            latency_headline_permitted=False,selection_policy=plan['selection_policy'],
            variant_order_policy=plan['variant_order_policy'],training_performed=False)
        write('merged/'+name+'/summary.json',summary);summaries[name]=summary;upload()
    write('worker-status.json',{'status':'completed' if all(s['fully_completed'] for s in summaries.values())
        else 'completed_with_unfinished_execution','acceptance':'not_established',
        'training_performed':False,'summaries':summaries,'elapsed_worker_seconds':time.monotonic()-STARTED})
    notice({'phase':'completed','continuation_only':True})


def run_configured():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from cloud.precision_worker import precision_settings
    from kvverify.blocked_attention import blocked_attention
    precision_settings(torch)
    with ExitStack() as stack:
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        stack.enter_context(blocked_attention(256))
        return main()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Continuation worker time limit')))
    try:run_configured()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed','acceptance':'not_established','training_performed':False})
        try:upload()
        except Exception:traceback.print_exc()
        raise
