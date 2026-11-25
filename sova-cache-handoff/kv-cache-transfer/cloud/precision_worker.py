"""Target-only precision/backend diagnosis on the exact failed control inputs."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from contextlib import nullcontext,ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.probe_worker import OUT,STARTED,write,upload,notice


def frozen_examples(examples,prior):
    """Preserve the prior 16 IDs and order; never select using new outputs."""
    from cloud.repair_worker import split_train_articles
    _,validation=split_train_articles(examples)
    by_id={e.id:e for e in validation}
    rows=prior.get('rows',[])
    ids=[r['id']for r in rows]
    if len(ids)!=16 or len(set(ids))!=16:raise ValueError('Exactly 16 unique frozen control IDs required')
    result=[]
    for row in rows:
        e=by_id.get(row['id'])
        if e is None or e.document_id!=row['document_id']or e.answer!=row['gold_answer']:
            raise ValueError('Prior control identity differs from frozen validation data')
        result.append(e)
    return result


def precision_settings(torch):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    # Available in the pinned worker Torch; math SDPA must retain FP32 reductions.
    setter=getattr(torch.backends.cuda,'allow_fp16_bf16_reduction_math_sdp',None)
    if setter is not None:setter(False)
    return {'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def check_example(runtime,e):
    from kvverify.qa_training import split_document_prompt
    from kvverify.boundary_probe import probability_equivalence
    ids,prefix,suffix=split_document_prompt(runtime,e)
    if len(ids)>runtime.prompt_token_limit:raise ValueError('Frozen control exceeds unchanged prompt limit')
    out=runtime.forward(runtime.target,ids)
    full=out.logits[0,-1].detach().float().cpu().clone()
    del out
    cache=runtime.forward(runtime.target,prefix).past_key_values
    if cache.get_seq_length()!=len(prefix):raise ValueError('Native prefix cache has incorrect length')
    out=runtime.forward(runtime.target,suffix,cache)
    segmented=out.logits[0,-1].detach().float().cpu().clone()
    if out.past_key_values.get_seq_length()!=len(ids):raise ValueError('Native suffix cache has incorrect final length')
    del cache,out
    return {'id':e.id,'document_id':e.document_id,'context_hash':e.context_hash,
        'prompt_tokens':len(ids),'prefix_tokens':len(prefix),'native_suffix_tokens':len(suffix),
        'token_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        'oracle':probability_equivalence(runtime.torch,full,segmented)},full,segmented


def run_arm(runtime,examples,output,deadline,callback=None):
    from safetensors.torch import save_file
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,e in enumerate(examples):
        if time.monotonic()>=deadline:
            rows.append({'id':e.id,'status':'not_run_time_budget','oracle':{'passed':False}})
            continue
        runtime.torch.cuda.reset_peak_memory_stats()
        row,full,segmented=check_example(runtime,e)
        path=output/f'{index:03d}.safetensors'
        save_file({'native':full.contiguous(),'native_cache_oracle':segmented.contiguous()},str(path))
        row.update(status='ok',logits_file=path.name,logits_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            peak_allocated_mib=runtime.torch.cuda.max_memory_allocated()/2**20)
        rows.append(row)
        (output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        if callback:callback({'completed_controls':index+1,'total_controls':len(examples)})
    passed=sum(r['oracle']['passed']for r in rows)
    metrics=[r['oracle']for r in rows if r['status']=='ok']
    summary={'examples':len(rows),'passed':passed,'all_passed':passed==len(examples),
        'first_token_matches':sum(r.get('first_token_equal',False)for r in metrics),
        'max_total_variation':max((r['total_variation']for r in metrics),default=None),
        'max_forward_kl_nats':max((r['native_to_candidate_kl_nats']for r in metrics),default=None),
        'max_reverse_kl_nats':max((r['candidate_to_native_kl_nats']for r in metrics),default=None),
        'rows':rows}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from huggingface_hub import snapshot_download
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='precision'or cfg['training_enabled']:raise ValueError('Diagnostic must not train')
    if not torch.cuda.is_available():raise RuntimeError('CUDA worker required; no CPU fallback')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg)
    for name,expected in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input hash mismatch: '+name)
    prior=json.loads((ROOT/'inputs/prior-native-control.json').read_text())
    examples=frozen_examples(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),prior)
    expected_tokens={r['id']:r['prompt_tokens']for r in prior['rows']}
    settings=precision_settings(torch)
    write('protocol.json',{'protocol':'target-precision-diagnostic-v1','selected_ids':[e.id for e in examples],
        'settings':settings,'unchanged_oracle_protocol':'oracle-probability-v2',
        'arms':cfg['precision_arms'],'source_loaded':False,'mapper_used':False,'training_enabled':False,
        'query_block_size':cfg.get('query_block_size'),
        'note':'FP16 auto versus math changes kernel dispatch; FP16 math versus FP32 math changes model arithmetic precision. No claim of identical kernels across dtypes.'})
    notice({'phase':'loading_target_checkpoint','controls':len(examples)})
    model=cfg['models'][-1]
    path=snapshot_download(model['repo'],revision=model['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)
    results={};deadline=STARTED+cfg['worker_timeout_seconds']-90
    for arm in cfg['precision_arms']:
        if time.monotonic()>=deadline:break
        notice({'phase':'precision_control','arm':arm['name'],'dtype':arm['dtype'],'backend':arm['backend']})
        # Reload original pinned tensors: casting an already-rounded FP16 model
        # back to FP32 would not restore its original checkpoint values.
        runtime=DenseRuntime.from_local(path,path,device='cuda:0',dtype=arm['dtype'])
        runtime.prompt_token_limit=cfg['max_prompt_tokens']
        for e in examples:
            if len(runtime.tokenize(e))!=expected_tokens[e.id]:raise ValueError('Frozen control tokenization drifted')
        context=sdpa_kernel(SDPBackend.MATH)if arm['backend']=='math'else nullcontext()
        with ExitStack()as stack:
            stack.enter_context(context)
            if cfg.get('query_block_size'):
                from kvverify.blocked_attention import blocked_attention
                stack.enter_context(blocked_attention(cfg['query_block_size']))
            result=run_arm(runtime,examples,OUT/arm['name'],deadline,
                callback=lambda row:notice({'phase':'precision_control','arm':arm['name'],**row}))
        result['query_block_size']=cfg.get('query_block_size')
        results[arm['name']]=result
        write('summary.json',{'status':'in_progress','arms':results,'settings':settings})
        del runtime;gc.collect();torch.cuda.empty_cache()
    write('summary.json',{'status':'completed'if len(results)==len(cfg['precision_arms'])else'time_budget_reached',
        'arms':results,'settings':settings,'training_performed':False,
        'acceptance':'Numerical diagnostic only; no learned-cache quality or speed claim.'})
    write('worker-status.json',{'status':'completed_diagnostic','arms':{k:{n:v for n,v in r.items()if n!='rows'}for k,r in results.items()},
        'elapsed_worker_seconds':time.monotonic()-STARTED,'training_performed':False})
    notice({'phase':'completed_diagnostic'})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker time limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed','training_performed':False})
        try:upload()
        except Exception:traceback.print_exc()
        raise
