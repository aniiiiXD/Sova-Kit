"""Fresh QuALITY task-aligned mapper; declared reference benchmarks and budgets."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import resource
from contextlib import ExitStack,nullcontext

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.probe_worker import OUT,STARTED,write,upload,notice
CONFIG={}


def freeze_base_models(runtime):
    """Freeze BEFORE constructing saved-tensor policy's storage allowlist."""
    for model in(runtime.source,runtime.target):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad=None


def split_train_articles(examples,seed=19,validation_documents=30):
    from kvverify.data import assert_disjoint
    groups={}
    for e in examples:
        if e.split!='calibration':raise ValueError('Only official train calibration records may be partitioned')
        groups.setdefault(e.document_id,[]).append(e)
    if len(groups)!=150 or len(examples)!=2523:raise ValueError('Unexpected frozen QuALITY training set')
    docs=sorted(groups);random.Random(seed).shuffle(docs)
    val=set(docs[:validation_documents])
    training=[e for e in examples if e.document_id not in val]
    validation=[e for e in examples if e.document_id in val]
    assert_disjoint(training,validation)
    return training,validation


def progress(row):
    # Linux ru_maxrss is KiB; the cloud worker is Linux. Only plain metadata is
    # retained, never tensors/graphs, so progress cannot pin GPU activations.
    row={**row,'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    path=row.get('checkpoint_path')or row.get('best_mapper')
    if path:
        source=Path(path);dest=Path('/opt/ml/checkpoints')/('best-final'if row.get('best_mapper')else source.name)
        if not dest.exists():shutil.copytree(source,dest)
    notice(row)
    if time.monotonic()-STARTED>CONFIG['worker_timeout_seconds']-100:
        raise TimeoutError('Recovery worker reserved final export window')


def native_control(runtime,examples,count=16,seed=19,evidence_dir=None):
    from kvverify.qa_training import select_mapper_validation
    from kvverify.qa_training import split_document_prompt
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.quality import option_token_ids,score_option_logits
    candidates,_=select_mapper_validation(examples,seed)
    eligible=[];excluded=[]
    for e in candidates:
        if len(runtime.tokenize(e))<=runtime.prompt_token_limit:eligible.append(e)
        else:excluded.append(e.id)
    if len(eligible)<count:raise ValueError('Insufficient eligible native-control articles')
    selected=random.Random(seed).sample(eligible,count)
    rows=[]
    for index,e in enumerate(selected):
        ids,prefix,suffix=split_document_prompt(runtime,e)
        row={'id':e.id,'document_id':e.document_id,'prompt_tokens':len(ids),'gold_answer':e.answer}
        out=runtime.forward(runtime.target,ids)
        reference=out.logits[0,-1].detach().float().cpu().clone()
        row.update(status='ok',**score_option_logits(reference,option_token_ids(runtime.tokenizer,e.choices),e.answer))
        del out
        # True target cache, no mapper/source: exact handoff used by training.
        # These questions have not been used by the preceding ContractNLI probe.
        cache=runtime.forward(runtime.target,prefix).past_key_values
        out=runtime.forward(runtime.target,suffix,cache)
        candidate=out.logits[0,-1].detach().float().cpu().clone()
        row['native_cache_oracle']=probability_equivalence(runtime.torch,reference,candidate)
        if evidence_dir is not None:
            from safetensors.torch import save_file
            evidence_dir.mkdir(parents=True,exist_ok=True)
            path=evidence_dir/f'{index:03d}.safetensors'
            save_file({'native':reference.contiguous(),'native_cache_oracle':candidate.contiguous()},str(path))
            row['logit_evidence']={'file':str(path.relative_to(OUT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        row.update(prefix_tokens=len(prefix),native_suffix_tokens=len(suffix))
        del out,cache
        rows.append(row)
    return {'selection':'Seeded mapper-validation articles, selected before outputs; no official dev tuning',
        'examples':len(rows),'decision_accuracy':sum(r.get('decision_correct',False)for r in rows)/len(rows),
        'rows':rows,'minimum_declared_accuracy':CONFIG['minimum_native_control_accuracy'],
        'excluded_over_limit_ids':excluded,'oracle_protocol':'oracle-probability-v2',
        'oracle_gate_passed':all(r['native_cache_oracle']['passed']for r in rows),
        'revision':'Posthoc replacement of historical raw-logit tolerance; must pass fresh operational-boundary controls before any training.'}


def main():
    global CONFIG
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.data import load_examples,assert_disjoint,check_heldout
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import NeuralMapper
    from kvverify.qa_training import train_qa_mapper
    from kvverify.reference_eval import load_reference_data,run_quality_reference,run_hellaswag_reference

    OUT.mkdir(parents=True,exist_ok=True)
    CONFIG=json.loads(Path(os.environ['KV_CONFIG']).read_text());write('config.json',CONFIG)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(4);torch.manual_seed(CONFIG['seed'])
    for name,value in CONFIG['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:raise ValueError('Frozen input hash mismatch: '+name)
    probe=json.loads((ROOT/'inputs/prior-probe-summary.json').read_text())
    if probe.get('protocol')!='tail-boundary-v1':raise ValueError('Historical diagnostic protocol missing')
    if CONFIG.get('require_fresh_native_oracle_gate')is not True:
        raise ValueError('Fresh native-cache parity gate cannot be disabled')
    if CONFIG.get('attention_backend')=='math':
        from cloud.recovery_submit import verify_precision_summary
        verify_precision_summary(ROOT/'inputs/prior-precision-summary.json')
    manifest_hash=CONFIG['input_sha256']['reference/manifest.json']
    quality,hellaswag,manifest=load_reference_data(ROOT/'reference',manifest_hash)
    source_training=load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration')
    train,validation=split_train_articles(source_training,CONFIG['seed'])
    assert_disjoint(train,quality);assert_disjoint(validation,quality)
    write('training-split.json',{'seed':CONFIG['seed'],'train_documents':sorted({e.document_id for e in train}),
        'validation_documents':sorted({e.document_id for e in validation}),
        'train_questions':len(train),'validation_questions':len(validation),
        'policy':'Official train articles partitioned 120/30 before length screening; every question pool retained.',
        'initialization':('Reuse validated untrained QuALITY step zero; no ridge refit; fresh optimizer'
            if CONFIG.get('use_step_zero_initialization')else
            'Fresh ridge on eligible official train documents, then new neural mapper; prior neural weights are not resumed.')})
    progress({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in CONFIG['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype=CONFIG['dtype'])
    freeze_base_models(runtime)
    runtime.prompt_token_limit=CONFIG['max_prompt_tokens']
    write('environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'identities':runtime.identities,'reference_manifest':manifest,
        'dtype':CONFIG['dtype'],'attention_backend':CONFIG.get('attention_backend','auto'),
        'query_block_size':CONFIG.get('query_block_size')})
    progress({'phase':'native_quality_control','examples':CONFIG['native_control_examples']})
    control=native_control(runtime,validation,CONFIG['native_control_examples'],CONFIG['seed'],OUT/'native-control-logits')
    write('native-control.json',control);upload()
    if not control['oracle_gate_passed']:
        write('worker-status.json',{'status':'stopped_native_cache_control','acceptance':'not_established',
            'reason':'Fresh probability-level native-cache parity failed; no mapper trained.',
            'native_control':control})
        progress({'phase':'stopped_native_cache_control'})
        return
    if control['decision_accuracy']<CONFIG['minimum_native_control_accuracy']:
        write('worker-status.json',{'status':'stopped_weak_native_baseline','acceptance':'not_established',
            'reason':'Predeclared native accuracy floor did not pass on mapper-validation articles; no mapper trained.',
            'native_control':control})
        progress({'phase':'stopped_weak_native_baseline'})
        return
    remaining=CONFIG['worker_timeout_seconds']-(time.monotonic()-STARTED)-1200
    if remaining<60:raise TimeoutError('Setup consumed training budget; refusing hidden extension')
    progress({'phase':'task_aligned_training','requested_updates':CONFIG['train_steps']})
    initial_mapper=None
    if CONFIG.get('use_step_zero_initialization'):
        directory=Path(CONFIG['input_mapper_dir'])
        initial_metadata=json.loads((directory/'manifest.json').read_text())
        canonical=json.dumps(initial_metadata,sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest()!=CONFIG['initial_mapper_manifest_sha256']:
            raise ValueError('Initialization snapshot manifest drifted')
        initial_mapper=NeuralMapper.load(directory,runtime.identities['source']['sha256'],
            runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
    if CONFIG.get('activation_offload_policy')=='selective_pageable':
        from kvverify.activation_offload import selective_saved_tensors_cpu
        offload=selective_saved_tensors_cpu([runtime.source,runtime.target],pin_memory=False)
        offload_description='Pageable CPU activations; frozen parameter storages stay resident on GPU'
    else:
        offload=torch.autograd.graph.save_on_cpu(pin_memory=True)if CONFIG.get('training_saved_tensors_cpu')else nullcontext()
        offload_description='CPU pinned memory'if CONFIG.get('training_saved_tensors_cpu')else'GPU'
    write('training-memory-policy.json',{
        'mapper_replay':'Per-layer/token-block checkpointing; no whole-mapper replay',
        'saved_backward_tensors':offload_description,
        'initialization':'Validated existing untrained step zero'if initial_mapper is not None else'Fresh ridge fit',
        'worst_case_backward_preflight':CONFIG.get('training_memory_preflight',False),
        'model_weights':'Remain frozen on GPU; no precision/context reduction',
        'requested_updates':CONFIG['train_steps'],'checkpoint_every':CONFIG.get('train_checkpoint_every',50)})
    # No-grad teacher/source/validation do not save autograd tensors. The outer
    # hook offloads student saved activations; bounded mapper checkpointing is
    # separately necessary because checkpoint's internal hooks take precedence.
    with offload as offload_telemetry:
        mapper=train_qa_mapper(runtime,train,validation,OUT/'training',steps=CONFIG['train_steps'],
            max_prompt_tokens=CONFIG['max_prompt_tokens'],answer_tokens=CONFIG['training_answer_tokens'],
            checkpoint_every=CONFIG.get('train_checkpoint_every',50),
            max_seconds=min(CONFIG['train_max_seconds'],remaining),callback=progress,
            checkpoint_mapper=True,memory_preflight=CONFIG.get('training_memory_preflight',False),
            initial_mapper=initial_mapper,seed=CONFIG['seed'])
    if isinstance(offload_telemetry,dict):write('activation-offload-traffic.json',offload_telemetry)
    del initial_mapper
    check_heldout(mapper.manifest,quality)
    runtime.mapper=None;del mapper;torch.cuda.empty_cache()
    mapper=NeuralMapper.load(OUT/'training/best',runtime.identities['source']['sha256'],
        runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0')
    runtime.mapper=mapper
    runtime.prompt_token_limit=CONFIG.get('eval_max_prompt_tokens',CONFIG['max_prompt_tokens'])
    remaining=CONFIG['worker_timeout_seconds']-(time.monotonic()-STARTED)-150
    if remaining<=0:raise TimeoutError('No benchmark budget after training')
    benchmark_deadline=time.monotonic()+min(CONFIG['benchmark_max_seconds'],remaining)
    progress({'phase':'reference_quality','planned_items':len(quality)})
    q=run_quality_reference(runtime,quality,mapper,OUT/'quality',
        freegen_max_new_tokens=CONFIG['quality_freegen_tokens'],
        max_seconds=max(1,min(CONFIG.get('quality_max_seconds',600),benchmark_deadline-time.monotonic())),callback=progress,seed=CONFIG['seed'])
    write('quality-summary.json',q);upload()
    progress({'phase':'reference_hellaswag','planned_items':len(hellaswag)})
    h=run_hellaswag_reference(runtime,hellaswag,mapper,OUT/'hellaswag',
        max_seconds=max(1,benchmark_deadline-time.monotonic()),callback=progress,seed=CONFIG['seed'])
    write('hellaswag-summary.json',h)
    write('worker-status.json',{'status':'completed'if q['fully_completed']and h['fully_completed']else'completed_with_budget_or_limit_outcomes',
        'acceptance':'not_established','quality':q,'hellaswag':h,'elapsed_worker_seconds':time.monotonic()-STARTED,
        'scope':'QuALITY task-trained small-pair development candidate; HellaSwag is out-of-domain mechanism evaluation, not a FineWeb paper reproduction.'})
    progress({'phase':'completed'})


def run_configured():
    """Keep the verified numerical path active for controls, training AND eval."""
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    with ExitStack()as stack:
        if cfg.get('attention_backend')=='math':
            import torch
            from torch.nn.attention import sdpa_kernel,SDPBackend
            from cloud.precision_worker import precision_settings
            from kvverify.blocked_attention import blocked_attention
            if cfg['dtype']!='float32'or cfg['query_block_size']!=256:
                raise ValueError('Unverified numerical execution path')
            precision_settings(torch)
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
            stack.enter_context(blocked_attention(256))
        return main()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker time limit')))
    try:run_configured()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed','acceptance':'not_established'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
