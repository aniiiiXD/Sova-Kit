"""Bounded generic soft distillation from a verified selected affine mapper."""
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloud.probe_worker import OUT,STARTED,write,upload,notice


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.continuation_training import ContinuationSampler,continuation_loss
    from kvverify.task_distillation import question_readout_loss
    from kvverify.native_prefix import header_tokens,replace_prefix
    from kvverify.activation_offload import selective_saved_tensors_cpu
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import bounded_optimizer_update
    from kvverify.qa_training import split_document_prompt,select_mapper_validation,group_questions,DocumentQuestionSampler
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.state_diagnostic import distribution_metrics
    from kvverify.blocked_attention import blocked_attention
    from cloud.precision_worker import precision_settings
    from cloud.repair_worker import split_train_articles
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='continuation'or not cfg['training_enabled']:raise ValueError('Explicit continuation-training worker required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no laptop model work')
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg)
    for n,sha in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=sha:raise ValueError('Input changed: '+n)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);precision_settings(torch)
    train,val=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),cfg['seed'])
    selected,_=select_mapper_validation(val,cfg['seed'])
    notice({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32')
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    mapper=AffineCacheMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256)
    if mapper.manifest['weights_sha256']!=cfg['initial_weights_sha256']:raise ValueError('Selected initialization hash changed')
    mapper.to('cuda:0').requires_grad_(True)
    prompt_lengths={e.id:len(runtime.tokenize(e))for e in train}
    eligible=[e for e in train if prompt_lengths[e.id]<=8192]
    documents={d:runtime.tokenizer(q[0].context,add_special_tokens=False)['input_ids']for d,q in group_questions(eligible).items()}
    sampler=ContinuationSampler(documents,cfg.get('sampling_seed',cfg['seed']),32)
    objective=cfg.get('training_objective','continuation')
    if objective not in('continuation','question_readout'):raise ValueError('Unknown distillation objective')
    policy=cfg.get('native_prefix_policy','none')
    if policy not in('none','header','64','256')or(objective=='continuation'and policy!='none'):
        raise ValueError('Unsupported explicit native-prefix training policy')
    question_sampler=DocumentQuestionSampler(eligible,cfg.get('sampling_seed',cfg['seed']))
    def prefix_count(example):
        return header_tokens(runtime,example)if policy=='header'else(0 if policy=='none'else int(policy))
    def task_loss(example):
        full,prefix,suffix=split_document_prompt(runtime,example)
        return question_readout_loss(runtime,mapper,full,prefix,suffix,native_prefix_tokens=prefix_count(example))
    if len(documents)!=115:raise ValueError('Optimization document pool drifted')
    boundaries={e.id:split_document_prompt(runtime,e)for e in selected}
    if max(len(x[0])for x in boundaries.values())>8192:raise ValueError('Validation cap changed')
    write('protocol.json',{'version':'continuation-soft-KL-v1'if objective=='continuation'else'question-readout-soft-KL-v1','training_documents':sorted(documents),
        'selection_ids':[e.id for e in selected],'native_prefix_choices':[512,1024,1024,1024,2048,2048,4096],
        'tail_tokens':32 if objective=='continuation'else 1,
        'objective':('Pure forward KL over32native-target continuation distributions; no hard CE or QA gold loss'if objective=='continuation'else
            'Pure full-vocabulary forward KL at native first-answer readout; exact full document prefix and fresh question suffix; no gold labels or hard CE'),
        'native_prefix_policy':policy,'native_prefix_cost_note':'Any native prefix is partial target recomputation, not pure cache translation',
        'bridge':('Map prefix except its last token; target consumes that last token plus31tail tokens at identical causal positions'if objective=='continuation'else
            'Identical exact tokenization: translated document prefix plus target-native question/template suffix'),
        'sampling':('Document-uniform permutation cycles; random contiguous text windows'if objective=='continuation'else
            'Document-uniform permutation cycles; cycle every distinct question per document before reuse'),
        'selection':'Minimum mean native first-answer KL across original30held-out selection articles',
        'acceptance_data_bundled':False,'requested_updates':cfg['steps'],'learning_rate':cfg['learning_rate'],
        'memory':'FP32/math attention,256query blocks, per-layer/token checkpoint replay, selective pageable activation offload'})
    deadline=STARTED+cfg['worker_timeout_seconds']-240
    native_logits={};best={'kl':float('inf'),'state':None,'step':0};history=[]
    def check_time(reserve=0):
        if time.monotonic()>=deadline-reserve:raise TimeoutError('Bounded continuation worker time limit')
    def validate(step):
        check_time();mapper.eval();rows=[];vectors={}
        with torch.no_grad():
            for e in selected:
                check_time();full,prefix,suffix=boundaries[e.id]
                if e.id not in native_logits:
                    out=runtime.forward(runtime.target,full);native_logits[e.id]=out.logits[0,-1].detach().float().cpu().clone();del out
                source=runtime.forward(runtime.source,prefix).past_key_values
                cache=mapper.apply(source,runtime.source,runtime.target);del source
                count=prefix_count(e)
                if count:
                    native_prefix=runtime.forward(runtime.target,prefix[:count]).past_key_values
                    cache=replace_prefix(cache,native_prefix,count);del native_prefix
                out=runtime.forward(runtime.target,suffix,cache);logits=out.logits[0,-1].detach().float().cpu().clone();del out,cache
                metrics=distribution_metrics(native_logits[e.id],logits);tokens=option_token_ids(runtime.tokenizer,e.choices)
                score=score_option_logits(logits,tokens,e.answer);native_score=score_option_logits(native_logits[e.id],tokens,e.answer)
                rows.append({'id':e.id,'document_id':e.document_id,'native_prefix_tokens':count,'score':score,'native_score':native_score,'distribution':metrics,
                    'native_agreement':score['decision_answer']==native_score['decision_answer']});vectors[e.id]=logits
        value=sum(r['distribution']['native_to_candidate_kl_nats']for r in rows)/len(rows)
        result={'step':step,'mean_native_kl':value,'correct':sum(r['score']['decision_correct']for r in rows),
            'native_correct':sum(r['native_score']['decision_correct']for r in rows),'native_agreement':sum(r['native_agreement']for r in rows),
            'documents':len(rows),'elapsed_worker_seconds':time.monotonic()-STARTED}
        if value<best['kl']:
            best.update(kl=value,step=step,state={k:v.detach().cpu().clone()for k,v in mapper.state_dict().items()})
            checkpoint=Path('/opt/ml/checkpoints')/f'step-{step:06d}';mapper.manifest.update(trained_steps=step,selected_at_step=step,
                training_objective=objective+'_soft_KL_no_QA_gold',native_prefix_policy=policy,
                initial_weights_sha256=cfg['initial_weights_sha256'],initial_ancestor_updates=cfg.get('initial_ancestor_updates',0),
                total_weight_updates=cfg.get('initial_ancestor_updates',0)+step,counter_scope='current stage; total_weight_updates includes selected ancestors')
            mapper.save(checkpoint)
            write('best-selection.json',{**result,'checkpoint_path':str(checkpoint),'native_prefix_policy':policy,
                'weights_sha256':json.loads((checkpoint/'manifest.json').read_text())['weights_sha256']})
        history.append(result);write('validation-history.json',history);write(f'validation/step-{step:06d}.json',rows)
        save_file(vectors,str(OUT/f'validation/step-{step:06d}.safetensors'))
        notice({'phase':'validation',**result,'best_step':best['step'],'best_kl':best['kl']});mapper.train()
    with ExitStack()as stack:
        stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        offload=stack.enter_context(selective_saved_tensors_cpu([runtime.source,runtime.target],pin_memory=False))
        optimizer=torch.optim.AdamW(mapper.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        longest=max(documents,key=lambda d:len(documents[d]));ids=documents[longest]
        count=min(4096,len(ids)-32)if objective=='continuation'else max(prompt_lengths[e.id]for e in eligible)
        snapshots={k:v.detach().cpu().clone()for k,v in mapper.state_dict().items()}
        notice({'phase':'worst_case_backward','prefix_tokens':count,'optimizer_updates':0})
        if objective=='continuation':loss,preflight=continuation_loss(runtime,mapper,ids[:count],ids[count:count+32])
        else:loss,preflight=task_loss(max(eligible,key=lambda e:prompt_lengths[e.id]))
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all()for p in mapper.parameters()):raise ValueError('Nonfinite preflight mapper gradients')
        if any(p.grad is not None for m in(runtime.source,runtime.target)for p in m.parameters()):raise ValueError('Frozen model received gradients')
        if any(not torch.equal(v.detach().cpu(),snapshots[k])for k,v in mapper.state_dict().items()):raise ValueError('Preflight mutated weights')
        optimizer.zero_grad(set_to_none=True);del loss,snapshots
        write('memory-preflight.json',{'status':'passed','prefix_tokens':count,'optimizer_updates':0,'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20})
        validate(0);completed=0;last=time.monotonic();train_start=last;consumed={};tokens=0
        for step in range(1,cfg['steps']+1):
            if time.monotonic()>min(deadline-150,train_start+cfg['train_max_seconds']):break
            if objective=='continuation':
                document,prefix,tail,start=sampler.next()
                metrics=bounded_optimizer_update(lambda:continuation_loss(runtime,mapper,prefix,tail),mapper,optimizer,scaler)
                extra={'window_start':start}
            else:
                example=question_sampler.next();document=example.document_id
                metrics=bounded_optimizer_update(lambda:task_loss(example),mapper,optimizer,scaler)
                extra={'question_id':example.id}
            completed=step;consumed[document]=consumed.get(document,0)+1;tokens+=metrics['supervised_tokens']
            row={'step':step,'document_id':document,**extra,'unique_documents':len(consumed),
                'total_supervised_tokens':tokens,**metrics}
            with (OUT/'training.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
            if time.monotonic()-last>=30:
                notice({'phase':'training',**row,'elapsed_worker_seconds':time.monotonic()-STARTED,
                    'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024});last=time.monotonic()
            if step%cfg['validate_every']==0 or step==cfg['steps']:validate(step)
        if completed%cfg['validate_every']and completed!=cfg['steps']:validate(completed)
        mapper.load_state_dict(best['state']);mapper.eval();mapper.manifest.update(trained_steps=completed,selected_at_step=best['step'],
            training_objective=objective+'_soft_KL_no_QA_gold',native_prefix_policy=policy,
            training_documents=sorted(documents),selection_documents=sorted({e.document_id for e in selected}),
            supervised_tokens=tokens,unique_training_documents=len(consumed),best_selection_kl=best['kl'],
            total_weight_updates=cfg.get('initial_ancestor_updates',0)+best['step'])
        mapper.save(OUT/'best')
        write('summary.json',{'status':'completed'if completed==cfg['steps']else'time_budget_reached',
            'completed_updates':completed,'requested_updates':cfg['steps'],'selected_step':best['step'],'best_selection_kl':best['kl'],
            'training_objective':objective,'native_prefix_policy':policy,
            'unique_training_documents':len(consumed),'supervised_tokens':tokens,'validation_history':history,
            'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'elapsed_worker_seconds':time.monotonic()-STARTED,'acceptance':'development_only_no_final_quality_or_speed_claim'})
        write('activation-offload.json',offload);write('worker-status.json',{'status':'completed_training','updates':completed,'best_step':best['step']})
        notice({'phase':'completed_training','updates':completed,'selected_step':best['step'],'best_kl':best['kl']})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker deadline')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed_or_incomplete'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
