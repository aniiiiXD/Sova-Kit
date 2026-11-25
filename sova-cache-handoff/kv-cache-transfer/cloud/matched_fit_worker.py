"""Paired K/V affine refit on exact full chat/document histories."""
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import random
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
    from kvverify.affine_cache import AffineCacheMapper,CrossHeadMoments
    from kvverify.data import load_examples,assert_disjoint
    from kvverify.hf_backend import DenseRuntime
    from kvverify.qa_training import split_document_prompt,select_mapper_validation
    from kvverify.native_prefix import replace_prefix
    from kvverify.key_geometry import project_target_cache
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.state_diagnostic import distribution_metrics
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.blocked_attention import blocked_attention
    from cloud.repair_worker import split_train_articles
    from cloud.precision_worker import precision_settings
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='matched_fit'or not torch.cuda.is_available():raise ValueError('Managed CUDA full-history calibration required')
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg);torch.set_num_threads(4);precision_settings(torch)
    for name,sha in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Input changed')
    train,val=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),19)
    selected,_=select_mapper_validation(val,19)
    notice({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32')
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    parent=AffineCacheMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0').requires_grad_(False)
    if parent.manifest['weights_sha256']!=cfg['comparison_parent_weights_sha256']:raise ValueError('Comparison adapter changed')
    by_id={e.id:e for e in train};ids=parent.manifest['calibration_ids']
    if len(ids)!=115 or not set(ids)<=set(by_id):raise ValueError('Use exact prior115optimization articles')
    calibration=[by_id[i]for i in ids]
    if len({e.document_id for e in calibration})!=115:raise ValueError('Duplicate optimization article')
    assert_disjoint(calibration,selected)
    boundaries={e.id:split_document_prompt(runtime,e)for e in selected}
    if max(len(parts[0])for parts in boundaries.values())>8192:raise ValueError('No validation cropping')
    write('protocol.json',{'version':'paired-full-chat-history-affine-v1','calibration_ids':ids,
        'validation_ids':[e.id for e in selected],'alphas':cfg['alphas'],'key_policies':cfg['key_policies'],
        'history':'Exact complete chat/user/Document prefix from operational tokenizer boundary; source and target see every history token',
        'sampling':'Up to1024distinct seeded positions/article beyond retained native256; remove RoPE at original positions BEFORE sampling',
        'fit':'Separate centered all-head K and V affine banks, both refit on the same paired states; same aligned source layers',
        'native_prefix_policy':'256','optimizer_updates':0,'acceptance_inputs_bundled':False,
        'selection':'Minimum mean full-vocabulary native first-answer KL on original30development articles; gold labels diagnostic only',
        'comparison':'Prior generic250adapter plus native256runtime policy; refit changes calibration context/distribution, not model architecture'})
    deadline=STARTED+cfg['worker_timeout_seconds']-240
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Bounded matched-fit job window')
    with ExitStack()as stack:
        stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        moments=CrossHeadMoments(runtime.target_geometry,list(range(runtime.target_geometry[2])),'cuda:0')
        rng=random.Random(19);sampling=[]
        for index,e in enumerate(calibration):
            check();full,prefix,suffix=split_document_prompt(runtime,e)
            if len(full)>8192 or len(prefix)<=257:raise ValueError('Frozen calibration support changed; no cropping/exclusion')
            positions=sorted(rng.sample(range(256,len(prefix)),min(1024,len(prefix)-256)))
            source=runtime.forward(runtime.source,prefix).past_key_values
            target=runtime.forward(runtime.target,prefix).past_key_values
            moments.update(source,target,runtime.source,runtime.target,token_indices=positions);del source,target
            sampling.append({'id':e.id,'document_id':e.document_id,'prefix_tokens':len(prefix),'positions':positions})
            if(index+1)%5==0 or index+1==115:
                write('calibration-sampling.json',sampling);notice({'phase':'full_history_moments','document':index+1,'total':115,'sampled_states':moments.n})
        metadata={'source_sha256':runtime.identities['source']['sha256'],'target_sha256':runtime.identities['target']['sha256'],
            'tokenizer_sha256':runtime.tokenizer_sha256,'calibration_ids':ids,
            'calibration_documents':sorted({e.document_id for e in calibration}),
            'calibration_context_hashes':sorted({e.context_hash for e in calibration}),
            'selection_documents':sorted({e.document_id for e in selected}),
            'calibration_protocol':'paired_full_chat_history','calibration_prefix_limit':8192,
            'sampled_positions_per_document_max':1024,'native_prefix_policy':'256','new_optimizer_updates':0}
        models={'parent':parent}
        for alpha in cfg['alphas']:
            check();name=f'alpha_{alpha:g}';notice({'phase':'matched_ridge_solve','candidate':name,'sampled_states':moments.n})
            models[name]=moments.fit(alpha,True,metadata).to('cuda:0')
        sampled_states=moments.n;del moments;gc.collect();torch.cuda.empty_cache()
        specs=[{'name':family+'__'+policy,'family':family,'key_projection':policy}
            for family in models for policy in cfg['key_policies']]
        rows=[];ordered=sorted(selected,key=lambda e:(len(boundaries[e.id][0]),e.id));controls={ordered[i].id for i in(0,7,14,21,29)}
        for index,e in enumerate(selected):
            check();notice({'phase':'matched_candidate_validation','document':index+1,'total':30})
            full,prefix,suffix=boundaries[e.id]
            out=runtime.forward(runtime.target,full);native=out.logits[0,-1].cpu().clone();del out
            if e.id in controls:
                cache=runtime.forward(runtime.target,prefix).past_key_values;out=runtime.forward(runtime.target,suffix,cache)
                gate=probability_equivalence(torch,native,out.logits[0,-1].cpu());del out,cache
                if not gate['passed']:raise ValueError('Native full-history cache oracle failed')
            else:gate=None
            source=runtime.forward(runtime.source,prefix).past_key_values
            native_initial=runtime.forward(runtime.target,prefix[:256]).past_key_values
            tokens=option_token_ids(runtime.tokenizer,e.choices)
            row={'id':e.id,'document_id':e.document_id,'native':score_option_logits(native,tokens,e.answer),'native_control':gate,'variants':{}}
            vectors={'native':native}
            for spec in specs:
                check();cache=models[spec['family']].apply(source,runtime.source,runtime.target)
                if spec['key_projection']!='none':cache=project_target_cache(cache,runtime.target)
                cache=replace_prefix(cache,native_initial,256)
                out=runtime.forward(runtime.target,suffix,cache);logits=out.logits[0,-1].cpu().clone();del out,cache
                score=score_option_logits(logits,tokens,e.answer)
                row['variants'][spec['name']]={'score':score,'distribution':distribution_metrics(native,logits),
                    'native_agreement':score['decision_answer']==row['native']['decision_answer'],'key_projection':spec['key_projection']}
                vectors[spec['name']]=logits
            if source.get_seq_length()!=len(prefix)or native_initial.get_seq_length()!=256:raise ValueError('Shared immutable validation prefix mutated')
            del source,native_initial
            folder=OUT/'logits';folder.mkdir(exist_ok=True);save_file(vectors,str(folder/f'{index:02d}.safetensors'))
            rows.append(row);write('rows.json',rows)
        history=[]
        for step,spec in enumerate(specs):
            history.append({'step':step,'candidate_name':spec['name'],'key_projection':spec['key_projection'],'documents':30,
                'correct':sum(r['variants'][spec['name']]['score']['decision_correct']for r in rows),
                'native_correct':sum(r['native']['decision_correct']for r in rows),
                'native_agreement':sum(r['variants'][spec['name']]['native_agreement']for r in rows),
                'mean_native_kl':sum(r['variants'][spec['name']]['distribution']['native_to_candidate_kl_nats']for r in rows)/30})
        winner=min(history,key=lambda row:(row['mean_native_kl'],row['candidate_name']));spec=specs[winner['step']];model=models[spec['family']]
        model.manifest.update(selected_at_step=winner['step'],selection_axis='candidate_index',selected_candidate=spec['name'],
            key_projection=spec['key_projection'],native_prefix_policy='256',new_optimizer_updates=0,
            comparison_parent_weights_sha256=cfg['comparison_parent_weights_sha256'])
        checkpoint=Path('/opt/ml/checkpoints')/f'step-{winner["step"]:06d}';manifest=model.save(checkpoint);model.save(OUT/'best')
        write('best-selection.json',{**winner,'native_prefix_policy':'256','weights_sha256':manifest['weights_sha256']})
        write('validation-history.json',history)
        summary={'status':'completed','selected_step':winner['step'],'selected_candidate':spec['name'],
            'key_projection':spec['key_projection'],'native_prefix_policy':'256','validation_history':history,
            'calibration_articles':115,'sampled_states':sampled_states,'optimizer_updates':0,'acceptance_used':False,'serving_speed_claim':False}
        write('summary.json',summary);write('worker-status.json',summary);notice({'phase':'completed','selected':spec['name']})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());write('worker-status.json',{'status':'failed_or_incomplete'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
