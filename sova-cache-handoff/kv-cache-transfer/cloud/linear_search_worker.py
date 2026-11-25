"""Controlled affine support/regularization sweep, optimization data only."""
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
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
    from kvverify.affine_cache import CrossHeadMoments
    from kvverify.data import load_examples,assert_disjoint
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import NeuralMapper
    from kvverify.qa_training import split_document_prompt,select_mapper_validation,group_questions
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.state_diagnostic import cpu_rows,gpu_cache,map_cpu_rows,distribution_metrics
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.blocked_attention import blocked_attention
    from cloud.precision_worker import precision_settings
    from cloud.repair_worker import split_train_articles
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='linear_search'or cfg['training_enabled']:raise ValueError('Affine calibration only')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no laptop fallback')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);precision_settings(torch)
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg)
    for name,expected in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen input changed: '+name)
    train,validation=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),cfg['seed'])
    notice({'phase':'loading_models','purpose':'controlled_linear_reference'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32');runtime.prompt_token_limit=8192
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    eligible=[e for e in train if len(runtime.tokenize(e))<=8192]
    calibrations=[questions[0]for questions in group_questions(eligible).values()]
    selected,_=select_mapper_validation(validation,cfg['seed'])
    if len(calibrations)!=115 or len(selected)!=30:raise ValueError('Frozen train/validation geometry drifted')
    assert_disjoint(calibrations,selected)
    old=NeuralMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0').requires_grad_(False)
    if old.manifest['weights_sha256']!=cfg['old_mapper_weights_sha256']:raise ValueError('Old comparison identity changed')
    lengths={e.id:len(split_document_prompt(runtime,e)[0])for e in selected}
    if max(lengths.values())>8192:raise ValueError('No silent validation exclusions')
    ordered=sorted(selected,key=lambda e:(lengths[e.id],e.id));control_ids={ordered[i].id for i in(0,5,10,15,20,29)}
    write('protocol.json',{'version':'affine-support-ridge-v1','calibration':'Exact old protocol:115eligible training articles, plain document first1024tokens, no question/gold used',
        'calibration_ids':[e.id for e in calibrations],'validation_ids':[e.id for e in selected],
        'validation_role':'development/model selection, not final test','alphas':cfg['alphas'],
        'regularization':'Centered XTX/n+alphaI; raw lambda=n*alpha',
        'source_layers':'same aligned depth for both support controls','forced_key_rms':False,
        'selection':'Minimum mean native full-vocabulary first-answer KL across all30validation articles; gold accuracy diagnostic only',
        'native_control_ids':sorted(control_ids),'training_updates':0,'no_final_holdout_inputs':True})
    deadline=STARTED+cfg['worker_timeout_seconds']-150
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Bounded affine-search deadline')
    with ExitStack()as stack:
        stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        moments=CrossHeadMoments(runtime.target_geometry,list(range(runtime.target_geometry[2])),'cuda:0')
        for index,e in enumerate(calibrations):
            check_time();ids=runtime.tokenizer(e.context,add_special_tokens=False)['input_ids'][:1024]
            source=runtime.forward(runtime.source,ids).past_key_values;target=runtime.forward(runtime.target,ids).past_key_values
            moments.update(source,target,runtime.source,runtime.target);del source,target
            if index%10==0 or index+1==len(calibrations):notice({'phase':'calibration_moments','documents':index+1,'total':len(calibrations),'tokens':moments.n})
        metadata={'source_sha256':runtime.identities['source']['sha256'],'target_sha256':runtime.identities['target']['sha256'],
            'tokenizer_sha256':runtime.tokenizer_sha256,'calibration_ids':[e.id for e in calibrations],
            'calibration_documents':sorted({e.document_id for e in calibrations}),
            'calibration_context_hashes':sorted({e.context_hash for e in calibrations}),
            'selection_documents':sorted({e.document_id for e in selected}),'calibration_prefix_limit':1024}
        candidates={}
        for mix in(False,True):
            for alpha in cfg['alphas']:
                check_time();name=('all_head'if mix else'same_head')+f'_alpha_{alpha:g}'
                notice({'phase':'ridge_solve','candidate':name,'calibration_tokens':moments.n})
                candidates[name]=moments.fit(alpha,mix,metadata)
        del moments;gc.collect();torch.cuda.empty_cache()
        rows=[]
        for index,e in enumerate(selected):
            check_time();notice({'phase':'candidate_validation','document':index+1,'total':30,'id':e.id})
            full,prefix,suffix=split_document_prompt(runtime,e)
            cache=runtime.forward(runtime.target,prefix).past_key_values
            out=runtime.forward(runtime.target,suffix,cache);native=out.logits[0,-1].detach().float().cpu().clone();del out,cache
            if e.id in control_ids:
                out=runtime.forward(runtime.target,full);unsegmented=out.logits[0,-1].detach().float().cpu().clone();del out
                gate=probability_equivalence(torch,unsegmented,native)
                if not gate['passed']:raise ValueError('Native numerical guard failed')
            else:gate=None
            source=runtime.forward(runtime.source,prefix).past_key_values;source_rows=cpu_rows(source);del source
            tokens=option_token_ids(runtime.tokenizer,e.choices)
            row={'id':e.id,'document_id':e.document_id,'context_hash':e.context_hash,'prompt_tokens':len(full),
                'native':score_option_logits(native,tokens,e.answer),'native_control':gate,'variants':{}}
            vectors={'native':native}
            for name,mapper in [*candidates.items(),('old_step48',old)]:
                check_time();mapper.to('cuda:0')
                mapped=map_cpu_rows(mapper,source_rows,runtime.source,runtime.target,
                    normalize_keys=name=='old_step48',block_size=256)
                cache=gpu_cache(mapped,'cuda:0');out=runtime.forward(runtime.target,suffix,cache)
                logits=out.logits[0,-1].detach().float().cpu().clone();del out,cache,mapped
                score=score_option_logits(logits,tokens,e.answer)
                row['variants'][name]={'score':score,'distribution':distribution_metrics(native,logits),
                    'native_choice_agreement':score['decision_answer']==row['native']['decision_answer']}
                vectors[name]=logits
                mapper.to('cpu')
            path=OUT/f'logits/{index:02d}.safetensors';path.parent.mkdir(exist_ok=True);save_file(vectors,str(path))
            row['logits_file']=path.relative_to(OUT).as_posix();row['logits_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row);write('rows.json',rows);del source_rows,vectors;gc.collect();torch.cuda.empty_cache()
        summaries={}
        for name in rows[0]['variants']:
            summaries[name]={'documents':len(rows),'accuracy':sum(r['variants'][name]['score']['decision_correct']for r in rows)/len(rows),
                'native_agreement':sum(r['variants'][name]['native_choice_agreement']for r in rows)/len(rows),
                'mean_native_kl':sum(r['variants'][name]['distribution']['native_to_candidate_kl_nats']for r in rows)/len(rows)}
        winner=min(candidates,key=lambda name:(summaries[name]['mean_native_kl'],name))
        for support in('same_head','all_head'):
            best=min((n for n in candidates if n.startswith(support)),key=lambda n:(summaries[n]['mean_native_kl'],n))
            candidates[best].manifest.update(selected_candidate=best,selection_metric='mean_native_first_answer_KL',selection_results=summaries[best])
            candidates[best].save(OUT/'selected'/support)
        result={'status':'completed','selected_candidate':winner,'selected_support':'all_head'if candidates[winner].manifest['head_mixing']else'same_head',
            'selection':'Development only; minimum mean native KL, no gold label selection','candidates':summaries,
            'native_accuracy':sum(r['native']['decision_correct']for r in rows)/len(rows),'validation_documents':len(rows),
            'calibration_documents':len(calibrations),'training_updates':0,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        write('summary.json',result);write('worker-status.json',result)
        notice({'phase':'completed','selected_candidate':winner,'documents':len(rows)})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker deadline')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json',{'status':'failed_or_incomplete'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
