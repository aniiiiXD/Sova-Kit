"""One frozen candidate on previously unscored source-disjoint public records."""
from contextlib import ExitStack
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


def economics_qualification(quality,prefix_competitor=None):
    """Prospective engineering gate; no statistical non-inferiority claim."""
    variants=quality.get('variants',{});native=variants.get('native',{});mapped=variants.get('mapper',{})
    complete=quality.get('fully_completed')is True and all(
        row.get('planned_items')==267 and row.get('completed_items')==267 and row.get('errors')==0
        for row in variants.values())and set(variants)=={'source','native','mapper'}
    native_acc=native.get('decision_correct',0.);mapped_acc=mapped.get('decision_correct',0.)
    agreement=mapped.get('native_agreement_decision_answer',0.)
    competitor_pass=True if prefix_competitor is None else(
        prefix_competitor.get('completed_items')==267 and prefix_competitor.get('errors')==0
        and mapped_acc>prefix_competitor.get('decision_correct',1.))
    return {'passed':complete and competitor_pass and native_acc>=.4 and mapped_acc>=.9*native_acc and agreement>=.8
                and mapped.get('completed_transfer_items')==267 and mapped.get('completed_native_fallback_items')==0,
            'native_accuracy':native_acc,'candidate_accuracy':mapped_acc,'native_agreement':agreement,
            'prefix_only_competitor':prefix_competitor,'candidate_beats_prefix_only':competitor_pass,
            'scope':'267long-document questions,15articles; no fallback transfer credit',
            'thresholds':'Complete,error-free; native>=40%; candidate>=90%ofnative; native agreement>=80%',
            'limitation':'Engineering timing qualification only; report grouped uncertainty separately'}


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from huggingface_hub import snapshot_download
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.qa_training import split_document_prompt
    from kvverify.reference_eval import run_quality_reference,run_hellaswag_reference,validate_hellaswag_record
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.transfer_economics import run_economics
    from kvverify.prefix_competitor import run_prefix_competitor
    from kvverify.blocked_attention import blocked_attention
    from cloud.precision_worker import precision_settings
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='holdout'or cfg['training_enabled']:raise ValueError('Frozen inference-only acceptance worker')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg)
    for n,sha in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=sha:raise ValueError('Acceptance input changed: '+n)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);precision_settings(torch)
    quality=load_examples(ROOT/'acceptance/quality.jsonl','eval')
    hellaswag=[json.loads(x)for x in(ROOT/'acceptance/hellaswag.jsonl').read_text().splitlines()if x.strip()]
    if len(quality)!=267 or len({e.document_id for e in quality})!=15 or len(hellaswag)!=200 or len({r['source_id']for r in hellaswag})!=200:
        raise ValueError('Frozen acceptance counts/source groups differ')
    for row in hellaswag:validate_hellaswag_record(row)
    notice({'phase':'loading_frozen_candidate','training_enabled':False})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32');runtime.prompt_token_limit=12000
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    mapper=AffineCacheMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0').requires_grad_(False)
    if mapper.manifest['weights_sha256']!=cfg['mapper_weights_sha256']:raise ValueError('Acceptance candidate changed')
    if mapper.manifest.get('native_prefix_policy','none')!=cfg['native_prefix_policy']:raise ValueError('Acceptance native-prefix policy differs')
    if mapper.manifest.get('key_projection','none')!=cfg['key_projection']:raise ValueError('Acceptance key-projection policy differs')
    used=set(mapper.manifest.get('calibration_documents',[]))|set(mapper.manifest.get('training_documents',[]))|set(mapper.manifest.get('selection_documents',[]))
    if used&{e.document_id for e in quality}:raise ValueError('Acceptance source appears in optimization or selection registry')
    tokenized={e.id:split_document_prompt(runtime,e)for e in quality}
    largest=max(len(parts[0])for parts in tokenized.values())
    if largest>12000:raise ValueError('Fixed acceptance inputs exceed predeclared support; no cropping or silent exclusion')
    ordered=sorted(quality,key=lambda e:(len(tokenized[e.id][0]),e.id))
    controls=[ordered[i]for i in(0,len(ordered)//4,len(ordered)//2,3*len(ordered)//4,len(ordered)-1)]
    write('protocol.json',{'version':'source-disjoint-public-acceptance-v1','quality_questions':267,'quality_articles':15,
        'hellaswag_records':200,'hellaswag_source_groups':200,'prompt_cap':12000,'max_actual_prompt_tokens':largest,
        'candidate_frozen_before_loading_inputs':True,'training_performed':False,'serving_speed_claim':False,
        'native_prefix_policy':cfg['native_prefix_policy'],
        'key_projection':cfg['key_projection'],
        'short_prefix_behavior':'For256policy,prefixes<=256use full native target; explicitly reported as fallback, not evidence of transfer',
        'quality':'Custom constrained-choice +16token generation, with separately reported metrics',
        'hellaswag':'Pinned full-ending conditional likelihood and character-normalized accuracy',
        'uncertainty':'Paired document/source-group bootstrap; all frozen items retained in denominator'})
    with ExitStack()as stack:
        stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        gates=[]
        for e in controls:
            ids,prefix,suffix=tokenized[e.id]
            out=runtime.forward(runtime.target,ids);native=out.logits[0,-1].detach().float().cpu().clone();del out
            cache=runtime.forward(runtime.target,prefix).past_key_values
            out=runtime.forward(runtime.target,suffix,cache);segmented=out.logits[0,-1].detach().float().cpu().clone();del out,cache
            gate=probability_equivalence(torch,native,segmented);gates.append({'id':e.id,'tokens':len(ids),'oracle':gate})
            if not gate['passed']:write('native-controls.json',gates);raise ValueError('Acceptance native-cache guard failed')
        write('native-controls.json',gates);notice({'phase':'quality_acceptance','items':267})
        remaining=cfg['worker_timeout_seconds']-(time.monotonic()-STARTED)-3600
        if remaining<600:raise TimeoutError('Insufficient declared complete-evaluation budget')
        q=run_quality_reference(runtime,quality,mapper,OUT/'quality',freegen_max_new_tokens=16,max_seconds=min(6500,remaining),callback=notice,seed=cfg['seed'])
        write('quality-summary.json',q)
        prefix_competitor=None
        if cfg['native_prefix_policy']=='256':
            native_answers={r['id']:r['decision_answer']for r in(
                json.loads(line)for line in(OUT/'quality/rows.jsonl').read_text().splitlines())
                if r['variant']=='native'and r['status']=='ok'}
            prefix_competitor=run_prefix_competitor(runtime,quality,OUT/'prefix-only-competitor',
                native_answers=native_answers,max_seconds=300,callback=notice)
            write('prefix-only-summary.json',prefix_competitor)
        notice({'phase':'hellaswag_acceptance','items':200})
        remaining=cfg['worker_timeout_seconds']-(time.monotonic()-STARTED)-150
        h=run_hellaswag_reference(runtime,hellaswag,mapper,OUT/'hellaswag',max_seconds=max(1,min(1500,remaining)),callback=notice,seed=cfg['seed'])
        write('hellaswag-summary.json',h)
        qualification=economics_qualification(q,prefix_competitor);write('economics-qualification.json',qualification)
        economics={'status':'not_qualified','qualification':qualification}
        remaining=cfg['worker_timeout_seconds']-(time.monotonic()-STARTED)-180
        if qualification['passed']and remaining>=600:
            try:
                notice({'phase':'economics','documents':12,'repeats':5})
                economics=run_economics(runtime,mapper,quality,OUT/'economics',example_count=12,repeats=5,
                    max_seconds=min(1800,remaining),seed=cfg['seed'],callback=notice)
            except Exception:
                (OUT/'economics-failure.txt').write_text(traceback.format_exc())
                economics={'status':'failed','reason':'Read raw failure and partial timings; no speed headline permitted'}
        elif qualification['passed']:economics={'status':'insufficient_remaining_budget','remaining_seconds':remaining}
        write('economics-summary.json',economics)
        summary={'status':'completed'if q['fully_completed']and h['fully_completed']else'incomplete_evaluation',
            'quality':q,'prefix_only_competitor':prefix_competitor,'hellaswag':h,'economics':economics,'mapper_weights_sha256':cfg['mapper_weights_sha256'],
            'elapsed_worker_seconds':time.monotonic()-STARTED,'training_performed':False,'serving_speed_claim':False}
        write('summary.json',summary);write('worker-status.json',summary);notice({'phase':summary['status']})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());write('worker-status.json',{'status':'failed_or_incomplete'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
