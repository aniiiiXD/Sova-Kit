"""Bounded inference-only normalization and K/V readout experiment."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.probe_worker import OUT,STARTED,write,upload,notice


def checkpoint_manifests(cfg):
    result={}
    for name,directory in [('step0',cfg['input_initializer_dir']),('step48',cfg['input_mapper_dir'])]:
        path=Path(directory);raw=(path/'manifest.json').read_bytes();manifest=json.loads(raw)
        expected=cfg['checkpoints'][name]
        canonical=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest()!=expected['manifest_sha256']:
            raise ValueError('Pinned manifest changed: '+name)
        if hashlib.sha256((path/'weights.safetensors').read_bytes()).hexdigest()!=expected['weights_sha256']:
            raise ValueError('Pinned weights changed: '+name)
        if manifest['weights_sha256']!=expected['weights_sha256']or manifest['trained_steps']!=expected['trained_steps']:
            raise ValueError('Checkpoint metadata inconsistent: '+name)
        result[name]=manifest
    if result['step0']['mapper_validation_question_ids']!=result['step48']['mapper_validation_question_ids']:
        raise ValueError('Initializer and selected mapper validation registry differs')
    return result


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.train_mapper import NeuralMapper
    from kvverify.qa_training import select_mapper_validation,split_document_prompt
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.blocked_attention import blocked_attention
    from kvverify.state_diagnostic import (select_panel,cpu_rows,gpu_cache,map_cpu_rows,
        capture_native_queries,layer_interventions,distribution_metrics,token_hash)
    from cloud.precision_worker import precision_settings
    from cloud.repair_worker import split_train_articles

    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='state_diagnostic'or cfg['training_enabled']or cfg['dtype']!='float32':
        raise ValueError('This diagnostic permits only FP32 inference, no training')
    if not torch.cuda.is_available():raise RuntimeError('CUDA worker required; no laptop fallback')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg)
    for name,expected in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen input changed: '+name)
    manifests=checkpoint_manifests(cfg)
    _,validation=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),cfg['seed'])
    candidates,_=select_mapper_validation(validation,cfg['seed'])
    if [e.id for e in candidates]!=manifests['step0']['mapper_validation_question_ids']:
        raise ValueError('Original validation selection drifted')
    notice({'phase':'loading_pinned_models','training_enabled':False})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32')
    runtime.prompt_token_limit=cfg['max_prompt_tokens']
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    lengths={e.id:len(split_document_prompt(runtime,e)[0])for e in candidates}
    if max(lengths.values())>cfg['max_prompt_tokens']:raise ValueError('Frozen validation panel exceeds unchanged length cap')
    panel=select_panel(candidates,lengths,seed=cfg['seed']);by_id={e.id:e for e in candidates}
    write('panel.json',{'selection':'24 of original30 validation articles; three token-length bins, eight per bin; seed19; outcome-blind',
        'seed':cfg['seed'],'rows':panel,'development_only':True,'training_performed':False})
    settings=precision_settings(torch)
    write('environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'model_identities':runtime.identities,'settings':settings})
    write('protocol.json',{'name':'state-diagnostic-v2','native_readout_atol':2e-5,'native_readout_rtol':2e-4,
        'native_oracle':'oracle-probability-v2','query_positions':'first, middle, final suffix positions',
        'native_query_interventions':'oracle diagnostic; native suffix K/V always preserved',
        'full_stack':'actual mapped cache plus complete native question suffix; first-answer distribution',
        'first_token_only':True,'free_generation_evaluated':False,'serving_latency_claim':False,
        'readout_reference':'Captured actual SDPA inputs, identical full suffix/mask/scale shapes; sample only after SDPA and full-shape native projection',
        'attention_probability_statistics':'Explicit selected-query FP32 softmax; readout errors use native math SDPA',
        'selection_frozen_before_outputs':True,'checkpoint_metadata':cfg['checkpoints']})
    mappers={}
    for name,path in [('step0',cfg['input_initializer_dir']),('step48',cfg['input_mapper_dir'])]:
        mapper=NeuralMapper.load(path,runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256)
        mapper.requires_grad_(False);mappers[name]=mapper.to('cuda:0')
    rows=[];deadline=STARTED+cfg['worker_timeout_seconds']-90
    def require_time():
        if time.monotonic()>=deadline:raise TimeoutError('Diagnostic time budget; no partial success')
    with ExitStack()as stack:
        stack.enter_context(torch.inference_mode())
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        stack.enter_context(blocked_attention(cfg['query_block_size']))
        for index,selection in enumerate(panel):
            require_time();e=by_id[selection['id']]
            notice({'phase':'native_control','document':index+1,'total':len(panel),'id':e.id})
            ids,prefix,suffix=split_document_prompt(runtime,e)
            out=runtime.forward(runtime.target,ids);native_logits=out.logits[0,-1].detach().float().cpu().clone();del out
            cache=runtime.forward(runtime.target,prefix).past_key_values
            with capture_native_queries(runtime.target)as queries:
                out=runtime.forward(runtime.target,suffix,cache)
            segmented_logits=out.logits[0,-1].detach().float().cpu().clone()
            native_rows=cpu_rows(out.past_key_values)
            del out,cache
            oracle=probability_equivalence(torch,native_logits,segmented_logits)
            if not oracle['passed']:raise ValueError('Unchanged native-cache numerical control failed: '+e.id)
            source=runtime.forward(runtime.source,prefix).past_key_values
            source_rows=cpu_rows(source);del source
            options=option_token_ids(runtime.tokenizer,e.choices)
            base={'id':e.id,'document_id':e.document_id,'context_hash':e.context_hash,
                'length_band':selection['length_band'],'prompt_tokens':len(ids),'prefix_tokens':len(prefix),
                'native_suffix_tokens':len(suffix),'token_ids_sha256':token_hash(ids),'status':'ok'}
            native_score=score_option_logits(native_logits,options,e.answer)
            row={**base,'native_control':oracle,'native_score':native_score,'variants':{}}
            logits_evidence={'native':native_logits,'native_cache_oracle':segmented_logits}
            for name,mapper in mappers.items():
                for normalize in(True,False):
                    require_time();arm=name+('_rms_on'if normalize else'_rms_off')
                    notice({'phase':'cache_interventions','document':index+1,'total':len(panel),'arm':arm,'id':e.id})
                    mapped=map_cpu_rows(mapper,source_rows,runtime.source,runtime.target,
                        normalize_keys=normalize,block_size=cfg['query_block_size'])
                    readouts=layer_interventions(runtime.target,native_rows,mapped,queries,len(prefix),deadline=deadline)
                    require_time();cache=gpu_cache(mapped,'cuda:0')
                    with capture_native_queries(runtime.target)as mapped_queries:
                        out=runtime.forward(runtime.target,suffix,cache)
                    logits=out.logits[0,-1].detach().float().cpu().clone()
                    del out,cache
                    for layer,measurement in enumerate(readouts):
                        q=queries[layer]['q'];mq=mapped_queries[layer]['q']
                        measurement['actual_transfer_query_relative_error']=((mq-q).norm(dim=-1)/q.norm(dim=-1).clamp_min(1e-8)).tolist()
                    filename=f'documents/{index:02d}/{arm}-readouts.json'
                    write(filename,{'id':e.id,'arm':arm,'layers':readouts})
                    score=score_option_logits(logits,options,e.answer)
                    row['variants'][arm]={'score':score,'distribution':distribution_metrics(native_logits,logits),
                        'native_choice_agreement':score['decision_answer']==native_score['decision_answer'],
                        'readouts_file':filename,'normalize_keys':normalize,'checkpoint':name}
                    logits_evidence[arm]=logits
                    del mapped,readouts,mapped_queries
            path=OUT/f'documents/{index:02d}/logits.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
            save_file(logits_evidence,str(path))
            row['logits_file']=path.relative_to(OUT).as_posix();row['logits_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row);write('rows.json',rows)
            write('summary.json',{'status':'in_progress','completed_documents':len(rows),'expected_documents':len(panel),
                'training_performed':False,'acceptance':'diagnostic_only_not_quality_or_speed_acceptance'})
            del native_rows,source_rows,queries,logits_evidence
            gc.collect();torch.cuda.empty_cache()
    # Also validate the checkpoint bytes after inference; no optimizer exists.
    checkpoint_manifests(cfg)
    summary={'status':'completed_diagnostic','documents':len(rows),'expected_documents':24,
        'native_controls_passed':sum(r['native_control']['passed']for r in rows),'variants':{},
        'training_performed':False,'serving_latency_claim':False,'elapsed_worker_seconds':time.monotonic()-STARTED,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'acceptance':'exploratory_validation_diagnostic_not_final_test'}
    for arm in rows[0]['variants']:
        summary['variants'][arm]={'completed':len(rows),
            'accuracy':sum(r['variants'][arm]['score']['decision_correct']for r in rows)/len(rows),
            'native_choice_agreement':sum(r['variants'][arm]['native_choice_agreement']for r in rows)/len(rows),
            'mean_native_to_candidate_kl_nats':sum(r['variants'][arm]['distribution']['native_to_candidate_kl_nats']for r in rows)/len(rows)}
    summary['native_accuracy']=sum(r['native_score']['decision_correct']for r in rows)/len(rows)
    write('summary.json',summary);write('worker-status.json',summary);notice({'phase':'completed_diagnostic','documents':len(rows)})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker time limit')))
    try:main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        if hasattr(exc,'diagnostics'):write('guard-failure.json',exc.diagnostics)
        write('worker-status.json',{'status':'failed_or_incomplete','training_performed':False,'elapsed_worker_seconds':time.monotonic()-STARTED})
        try:upload()
        except Exception:traceback.print_exc()
        raise
