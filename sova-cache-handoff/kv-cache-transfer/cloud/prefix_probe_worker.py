"""Development-only partial native-prefix control on a frozen checkpoint."""
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


def main():
    import torch
    from transformers import DynamicCache
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.native_prefix import header_tokens,replace_prefix,oracle_components
    from kvverify.prefix_competitor import run_prefix_competitor
    from kvverify.key_geometry import project_cache_cpu_rows
    from kvverify.qa_training import split_document_prompt,select_mapper_validation
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.state_diagnostic import cpu_rows,gpu_cache,map_cpu_rows,distribution_metrics
    from kvverify.blocked_attention import blocked_attention
    from cloud.precision_worker import precision_settings
    from cloud.repair_worker import split_train_articles
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['training_enabled']or cfg['mode']!='prefix_probe':raise ValueError('Inference-only development control')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg);torch.set_num_threads(4);precision_settings(torch)
    for name,sha in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Input changed')
    _,validation=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),19)
    selected,_=select_mapper_validation(validation,19)
    notice({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32')
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    mapper=AffineCacheMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0').requires_grad_(False)
    if mapper.manifest['weights_sha256']!=cfg['mapper_weights_sha256']:raise ValueError('Frozen probe checkpoint differs')
    components=cfg.get('component_oracles',False)
    arms=[('native256',256),('native_all_keys',256),('native_all_values',256),('both_native',256)]if components else[('translated_all',0),('native_header',None),('native64',64),('native256',256)]
    if cfg.get('project_keys'):arms=[('native256',256),('key_ellipsoid_native256',256),('both_native',256)]
    write('protocol.json',{'version':'component-oracle-intervention-v1'if components else'native-prefix-intervention-v1','selection_documents':30,
        'arms':[name for name,_ in arms],'training_updates':0,
        'note':('Full native K/V substitutions are offline ORACLES, not deployable fixes. Retained native256prefix is identical across arms. Both-native must pass original probability guard; no speed claim.'if components else
            'Partial target-native prefix recomputation, not pure/prefill-free translation. All target-native token counts/fractions are reported. Development only; no acceptance input.')})
    rows=[];deadline=STARTED+cfg['worker_timeout_seconds']-90
    with ExitStack()as stack:
        stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        for index,e in enumerate(selected):
            if time.monotonic()>=deadline:raise TimeoutError('No partial probe success')
            notice({'phase':'prefix_probe','document':index+1,'total':30})
            full,prefix,suffix=split_document_prompt(runtime,e);header=header_tokens(runtime,e)
            out=runtime.forward(runtime.target,full);native=out.logits[0,-1].detach().float().cpu().clone();del out
            source=runtime.forward(runtime.source,prefix).past_key_values;source_rows=cpu_rows(source);del source
            mapped=map_cpu_rows(mapper,source_rows,runtime.source,runtime.target,normalize_keys=False);del source_rows
            projected=project_cache_cpu_rows(mapped,runtime.target)if cfg.get('project_keys')else None
            target_header=runtime.forward(runtime.target,prefix if components else prefix[:256]).past_key_values;native_rows=cpu_rows(target_header);del target_header
            row={'id':e.id,'document_id':e.document_id,'prefix_tokens':len(prefix),'header_tokens':header,
                'header_ids':prefix[:header],'native':score_option_logits(native,option_token_ids(runtime.tokenizer,e.choices),e.answer),'variants':{}}
            vectors={'native':native}
            for arm,count in arms:
                count=header if count is None else count
                base=DynamicCache.from_legacy_cache(projected if arm=='key_ellipsoid_native256'else mapped)
                if components:base=oracle_components(base,DynamicCache.from_legacy_cache(native_rows),keys=arm in('native_all_keys','both_native'),
                    values=arm in('native_all_values','both_native'),retained_native_tokens=256)
                elif count:base=replace_prefix(base,DynamicCache.from_legacy_cache(native_rows),count)
                cache=gpu_cache(cpu_rows(base),'cuda:0');del base
                out=runtime.forward(runtime.target,suffix,cache);logits=out.logits[0,-1].detach().float().cpu().clone();del out,cache
                score=score_option_logits(logits,option_token_ids(runtime.tokenizer,e.choices),e.answer)
                row['variants'][arm]={'native_prefix_tokens':count,'native_prefix_fraction':count/len(prefix),
                    'score':score,'distribution':distribution_metrics(native,logits),'native_agreement':score['decision_answer']==row['native']['decision_answer']}
                if components:
                    row['variants'][arm].update(offline_oracle=arm in('native_all_keys','native_all_values','both_native'),
                        key_projection='target_rms_ellipsoid'if arm=='key_ellipsoid_native256'else'none',
                        native_keys_tokens=len(prefix)if arm in('native_all_keys','both_native')else 256,
                        native_values_tokens=len(prefix)if arm in('native_all_values','both_native')else 256)
                    if arm=='both_native'and not row['variants'][arm]['distribution']['passed']:raise ValueError('Native component oracle guard failed')
                vectors[arm]=logits
            path=OUT/f'logits/{index:02d}.safetensors';path.parent.mkdir(exist_ok=True);save_file(vectors,str(path))
            row['logits_file']=path.relative_to(OUT).as_posix();row['logits_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row);write('rows.json',rows);del mapped,native_rows,projected
        metrics={arm:{'correct':sum(r['variants'][arm]['score']['decision_correct']for r in rows),
            'native_agreement':sum(r['variants'][arm]['native_agreement']for r in rows),
            'mean_native_kl':sum(r['variants'][arm]['distribution']['native_to_candidate_kl_nats']for r in rows)/len(rows),
            'mean_native_prefix_tokens':sum(r['variants'][arm]['native_prefix_tokens']for r in rows)/len(rows)}for arm in rows[0]['variants']}
        competitor=None
        if components:
            competitor=run_prefix_competitor(runtime,selected,OUT/'prefix-only-competitor',max_seconds=90,
                native_answers={r['id']:r['native']['decision_answer']for r in rows})
        summary={'status':'completed','documents':len(rows),'native_correct':sum(r['native']['decision_correct']for r in rows),
            'variants':metrics,'mapper_weights_sha256':cfg['mapper_weights_sha256'],'training_updates':0,'serving_speed_claim':False}
        if components:summary.update(component_oracles=True,prefix_only_competitor=competitor)
        write('summary.json',summary);write('worker-status.json',summary);notice({'phase':'completed','documents':len(rows)})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());write('worker-status.json',{'status':'failed'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
