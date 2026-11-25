"""Layer-ordered offline value correction with a bounded source-feature metric."""
from contextlib import ExitStack
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
    from safetensors.torch import load_file,save_file
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.data import load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.qa_training import group_questions,split_document_prompt,select_mapper_validation
    from kvverify.layer_readout import query_layer,readout_rows,bounded_readout_delta,mapped_layer_pair
    from kvverify.native_prefix import replace_prefix
    from kvverify.state_diagnostic import cpu_rows,distribution_metrics
    from kvverify.state_diagnostic import gpu_cache
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.quality import option_token_ids,score_option_logits
    from kvverify.blocked_attention import blocked_attention
    from cloud.repair_worker import split_train_articles
    from cloud.precision_worker import precision_settings
    cfg=json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if cfg['mode']!='layer_readout'or not torch.cuda.is_available():raise ValueError('Explicit managed CUDA calibration required')
    OUT.mkdir(parents=True,exist_ok=True);write('config.json',cfg);torch.set_num_threads(4);precision_settings(torch)
    for name,sha in cfg['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Input drift')
    train,val=split_train_articles(load_examples(ROOT/'reference/processed/quality/train.jsonl','calibration'),19)
    selected,_=select_mapper_validation(val,19)
    notice({'phase':'loading_models'})
    paths=[snapshot_download(m['repo'],revision=m['revision'],token=False,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=4)for m in cfg['models']]
    runtime=DenseRuntime.from_local(*paths,device='cuda:0',dtype='float32')
    for model in(runtime.source,runtime.target):model.requires_grad_(False)
    mapper=AffineCacheMapper.load(cfg['input_mapper_dir'],runtime.identities['source']['sha256'],runtime.identities['target']['sha256'],runtime.tokenizer_sha256).to('cuda:0').requires_grad_(False)
    if not mapper.manifest['head_mixing']or mapper.manifest['weights_sha256']!=cfg['initial_weights_sha256']:
        raise ValueError('Frozen all-head initialization differs')
    h,dim,layers=mapper.manifest['target_geometry'];width=h*dim
    if runtime.source_geometry!=runtime.target_geometry:raise ValueError('This experiment is shape-preserving')
    rng=random.Random(19);groups=group_questions(train)
    allowed=sorted(set(mapper.manifest['calibration_documents']))
    if not set(allowed)<=set(groups):raise ValueError('Calibration sources outside official optimization pool')
    documents=rng.sample(allowed,24);queries=[];registry=[]
    for doc in documents:
        candidates=[]
        for e in groups[doc]:
            parts=split_document_prompt(runtime,e)
            if len(parts[0])<=8192:candidates.append((e,parts))
        if len(candidates)<4:raise ValueError('Four complete eligible queries per sampled article required')
        chosen=rng.sample(candidates,4);prefix=chosen[0][1][1]
        if len(prefix)<=256 or any(parts[1]!=prefix for _,parts in chosen):raise ValueError('Queries must share exact complete history')
        registry.append({'document_id':doc,'ids':[e.id for e,_ in chosen],'prefix_tokens':len(prefix)})
        queries.append(chosen)
    write('protocol.json',{'version':'layer-readout-all-head-native256-v1','documents':registry,
        'method':'Frozen K maps; same current mapped query state for mapped/native-history readout; layer-ordered V correction',
        'origin':'Inspired by CacheBridge sections4.2/AppendixB; all-head/native256/full-history energy sampling are our variations',
        'reference':'https://arxiv.org/html/2609.00891#S4.SS2','optimizer_updates':0,
        'metric_states':4096,'metric_sampling':'Uniform seeded translated positions across the24complete calibration histories',
        'energy_fraction':cfg['energy_fraction'],'ridge':'max(1e-8,0.01*trace(G)/feature_width)',
        'query_weight':'Each document contributes four equally weighted queries, each averaged over all its tokens/GQA heads',
        'selection':'Minimum native full-vocabulary first-answer KL on original30development articles',
        'native_prefix_policy':'256','acceptance_data_bundled':False,'ephemeral_cache':'Outside model/output/checkpoints; deleted with managed worker'})
    scratch=Path('/opt/ml/readout-cache');scratch.mkdir(exist_ok=False)
    native_logits={};boundaries={e.id:split_document_prompt(runtime,e)for e in selected}
    history=[];best={'kl':float('inf'),'stage':0,'state':None};deadline=STARTED+cfg['worker_timeout_seconds']-240
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Finite readout worker budget reached')
    def validate(stage):
        rows=[];vectors={}
        for e in selected:
            check_time();full,prefix,suffix=boundaries[e.id]
            if e.id not in native_logits:
                out=runtime.forward(runtime.target,full);native_logits[e.id]=out.logits[0,-1].float().cpu().clone();del out
            source=runtime.forward(runtime.source,prefix).past_key_values
            cache=mapper.apply(source,runtime.source,runtime.target);del source
            native=runtime.forward(runtime.target,prefix[:256]).past_key_values
            cache=replace_prefix(cache,native,256);del native
            out=runtime.forward(runtime.target,suffix,cache);logits=out.logits[0,-1].float().cpu().clone();del out,cache
            tokens=option_token_ids(runtime.tokenizer,e.choices);score=score_option_logits(logits,tokens,e.answer)
            ref=score_option_logits(native_logits[e.id],tokens,e.answer)
            rows.append({'id':e.id,'document_id':e.document_id,'score':score,'native_score':ref,
                'distribution':distribution_metrics(native_logits[e.id],logits),'native_agreement':score['decision_answer']==ref['decision_answer']})
            vectors[e.id]=logits
        result={'step':stage,'corrected_layers':stage,'documents':30,'correct':sum(r['score']['decision_correct']for r in rows),
            'native_correct':sum(r['native_score']['decision_correct']for r in rows),'native_agreement':sum(r['native_agreement']for r in rows),
            'mean_native_kl':sum(r['distribution']['native_to_candidate_kl_nats']for r in rows)/30}
        if result['mean_native_kl']<best['kl']:
            best.update(kl=result['mean_native_kl'],stage=stage,state={k:v.cpu().clone()for k,v in mapper.state_dict().items()})
            mapper.manifest.update(selected_at_step=stage,selected_at_layer=stage,selection_axis='corrected_layers',
                new_optimizer_updates=0,initial_weights_sha256=cfg['initial_weights_sha256'],native_prefix_policy='256',
                readout_calibration_documents=documents,training_documents=sorted(set(mapper.manifest.get('training_documents',[]))|set(documents)))
            checkpoint=Path('/opt/ml/checkpoints')/f'step-{stage:06d}';metadata=mapper.save(checkpoint)
            write('best-selection.json',{**result,'native_prefix_policy':'256','weights_sha256':metadata['weights_sha256']})
        history.append(result);write('validation-history.json',history);write(f'validation/step-{stage:06d}.json',rows)
        save_file(vectors,str(OUT/f'validation/step-{stage:06d}.safetensors'))
        notice({'phase':'validation',**result,'best_layer':best['stage']})
    with ExitStack()as stack:
        stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(blocked_attention(256))
        validate(0);hidden=[];sample_indices=[]
        for index,chosen in enumerate(queries):
            check_time();prefix=chosen[0][1][1];notice({'phase':'cache_calibration','document':index+1,'total':24})
            source=runtime.forward(runtime.source,prefix).past_key_values;source_rows=cpu_rows(source);del source
            native=runtime.forward(runtime.target,prefix).past_key_values;native_rows=cpu_rows(native);del native
            if index==0:
                full,_,suffix=chosen[0][1];vectors={}
                out=runtime.forward(runtime.target,full);vectors['native']=out.logits[0,-1].cpu().clone();del out
                source=gpu_cache(source_rows,'cuda:0');mapped=mapper.apply(source,runtime.source,runtime.target);del source
                initial=runtime.forward(runtime.target,prefix[:256]).past_key_values;mapped=replace_prefix(mapped,initial,256);del initial
                out=runtime.forward(runtime.target,suffix,mapped);vectors['mapped']=out.logits[0,-1].cpu().clone();del out,mapped
                gates={}
                for arm in('native','mapped'):
                    state=runtime.target.model.embed_tokens(torch.tensor([suffix],device='cuda:0'))
                    for li,sl in enumerate(mapper.manifest['source_layers']):
                        npair=tuple(t.to('cuda:0')for t in native_rows[li])
                        if arm=='native':pair=npair
                        else:pair=mapped_layer_pair(runtime,mapper,li,tuple(t.to('cuda:0')for t in source_rows[sl]),npair)
                        state,_,_=query_layer(runtime,li,state,pair)
                    logits=runtime.target.lm_head(runtime.target.model.norm(state))[0,-1].cpu().clone()
                    gates[arm]=probability_equivalence(torch,vectors[arm],logits)
                    vectors[arm+'_streamed']=logits
                    del state,pair,npair
                save_file(vectors,str(OUT/'streaming-guard-logits.safetensors'));write('streaming-guards.json',gates)
                if not all(g['passed']for g in gates.values()):raise ValueError('Native/mapped full-layer replay guard failed')
            folder=scratch/str(index);folder.mkdir()
            for layer,sl in enumerate(mapper.manifest['source_layers']):
                save_file({name:t.contiguous()for name,t in zip(('source_k','source_v','native_k','native_v'),(*source_rows[sl],*native_rows[layer]))},str(folder/f'{layer:02d}.safetensors'))
            del source_rows,native_rows
            n=4096//24+(index<4096%24);sample_indices.append(rng.sample(range(256,len(prefix)),n))
            hidden.append([runtime.target.model.embed_tokens(torch.tensor([parts[2]],device='cuda:0')).cpu().clone()for _,parts in chosen])
        key_snapshot=mapper.weight[0].cpu().clone();key_bias=mapper.bias[0].cpu().clone();corrections=[]
        for layer in range(layers):
            check_time();notice({'phase':'readout_layer','layer':layer+1,'total_layers':layers})
            g=torch.zeros((h,width+1,width+1),dtype=torch.float64,device='cuda:0');c=torch.zeros((h,width+1,dim),dtype=torch.float64,device='cuda:0')
            metric=torch.zeros((width+1,width+1),dtype=torch.float64,device='cuda:0');energy=torch.zeros(h,dtype=torch.float64,device='cuda:0')
            target_residual=torch.zeros(h,dtype=torch.float64,device='cuda:0')
            banks=[]
            for index,chosen in enumerate(queries):
                data=load_file(str(scratch/str(index)/f'{layer:02d}.safetensors'));banks.append(data)
                source_pair=tuple(data[n].to('cuda:0')for n in('source_k','source_v'));native_pair=tuple(data[n].to('cuda:0')for n in('native_k','native_v'))
                mapped=mapped_layer_pair(runtime,mapper,layer,source_pair,native_pair,native_tokens=0)
                indices=sample_indices[index];x=source_pair[1][0].transpose(0,1).reshape(-1,width)[indices].double()
                x=torch.cat((x,torch.ones((len(indices),1),device=x.device,dtype=x.dtype)),1);metric+=x.T@x
                residual=(mapped[1]-native_pair[1])[0,:,indices].double();energy+=residual.square().sum((1,2))
                mapped=tuple(torch.cat((n[:,:,:256],m[:,:,256:]),-2)for m,n in zip(mapped,native_pair))
                for qi,(_,parts)in enumerate(chosen):
                    case={'layer':layer,'query_id':chosen[qi][0].id,'document_id':chosen[qi][0].document_id,
                        'prefix_tokens':len(parts[1]),'query_tokens':len(parts[2])}
                    write('current-readout-case.json',case)
                    _,current,record=query_layer(runtime,layer,hidden[index][qi].to('cuda:0'),mapped,capture=True)
                    try:design,residual=readout_rows(record,current,native_pair,source_pair[1],retained_native_tokens=256)
                    except AssertionError:
                        fixture={'current_k':current[0],'current_v':current[1],'native_k':native_pair[0],'native_v':native_pair[1],
                            'source_v':source_pair[1],'query':record['full_q'],'actual_readout':record['actual_readout_full']}
                        if record['full_mask']is not None:fixture['mask']=record['full_mask']
                        save_file({name:value.detach().cpu().contiguous()for name,value in fixture.items()},str(OUT/'failed-readout-fixture.safetensors'))
                        write('failed-readout-case.json',{**case,'scale':record['scale'],'native_prefix_tokens':256,
                            'note':'Internal offline oracle fixture; keep on managedGPU for reproduction; not a serving input'})
                        raise
                    design=design.double();residual=residual.double();weight=1/(4*design.shape[1])
                    g+=weight*design.transpose(-1,-2)@design;c+=weight*design.transpose(-1,-2)@residual
                    target_residual+=weight*residual.square().sum((1,2))
                    del current,record,design,residual
                del source_pair,native_pair,mapped,x
            metric=metric/4096+1e-4*torch.eye(width+1,dtype=torch.float64,device='cuda:0');energy/=4096
            delta,report=bounded_readout_delta(g,c,metric,energy,energy_fraction=cfg['energy_fraction'])
            mapper.weight[1,layer].add_(delta[:,:-1].permute(1,0,2).reshape(width,width).float())
            mapper.bias[1,layer].add_(delta[:,-1].reshape(width).float())
            fitted=OUT/'fitted-values';fitted.mkdir(exist_ok=True)
            save_file({'value_weight':mapper.weight[1,layer].cpu().contiguous(),'value_bias':mapper.bias[1,layer].cpu().contiguous()},str(fitted/f'layer-{layer:02d}.safetensors'))
            for head,row in enumerate(report):
                row.update(surrogate_squared_error_before=float(target_residual[head]),
                    surrogate_squared_error_after=float(target_residual[head])-row['predicted_readout_squared_error_reduction'])
            corrections.append({'layer':layer,'heads':report});write('corrections.json',corrections)
            for index,chosen in enumerate(queries):
                data=banks[index];source_pair=tuple(data[n].to('cuda:0')for n in('source_k','source_v'));native_pair=tuple(data[n].to('cuda:0')for n in('native_k','native_v'))
                mapped=mapped_layer_pair(runtime,mapper,layer,source_pair,native_pair)
                for qi in range(4):
                    result,_,_=query_layer(runtime,layer,hidden[index][qi].to('cuda:0'),mapped)
                    hidden[index][qi]=result.cpu().clone();del result
                del source_pair,native_pair,mapped
            del banks,g,c,metric,energy,delta,target_residual
            if not torch.equal(mapper.weight[0].cpu(),key_snapshot)or not torch.equal(mapper.bias[0].cpu(),key_bias):raise ValueError('Readout fit changed frozen key maps')
            if(layer+1)in(7,14,21,28):validate(layer+1)
        mapper.load_state_dict(best['state']);mapper.manifest.update(selected_at_step=best['stage'],selected_at_layer=best['stage'],
            native_prefix_policy='256',selection_axis='corrected_layers',new_optimizer_updates=0)
        mapper.save(OUT/'best')
        summary={'status':'completed','selected_step':best['stage'],'selected_corrected_layers':best['stage'],
            'native_prefix_policy':'256','validation_history':history,'optimizer_updates':0,'calibration_documents':24,'calibration_queries':96,
            'serving_speed_claim':False,'acceptance_data_used':False}
        write('summary.json',summary);write('worker-status.json',summary);notice({'phase':'completed','selected_layers':best['stage']})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(TimeoutError('Worker limit')))
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());write('worker-status.json',{'status':'failed_or_incomplete'})
        try:upload()
        except Exception:traceback.print_exc()
        raise
