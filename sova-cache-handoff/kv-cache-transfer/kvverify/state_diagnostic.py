"""Explicit, inference-only state interventions; no fitting or cloud submission.

Native receiver queries are oracle diagnostics, never a deployable substitute
for target prefill. CPU cache staging bounds GPU residency. Runtime callers
must explicitly enable inference mode and a validated attention backend.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import random
from pathlib import Path

from .hf_backend import legacy


def select_panel(candidates, lengths, *, seed=19):
    """24/30 frozen validation articles, selected without outcome access."""
    if len(candidates)!=30 or len({e.id for e in candidates})!=30 or len({e.document_id for e in candidates})!=30:
        raise ValueError('Exactly 30 distinct validation articles required')
    if set(lengths)!={e.id for e in candidates} or any(type(n)is not int or n<2 for n in lengths.values()):
        raise ValueError('Complete positive token lengths required')
    ordered=sorted(candidates,key=lambda e:(lengths[e.id],e.id))
    rng=random.Random(seed)
    rows=[]
    for band in range(3):
        chosen=rng.sample(ordered[band*10:(band+1)*10],8)
        rows.extend({'id':e.id,'document_id':e.document_id,'context_hash':e.context_hash,
                     'prompt_tokens':lengths[e.id],'length_band':band}for e in sorted(chosen,key=lambda e:e.id))
    return rows


def cpu_rows(cache):
    return tuple(tuple(x.detach().to('cpu',copy=True)for x in pair)for pair in legacy(cache))


def gpu_cache(rows,device):
    from transformers import DynamicCache
    return DynamicCache.from_legacy_cache(tuple(tuple(x.to(device)for x in pair)for pair in rows))


def map_cpu_rows(mapper,source_rows,source_model,target_model,*,normalize_keys,block_size=256):
    """One target layer and one input block at a time; no whole-source clone."""
    import torch
    if type(normalize_keys)is not bool or type(block_size)is not int or block_size<1:
        raise ValueError('Explicit RMS setting and positive block size required')
    device=next(target_model.parameters()).device
    count=source_rows[0][0].shape[-2]
    heads,dim,layers=mapper.manifest['source_geometry']
    if len(source_rows)!=layers or any(k.shape!=(1,heads,count,dim)or v.shape!=k.shape for k,v in source_rows):
        raise ValueError('Source cache geometry differs from checkpoint')
    result=[]
    for layer,group in enumerate(mapper.manifest['source_layer_groups']):
        needed=tuple(sorted(set([mapper.manifest['source_layers'][layer],*group])))
        buffers=[[],[]]
        for start in range(0,count,block_size):
            tensors=tuple(x[:,:,start:start+block_size].to(device)for i in needed for x in source_rows[i])
            pair=mapper._map_layer_token_block(layer,start,needed,tensors,source_model,target_model,
                                               normalize_keys=normalize_keys)
            for side,value in enumerate(pair):
                if not torch.isfinite(value).all():raise ValueError('Nonfinite diagnostic mapped state')
                buffers[side].append(value.detach().cpu())
            del tensors,pair,value
        result.append(tuple(torch.cat(parts,dim=-2)for parts in buffers))
    return tuple(result)


@contextmanager
def capture_native_queries(model):
    """Capture the actual SDPA inputs/outputs, retaining complete suffix shape.

    Capture runs outside training in one worker thread. Native SDPA is called
    unchanged, including its active query-blocking wrapper. No Q recomputation
    or shortened matrix multiplication substitutes for the real forward pass.
    """
    import torch
    from torch.nn import functional as F
    records={};handles=[];active={'layer':None};original=F.scaled_dot_product_attention
    def before(layer):
        def hook(module,args,kwargs):
            active['layer']=layer
        return hook
    def after(layer):
        def hook(module,args,kwargs,output):
            if layer not in records:raise ValueError('Expected pinned Qwen SDPA call was not captured')
            records[layer]['projected_output_full']=output[0].detach().cpu()
            active['layer']=None
        return hook
    def capture(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        layer=active['layer']
        if layer is None:raise ValueError('SDPA invoked outside the selected target layer')
        if layer in records or q.shape[0]!=1 or dropout_p!=0 or enable_gqa or q.shape[1]!=k.shape[1]:
            raise ValueError('Expected one batch-one repeated-KV Qwen SDPA call per layer')
        output=original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        count=q.shape[-2];indices=sorted({0,count//2,count-1})
        mask=attn_mask.detach().cpu()if attn_mask is not None else None
        if mask is not None and mask.ndim!=4:raise ValueError('Unsupported diagnostic attention mask')
        selected_mask=mask[:,:,indices,:]if mask is not None and mask.shape[-2]!=1 else mask
        full_q=q.detach().cpu()
        records[layer]={'q':full_q[:,:,indices,:],'full_q':full_q,'indices':indices,
            'suffix_tokens':count,'scale':float(scale if scale is not None else q.shape[-1]**-.5),
            'sdpa_scale':scale,'is_causal':is_causal,'mask':selected_mask,'full_mask':mask,
            'actual_readout_full':output.detach().cpu(),'sdpa_shapes':[list(x.shape)for x in(q,k,v)]}
        return output
    try:
        for layer,block in enumerate(model.model.layers):
            handles.append(block.self_attn.register_forward_pre_hook(before(layer),with_kwargs=True))
            handles.append(block.self_attn.register_forward_hook(after(layer),with_kwargs=True))
        F.scaled_dot_product_attention=capture
        yield records
    finally:
        F.scaled_dot_product_attention=original
        for handle in handles:handle.remove()


def attention_readout(q,k,v,*,prefix_tokens,query_indices,scale,mask=None,query_block_size=None):
    """Explicit GQA, causal suffix positions and stable FP32 softmax."""
    import torch
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=1 or k.shape[0]!=1:
        raise ValueError('Unsupported attention diagnostic shapes')
    if q.shape[-1]!=k.shape[-1]or q.shape[1]%k.shape[1]or len(query_indices)!=q.shape[-2]:
        raise ValueError('Head/query geometry mismatch')
    count=q.shape[-2]
    if count<1 or(query_block_size is not None and(type(query_block_size)is not int or query_block_size<1)):
        raise ValueError('Nonempty queries and explicit positive block size required')
    if mask is not None and(mask.ndim<2 or mask.shape[-2]not in(1,count)):
        raise ValueError('Unaligned readout query mask')
    groups=q.shape[1]//k.shape[1]
    k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
    # Match the pinned Torch2.5.1 math-SDPA evaluation order, not merely the
    # algebraically equivalent post-matmul scale. See aten/native/transformers/
    # attention.cpp: both operands receive sqrt(abs(scale)) before matmul.
    factor=math.sqrt(abs(float(scale)))
    positions=prefix_tokens+torch.tensor(query_indices,device=q.device)
    outputs=[];probabilities=[];logs=[];raw=[];block=query_block_size or count
    for start in range(0,count,block):
        end=min(count,start+block)
        scores=(q[...,start:end,:].float()*(-factor if scale<0 else factor))@(k.float().transpose(-1,-2)*factor)
        allowed=torch.arange(k.shape[-2],device=q.device)[None,:]<=positions[start:end,None]
        scores=scores.masked_fill(~allowed[None,None],float('-inf'))
        if mask is not None:
            part=mask.to(q.device)
            if part.shape[-2]!=1:part=part[...,start:end,:]
            if part.dtype==torch.bool:scores=scores.masked_fill(~part,float('-inf'))
            else:scores=scores+part[...,:k.shape[-2]]
        logp=scores.log_softmax(-1);p=scores.softmax(-1)
        # Torch's _safe_softmax returns zeros for fully -inf-masked rows.
        empty=torch.isneginf(scores).all(-1,keepdim=True)
        p=torch.where(empty,torch.zeros_like(p),p)
        logp=torch.where(empty,torch.full_like(logp,float('-inf')),logp)
        if not torch.isfinite(p).all():raise ValueError('Nonfinite attention probabilities')
        outputs.append(p@v.float());probabilities.append(p);logs.append(logp);raw.append(scores)
    return tuple(torch.cat(parts,dim=-2)for parts in(outputs,probabilities,logs,raw))


def compare_readouts(native,candidate):
    import torch
    from torch.nn import functional as F
    ref,p,logp,scores=native
    value,candp,candlogp,candscores=candidate
    allowed=p>0
    kl=torch.where(allowed,p*(logp-candlogp),0.).sum(-1)
    # -inf causal entries are masked from moments rather than clamped to zero.
    def moments(x):
        finite=torch.isfinite(x);n=finite.sum(-1).clamp_min(1)
        finite_x=torch.where(finite,x,0.)
        mean=finite_x.sum(-1)/n
        var=torch.where(finite,(x-mean[...,None]).square(),0.).sum(-1)/n
        return {'mean':mean.cpu().tolist(),'std':var.sqrt().cpu().tolist()}
    return {'relative_output_error':((value-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)).cpu().tolist(),
            'output_cosine':F.cosine_similarity(value,ref,dim=-1,eps=1e-8).cpu().tolist(),
            'native_to_candidate_attention_kl':kl.cpu().tolist(),
            'native_entropy':(-torch.where(allowed,p*logp,0.)).sum(-1).cpu().tolist(),
            'candidate_entropy':(-torch.where(candp>0,candp*candlogp,0.)).sum(-1).cpu().tolist(),
            'native_logit_moments':moments(scores),'candidate_logit_moments':moments(candscores)}


def norm_summary(x):
    z=x.float();rms=z.square().mean(-1).sqrt()
    return {'per_head_rms_mean':rms.mean(dim=(0,2)).cpu().tolist(),
            'per_head_rms_std':rms.std(dim=(0,2),unbiased=False).cpu().tolist()}


class ReadoutMismatch(ValueError):
    def __init__(self,details):
        self.diagnostics=details
        super().__init__('Exact-shape native SDPA replay failed: '+json.dumps(details))


def replay_readout(record,k,v,prefix_tokens):
    """Actual SDPA over the full suffix; sample positions only after execution.

    Probability/entropy diagnostics use explicit selected-query FP32 softmax.
    Output errors use the native math-SDPA kernel at its original query shape.
    """
    from torch.nn import functional as F
    q=record['full_q'].to(k.device).contiguous()
    groups=q.shape[1]//k.shape[1]
    expanded_k=k.repeat_interleave(groups,dim=1).contiguous()
    expanded_v=v.repeat_interleave(groups,dim=1).contiguous()
    if [list(x.shape)for x in(q,expanded_k,expanded_v)]!=record['sdpa_shapes']:
        raise ValueError('Diagnostic replay changed the native SDPA shapes')
    mask=record['full_mask']
    full=F.scaled_dot_product_attention(q,expanded_k,expanded_v,
        attn_mask=mask.to(k.device)if mask is not None else None,dropout_p=0.,
        is_causal=record['is_causal'],scale=record['sdpa_scale'])
    _,p,logp,scores=attention_readout(record['q'].to(k.device),k,v,prefix_tokens=prefix_tokens,
        query_indices=record['indices'],scale=record['scale'],mask=record['mask'])
    return (full[:,:,record['indices'],:],p,logp,scores),full


def layer_interventions(target,native_rows,mapped_rows,queries,prefix_tokens,*,deadline=None):
    """Native-Q single-layer counterfactuals; native suffix KV always retained."""
    import time
    import torch
    device=next(target.parameters()).device;result=[]
    if len(native_rows)!=len(mapped_rows)or len(queries)!=len(native_rows):
        raise ValueError('Native, mapped and query layer counts differ')
    for layer,((nk,nv),(mk,mv))in enumerate(zip(native_rows,mapped_rows)):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Readout diagnostic deadline')
        rec=queries[layer]
        nk=nk.to(device);nv=nv.to(device);mk=mk.to(device);mv=mv.to(device)
        if mk.shape[-2]!=prefix_tokens or mv.shape!=mk.shape:
            raise ValueError('Only the complete document prefix may be substituted')
        ref,full=replay_readout(rec,nk,nv,prefix_tokens)
        projected=target.model.layers[layer].self_attn.o_proj(full.transpose(1,2).contiguous().reshape(1,rec['suffix_tokens'],-1).contiguous())
        actual=rec['projected_output_full'].to(device)
        if not torch.allclose(projected,actual,atol=2e-5,rtol=2e-4):
            raise ReadoutMismatch({'layer':layer,'max_abs_error':float((projected-actual).abs().max()),
                'relative_l2_error':float((projected-actual).norm()/actual.norm().clamp_min(1e-8)),
                'head_readout_max_abs_error':float((full-rec['actual_readout_full'].to(device)).abs().max()),
                'actual_output_norm':float(actual.norm()),'sdpa_shapes':rec['sdpa_shapes'],
                'atol':2e-5,'rtol':2e-4})
        k=torch.cat((mk,nk[:,:,prefix_tokens:]),dim=-2)
        v=torch.cat((mv,nv[:,:,prefix_tokens:]),dim=-2)
        row={'layer':layer,'query_indices':rec['indices'],'diagnostic':'native_query_oracle_not_deployable',
             'native_k':norm_summary(nk[:,:,:prefix_tokens]),'mapped_k':norm_summary(mk),
             'native_v':norm_summary(nv[:,:,:prefix_tokens]),'mapped_v':norm_summary(mv),'arms':{}}
        for name,keys,values in(('mapped_k_native_v',k,nv),('native_k_mapped_v',nk,v),('mapped_k_mapped_v',k,v)):
            candidate,candidate_full=replay_readout(rec,keys,values,prefix_tokens)
            row['arms'][name]=compare_readouts(ref,candidate)
            del candidate,candidate_full
        result.append(row)
        del nk,nv,mk,mv,k,v,ref,full,projected,actual
    return result


def distribution_metrics(native,candidate):
    from .boundary_probe import probability_equivalence
    import torch
    oracle=probability_equivalence(torch,native,candidate)
    # Numerical oracle thresholds are descriptive here, not a learned-state gate.
    return {**oracle,'used_as_acceptance_gate':False}


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
