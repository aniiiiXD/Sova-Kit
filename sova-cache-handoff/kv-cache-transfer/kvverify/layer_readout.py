"""Exact frozen Qwen3 query-layer replay and bounded value-readout fitting.

This is an offline calibration primitive. Native history is an oracle target
for fitting, never a serving input. Queries come from the current mapped path.
"""


def mapped_layer_pair(runtime,mapper,layer,source,native,*,native_tokens=256):
    """Same256-token mapping blocks as the deployed affine mapper."""
    import torch
    sl=mapper.manifest['source_layers'][layer];parts=[];length=source[0].shape[-2]
    if not 0<=native_tokens<length:raise ValueError('Translated remainder required')
    for start in range(0,length,256):
        parts.append(mapper._map_layer_token_block(layer,start,(sl,),tuple(t[:,:,start:start+256]for t in source),runtime.source,runtime.target))
    mapped=tuple(torch.cat([p[side]for p in parts],-2)for side in range(2))
    return tuple(torch.cat((n[:,:,:native_tokens],m[:,:,native_tokens:]),-2)for m,n in zip(mapped,native))


def query_layer(runtime,layer,hidden,prefix,*,capture=False):
    import torch
    from transformers import DynamicCache
    from contextlib import nullcontext
    from .state_diagnostic import capture_native_queries
    model=runtime.target
    if model.config.model_type!='qwen3' or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen pinned Qwen3 layer path required')
    if not 0<=layer<len(model.model.layers)or hidden.ndim!=3 or hidden.shape[0]!=1:
        raise ValueError('One complete query and valid layer required')
    k,v=prefix;length=k.shape[-2];count=hidden.shape[1]
    if k.shape!=v.shape or length<1 or count<1:raise ValueError('Nonempty exact history KV required')
    empty=k[:,:,:0];cache=DynamicCache.from_legacy_cache(tuple((empty,empty)for _ in range(layer))+(prefix,))
    positions=torch.arange(length,length+count,device=hidden.device)
    mask=torch.zeros((1,1,count,length+count),dtype=hidden.dtype,device=hidden.device)
    future=torch.arange(length+count,device=hidden.device)[None,:]>positions[:,None]
    mask.masked_fill_(future[None,None],torch.finfo(hidden.dtype).min)
    rotary=model.model.rotary_emb(hidden,positions[None,:])
    with capture_native_queries(model)if capture else nullcontext({})as records:
        result=model.model.layers[layer](hidden_states=hidden,attention_mask=mask,position_ids=positions[None,:],
            past_key_value=cache,use_cache=True,output_attentions=False,cache_position=positions,position_embeddings=rotary)[0]
    current=(cache.key_cache[layer],cache.value_cache[layer])
    if current[0].shape[-2]!=length+count:raise ValueError('Layer cache position drift')
    return result,current,records.get(layer)


def readout_rows(record,current,native_prefix,source_values,*,retained_native_tokens=0):
    """Return per-KV-head design/residual rows at the SAME mapped query state.

    The bias feature is attention mass on translated positions. Prefix and
    fresh-query contributions are identical between the compared branches.
    """
    import torch
    from .state_diagnostic import attention_readout
    k,v=current;nk,nv=native_prefix;length=nk.shape[-2];count=k.shape[-2]-length
    if nk.shape!=nv.shape or source_values.shape!=nk.shape or count<1 or not 0<=retained_native_tokens<length:
        raise ValueError('Aligned source/native history and translated remainder required')
    q=record['full_q'].to(k.device);mask=record['full_mask']
    kwargs={'prefix_tokens':length,'query_indices':list(range(count)),'scale':record['scale'],'mask':mask,'query_block_size':256}
    mapped,prob,*_=attention_readout(q,k,v,**kwargs)
    torch.testing.assert_close(mapped,record['actual_readout_full'].to(k.device),atol=2e-5,rtol=2e-4)
    reference,*_=attention_readout(q,torch.cat((nk,k[:,:,length:]),-2),torch.cat((nv,v[:,:,length:]),-2),**kwargs)
    h,dim=nk.shape[1],nk.shape[-1];groups=q.shape[1]//h
    x=source_values[0].transpose(0,1).reshape(length,h*dim)[retained_native_tokens:].float()
    weights=prob[0,:,:,retained_native_tokens:length]
    design=torch.cat((weights@x,weights.sum(-1,keepdim=True)),-1)
    residual=(reference-mapped)[0]
    return design.reshape(h,groups*count,h*dim+1),residual.reshape(h,groups*count,dim)


def bounded_readout_delta(gram,cross,metric,reference_energy,*,energy_fraction=.01):
    """Solve (G+ridge*I+mu*M)delta=C with a feasible energy multiplier."""
    import math
    import torch
    if gram.ndim!=3 or cross.ndim!=3 or gram.shape[:2]!=cross.shape[:2]or gram.shape[-1]!=gram.shape[-2]:
        raise ValueError('Head-batched sufficient statistics required')
    h,p,_=gram.shape
    if metric.shape==(p,p):metric=metric.expand(h,p,p)
    if metric.shape!=gram.shape or reference_energy.shape!=(h,)or not 0<energy_fraction<=1:
        raise ValueError('Aligned positive feature-energy metric required')
    if any(not torch.isfinite(t).all()for t in(gram,cross,metric,reference_energy))or bool((reference_energy<0).any()):
        raise ValueError('Finite nonnegative calibration statistics required')
    results=[];reports=[]
    for index in range(h):
        g=gram[index].double();c=cross[index].double();m=metric[index].double()
        if not torch.allclose(m,m.T,atol=1e-10,rtol=1e-8)or int(torch.linalg.cholesky_ex(m).info):
            raise ValueError('Feature metric must be symmetric positive definite')
        ridge=max(1e-8,.01*float(g.trace())/p)
        base=(g+g.T)/2+ridge*torch.eye(p,dtype=g.dtype,device=g.device)
        bound=energy_fraction*float(reference_energy[index])
        def solve(mu):
            value=torch.linalg.solve(base+mu*m,c)
            return value,float((value*(m@value)).sum())
        initial,initial_energy=solve(0.)
        if bound==0:delta=torch.zeros_like(initial);mu=None;energy=0.
        elif initial_energy<=bound:delta=initial;mu=0.;energy=initial_energy
        else:
            low,high=0.,1e-12
            for _ in range(100):
                delta,energy=solve(high)
                if energy<=bound:break
                low=high;high*=2
            else:raise ValueError('Could not bracket a feasible readout correction')
            for _ in range(45):
                mid=(low+high)/2;candidate,e=solve(mid)
                if e<=bound:high=mid;delta=candidate;energy=e
                else:low=mid
            mu=high
        if not torch.isfinite(delta).all()or not math.isfinite(energy)or energy>bound*(1+1e-8)+1e-12:
            raise ValueError('Correction violates its declared energy budget')
        improvement=float(2*(delta*c).sum()-(delta*(g@delta)).sum())
        results.append(delta);reports.append({'ridge':ridge,'mu':mu,'energy_before':initial_energy,'energy_after':energy,
            'predicted_readout_squared_error_reduction':improvement,
            'predicted_regularized_objective_reduction':improvement-ridge*float(delta.square().sum()),
            'energy_budget':bound,'energy_fraction':energy_fraction,'feature_width':p})
    return torch.stack(results),reports
