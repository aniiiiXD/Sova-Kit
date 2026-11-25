"""Optional Qwen3 key-manifold intervention; not yet quality-qualified.

Native keys follow gain-weighted RMS normalization before RoPE. Their weighted
radius approaches one when pre-normalization variance dominates RMS epsilon.
Projecting an affine estimate to that radius is a hypothesis, not a guarantee
that its direction or receiver behavior improves.
"""


def project_target_key_radius(keys,model,layer,*,position_offset=0):
    import torch
    from .train_mapper import NeuralMapper
    if model.config.model_type!='qwen3'or not 0<=layer<len(model.model.layers):
        raise ValueError('Pinned Qwen3 key normalization required')
    gamma=model.model.layers[layer].self_attn.k_norm.weight.detach().float().to(keys.device)
    if gamma.ndim!=1 or gamma.numel()!=keys.shape[-1]or not torch.isfinite(gamma).all()or bool((gamma.abs()<1e-6).any()):
        raise ValueError('Finite invertible target key gain required')
    content=NeuralMapper._rotate_token_block(keys,model,position_offset,True)
    radius=(content/gamma).square().mean(-1,keepdim=True)
    if not torch.isfinite(radius).all()or bool((radius<=1e-12).any()):raise ValueError('Nonzero finite predicted key direction required')
    projected=content*radius.rsqrt()
    return NeuralMapper._rotate_token_block(projected,model,position_offset).to(keys.dtype)


def project_cache_cpu_rows(rows,model,*,block_size=256):
    """Bounded diagnostic staging; values and input CPU rows stay unchanged."""
    import torch
    device=next(model.parameters()).device;output=[]
    for layer,(keys,values)in enumerate(rows):
        parts=[]
        for start in range(0,keys.shape[-2],block_size):
            part=keys[:,:,start:start+block_size].to(device)
            parts.append(project_target_key_radius(part,model,layer,position_offset=start).cpu())
        output.append((torch.cat(parts,-2),values))
    return tuple(output)


def project_target_cache(cache,model):
    """Same radius policy on a resident inference cache; retain native values."""
    from transformers import DynamicCache
    from .hf_backend import legacy
    return DynamicCache.from_legacy_cache(tuple((project_target_key_radius(k,model,layer),v)
        for layer,(k,v)in enumerate(legacy(cache))))
