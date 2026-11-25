"""Bound math-attention temporary memory without dropping keys or queries."""
from contextlib import contextmanager


@contextmanager
def blocked_attention(query_block_size=256, *, rematerialize=False):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint
    if type(query_block_size)is not int or query_block_size<1:raise ValueError('Positive integer query block required')
    original=F.scaled_dot_product_attention

    def apply_blocks(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        if dropout_p!=0:raise ValueError('Diagnostic blocking requires dropout=0')
        if is_causal and attn_mask is not None:raise ValueError('Ambiguous explicit and implicit causal mask')
        queries,keys=query.shape[-2],key.shape[-2]
        if queries<=query_block_size:
            return original(query,key,value,attn_mask=attn_mask,dropout_p=0.,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        if attn_mask is not None and(attn_mask.ndim<2 or attn_mask.shape[-2]not in(1,queries)):
            raise ValueError('Attention mask query dimension cannot be aligned')
        outputs=[]
        for start in range(0,queries,query_block_size):
            end=min(queries,start+query_block_size)
            mask=attn_mask
            if mask is not None and mask.shape[-2]==queries:mask=mask[...,start:end,:]
            if is_causal:
                # SDPA's upper-left causal convention, preserving the ORIGINAL
                # query index. Cached suffixes receive their explicit HF mask.
                qpositions=torch.arange(start,end,device=query.device).unsqueeze(-1)
                kpositions=torch.arange(keys,device=query.device).unsqueeze(0)
                mask=kpositions<=qpositions
            outputs.append(original(query[...,start:end,:],key,value,attn_mask=mask,
                dropout_p=0.,is_causal=False,scale=scale,enable_gqa=enable_gqa))
        return torch.cat(outputs,dim=-2)

    def attention(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        if not rematerialize or not torch.is_grad_enabled() or not any(t.requires_grad for t in (query,key,value)):
            return apply_blocks(query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)

        def replay(q,k,v):
            # The backward recomputation happens after the caller's attention context exits.
            with sdpa_kernel(SDPBackend.MATH):
                return apply_blocks(q,k,v,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)

        return checkpoint(replay,query,key,value,use_reentrant=False,preserve_rng_state=False)

    F.scaled_dot_product_attention=attention
    try:yield
    finally:F.scaled_dot_product_attention=original
