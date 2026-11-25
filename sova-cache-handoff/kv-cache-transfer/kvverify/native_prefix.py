"""Explicit partial native recomputation; never call this pure cache transfer."""
from .hf_backend import legacy


def header_tokens(runtime,example):
    tokenizer=runtime.tokenizer
    if not getattr(tokenizer,'is_fast',False):raise ValueError('Fast tokenizer offsets required')
    content=example.prompt()
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    location=text.find(content)
    if location<0 or text.find(content,location+1)>=0:raise ValueError('Ambiguous chat content')
    boundary=location+len('Document:\n')
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if list(encoded['input_ids'])!=list(runtime.tokenize(example)):raise ValueError('Header tokenization changed')
    cut=0
    for i,(start,end)in enumerate(encoded['offset_mapping']):
        if end<=boundary and cut==i:cut+=1
    if not 0<cut<len(encoded['input_ids']):raise ValueError('Invalid header boundary')
    return cut


def replace_prefix(mapped_cache,native_cache,count):
    """Native firstcount positioned KV entries, unchanged translated remainder.

    Both inputs must start at position0. This does not translate or rotate the
    already-positioned native prefix a second time, nor mutate either input.
    """
    import torch
    from transformers import DynamicCache
    mapped=legacy(mapped_cache);native=legacy(native_cache)
    if type(count)is not int or count<0 or len(mapped)!=len(native):raise ValueError('Invalid native prefix count/layers')
    output=[]
    for mp,np in zip(mapped,native):
        pair=[]
        for old,new in zip(mp,np):
            if old.shape[:2]!=new.shape[:2]or old.shape[-1]!=new.shape[-1]or count>old.shape[-2]or count>new.shape[-2]:
                raise ValueError('Native replacement geometry or length mismatch')
            pair.append(torch.cat((new[:,:,:count].to(old.device,dtype=old.dtype),old[:,:,count:]),dim=-2))
        output.append(tuple(pair))
    return DynamicCache.from_legacy_cache(tuple(output))


def oracle_components(mapped_cache,native_cache,*,keys=False,values=False,retained_native_tokens=256):
    """Offline K/V attribution with full native history; not a serving path."""
    import torch
    from transformers import DynamicCache
    mapped=legacy(mapped_cache);native=legacy(native_cache)
    if type(keys)is not bool or type(values)is not bool or type(retained_native_tokens)is not int or retained_native_tokens<0 or len(mapped)!=len(native):
        raise ValueError('Explicit component oracle flags and prefix required')
    output=[]
    for left,right in zip(mapped,native):
        pair=[]
        for old,new,whole_native in zip(left,right,(keys,values)):
            if old.shape!=new.shape or retained_native_tokens>old.shape[-2]:raise ValueError('Full aligned native history required')
            new=new.to(old.device,dtype=old.dtype)
            pair.append(new.clone()if whole_native else torch.cat((new[:,:,:retained_native_tokens],old[:,:,retained_native_tokens:]),-2))
        output.append(tuple(pair))
    return DynamicCache.from_legacy_cache(tuple(output))
