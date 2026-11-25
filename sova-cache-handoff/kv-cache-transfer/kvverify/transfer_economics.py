"""Matched cold/warm reference timings with explicit GPU cache residency.

No worker allocation or model loading. Both frozen models and adapter stay
resident for all arms. This is an in-process batch-one HF measurement; remote
cache transport and an HTTP serving stack require separate measurements.
"""
import json
from pathlib import Path
import random
import statistics
import time

ARMS=('native_cold','transfer_cold','transfer_source_cache_hit','native_target_cache_hit')


def measure_arm(runtime,mapper,full,prefix,suffix,arm):
    from transformers import DynamicCache
    from .hf_backend import legacy
    from .native_prefix import replace_prefix
    from .reference_eval import _prepare,_map_cache
    if arm not in ARMS or full!=prefix+suffix or not prefix or not suffix:
        raise ValueError('Known arm and exact document/question boundary required')
    policy=mapper.manifest.get('native_prefix_policy','none')
    if policy not in('none','256')or(policy=='256'and len(prefix)<=256):
        raise ValueError('Economics requires an actual supported transfer, not native fallback')
    native_count=256 if policy=='256'else 0
    cache=None;setup_ms=0.
    # Cache-hit setup is deliberately outside the request timer, but its
    # duration and GPU-resident payload are reported. No CPU staging/network.
    if arm.endswith('_cache_hit'):
        model=runtime.source if arm=='transfer_source_cache_hit'else runtime.target
        cache,setup_ms=runtime.timed(lambda:runtime.forward(model,prefix).past_key_values)
    cache_bytes=0 if cache is None else sum(t.numel()*t.element_size()for row in legacy(cache)for t in row)
    device=next(runtime.target.parameters()).device
    if device.type=='cuda':runtime.torch.cuda.reset_peak_memory_stats(device)
    runtime.sync();started=time.perf_counter()
    if arm in('native_cold','transfer_cold'):
        result,logits,_,timing=_prepare(runtime,mapper,'native'if arm=='native_cold'else'mapper',full,len(prefix))
        del result
    else:
        mapping_ms=prefix_ms=0.
        if arm=='transfer_source_cache_hit':
            result,mapping_ms=runtime.timed(lambda:_map_cache(runtime,mapper,cache,len(prefix)))
            if native_count:
                def native_initial():
                    native=runtime.forward(runtime.target,prefix[:native_count]).past_key_values
                    return replace_prefix(result,native,native_count)
                result,prefix_ms=runtime.timed(native_initial)
        else:
            # Cache object is mutable; its prefix tensors are read-only. Each
            # native forward receives a fresh wrapper, with no payload clone.
            result=DynamicCache.from_legacy_cache(legacy(cache))
        def readout():
            out=runtime.forward(runtime.target,suffix,result)
            return out.logits[0,-1],int(out.logits[0,-1].argmax().item())
        (logits,_),suffix_ms=runtime.timed(readout)
        timing={'source_prefill_ms':0.,'map_ms':mapping_ms,
                'native_prefix_recompute_ms':prefix_ms,'native_suffix_ms':suffix_ms,
                'target_prefill_tokens':len(suffix)+(native_count if arm=='transfer_source_cache_hit'else 0),
                'source_prefill_tokens':0,'imported_tokens':len(prefix)-native_count if arm=='transfer_source_cache_hit'else 0}
        if cache.get_seq_length()!=len(prefix):raise ValueError('Warm input cache mutated')
    runtime.sync();elapsed_ms=(time.perf_counter()-started)*1000
    memory={}if device.type!='cuda'else{
        'peak_allocated_mib':runtime.torch.cuda.max_memory_allocated(device)/2**20,
        'peak_reserved_mib':runtime.torch.cuda.max_memory_reserved(device)/2**20}
    # CPU materialization and numerical checks occur AFTER the synchronized
    # request timer. No vocabulary formatting is hidden inside the model time.
    vector=logits.detach().float().cpu().clone()
    return {'arm':arm,'request_ms':elapsed_ms,'setup_excluded_from_request_ms':setup_ms,
            'resident_prefix_cache_bytes':cache_bytes,'cache_residency':'same_'+device.type if cache_bytes else'no_prefix_cache',
            'native_prefix_tokens':native_count if arm.startswith('transfer')else 0,
            'transport_measured':False,'fallback':False,**timing,**memory},vector


def run_economics(runtime,mapper,examples,output,*,example_count=12,repeats=5,seed=19,max_seconds=3600,callback=None):
    from .qa_training import split_document_prompt
    from .boundary_probe import probability_equivalence
    if not examples or not 1<=example_count<=30 or repeats<1 or max_seconds<=0:
        raise ValueError('Bounded nonempty economics protocol required')
    unique={}
    for e in sorted(examples,key=lambda e:e.id):unique.setdefault(e.document_id,e)
    ordered=sorted(unique.values(),key=lambda e:(len(runtime.tokenize(e)),e.id))
    count=min(example_count,len(ordered))
    indices=[round(i*(len(ordered)-1)/max(1,count-1))for i in range(count)]
    selected=[ordered[i]for i in indices]
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    config={'protocol':'matched-cold-warm-gpu-cache-v1','example_ids':[e.id for e in selected],
            'repeats':repeats,'warmups_per_arm':1,'arms':ARMS,'mapper':mapper.manifest,
            'models':runtime.identities,'cache_transport':'No remote transport measured; warm caches already on sameGPU',
            'residency':'Both models and mapper resident in every arm; target-only deployment may have a different footprint',
            'backend':'Caller-selected validated backend; do not extrapolate to an optimized serving engine',
            'measurement':'Synchronized in-process first-token selection; no HTTP/tokenization/network time'}
    (root/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    rng=random.Random(seed);rows=[];deadline=time.monotonic()+max_seconds
    with runtime.torch.inference_mode():
        first=split_document_prompt(runtime,selected[0])
        with(root/'warmup.jsonl').open('w')as stream:
            for arm in ARMS:
                row,_=measure_arm(runtime,mapper,*first,arm);stream.write(json.dumps(row)+'\n')
        with(root/'rows.jsonl').open('w')as stream:
            for e in selected:
                full,prefix,suffix=split_document_prompt(runtime,e)
                for repeat in range(repeats):
                    vectors={};order=list(ARMS);rng.shuffle(order)
                    for arm in order:
                        if time.monotonic()>deadline:raise TimeoutError('Incomplete economics, no headline permitted')
                        row,logits=measure_arm(runtime,mapper,full,prefix,suffix,arm)
                        row.update(id=e.id,document_id=e.document_id,repeat=repeat,prefix_tokens=len(prefix),suffix_tokens=len(suffix))
                        vectors[arm]=logits;rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    gates={'native_cache_hit':probability_equivalence(runtime.torch,vectors['native_cold'],vectors['native_target_cache_hit']),
                           'translated_cache_hit':probability_equivalence(runtime.torch,vectors['transfer_cold'],vectors['transfer_source_cache_hit'])}
                    if not all(g['passed']for g in gates.values()):
                        (root/'failed-parity.json').write_text(json.dumps({'id':e.id,'repeat':repeat,**gates},indent=2)+'\n')
                        raise ValueError('Cold/warm readout parity failed; no speed headline')
                if callback:callback({'phase':'economics','completed_documents':len({r['document_id']for r in rows}),'total_documents':count})
    byarm={arm:[r['request_ms']for r in rows if r['arm']==arm]for arm in ARMS}
    def quantile(xs,q):
        values=sorted(xs);position=(len(values)-1)*q;lower=int(position);upper=min(lower+1,len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(position-lower)
    pairs={(r['id'],r['repeat'],r['arm']):r['request_ms']for r in rows}
    ratios={arm:[pairs[(e.id,n,'native_cold')]/pairs[(e.id,n,arm)]for e in selected for n in range(repeats)]for arm in ARMS if arm!='native_cold'}
    summary={'status':'completed','documents':count,'repeats':repeats,'timed_requests':len(rows),
        'request_ms':{arm:{'median':statistics.median(xs),'p95':quantile(xs,.95),'count':len(xs)}for arm,xs in byarm.items()},
        'median_paired_speedup_vs_native_cold':{arm:statistics.median(xs)for arm,xs in ratios.items()},
        'quality_note':'Timing inputs are development documents; independent quality qualification is separate',
        'limitations':config['measurement']+'; '+config['cache_transport']+'; '+config['backend']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
