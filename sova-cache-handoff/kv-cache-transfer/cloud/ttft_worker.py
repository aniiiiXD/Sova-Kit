"""Paired SageMaker timing of the reference and optimized affine mapper."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
from statistics import median
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.probe_worker import OUT, notice, upload, write

ARMS = ("native_cold", "native_cache_hit", "reference_cold", "reference_source_hit",
        "fast_cold", "fast_source_hit")


def timing_arms(native_prefix_tokens):
    if native_prefix_tokens not in (0, 256):
        raise ValueError("Unsupported native prefix timing policy")
    return (*ARMS, "native_256_hit") if native_prefix_tokens else ARMS


def bench(runtime, mapper, example, repeats, rng, *, fast_block_tokens, diagnose_cache_tensors,
          native_prefix_tokens=0):
    from kvverify.boundary_probe import probability_equivalence
    from kvverify.fast_affine import map_affine_fast
    from kvverify.hf_backend import legacy
    from kvverify.native_prefix import replace_prefix
    from kvverify.qa_training import split_document_prompt
    from kvverify.quality import option_token_ids

    full, prefix, suffix = split_document_prompt(runtime, example)
    arms = timing_arms(native_prefix_tokens)
    if native_prefix_tokens >= len(prefix):
        raise ValueError("Timing requires an actual translated prefix")
    source_cache = runtime.forward(runtime.source, prefix).past_key_values
    target_cache = runtime.forward(runtime.target, prefix).past_key_values
    tokens = option_token_ids(runtime.tokenizer, example.choices)

    def compute_native_head():
        return runtime.forward(runtime.target, prefix[:native_prefix_tokens]).past_key_values

    native_head_hit = compute_native_head() if native_prefix_tokens else None

    def clone_view(cache):
        return runtime.DynamicCache.from_legacy_cache(legacy(cache))

    def convert(cache, fast):
        if fast:
            return map_affine_fast(mapper, cache, runtime.source, runtime.target, block_tokens=fast_block_tokens)
        return mapper.apply(cache, runtime.source, runtime.target)

    def timed_request(arm):
        if arm == "native_cold":
            out = runtime.forward(runtime.target, full)
        elif arm == "native_cache_hit":
            out = runtime.forward(runtime.target, suffix, clone_view(target_cache))
        elif arm == "native_256_hit":
            out = runtime.forward(runtime.target, prefix[native_prefix_tokens:] + suffix,
                                  clone_view(native_head_hit))
        else:
            source = (source_cache if arm.endswith("_source_hit") else
                      runtime.forward(runtime.source, prefix).past_key_values)
            cache = convert(source, arm.startswith("fast_"))
            if native_prefix_tokens:
                # The hybrid reconstructs a native head per transfer request.
                # This prefill and the cache splice are inside the TTFT clock.
                cache = replace_prefix(cache, compute_native_head(), native_prefix_tokens)
            out = runtime.forward(runtime.target, suffix, cache)
        logits = out.logits[0, -1]
        return logits, int(logits.argmax().item())

    torch = runtime.torch
    for arm in arms:
        runtime.timed(lambda: timed_request(arm))
    records, checks = [], []
    map_only = {"reference": [], "fast": []}
    for repeat in range(repeats):
        order = list(arms)
        rng.shuffle(order)
        vectors = {}
        for arm in order:
            torch.cuda.reset_peak_memory_stats()
            (logits, first), elapsed_ms = runtime.timed(lambda: timed_request(arm))
            transfer = arm.startswith(("reference_", "fast_"))
            target_tokens = {
                "native_cold": len(full),
                "native_cache_hit": len(suffix),
                "native_256_hit": len(full) - native_prefix_tokens,
            }
            records.append({"id": example.id, "document_id": example.document_id, "repeat": repeat,
                            "source_prefill_tokens": len(prefix) if transfer and arm.endswith("_cold") else 0,
                            "target_prefill_tokens": (native_prefix_tokens + len(suffix) if transfer
                                                      else target_tokens[arm]),
                            "imported_translated_tokens": len(prefix) - native_prefix_tokens if transfer else 0,
                            "arm": arm, "ttft_ms": elapsed_ms, "prefix_tokens": len(prefix),
                            "suffix_tokens": len(suffix), "first_token_id": first,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20})
            vectors[arm] = logits.detach().float().cpu()
        for suffix_name in ("cold", "source_hit"):
            reference, fast = vectors["reference_"+suffix_name], vectors["fast_"+suffix_name]
            parity = probability_equivalence(torch, reference, fast)
            pick = lambda scores: max(tokens, key=lambda label: float(scores[tokens[label]]))
            checks.append({"id": example.id, "repeat": repeat, "path": suffix_name,
                           "option_agreement": pick(reference) == pick(fast), **parity})
        alternatives = ["reference", "fast"]
        rng.shuffle(alternatives)
        for label in alternatives:
            cache, milliseconds = runtime.timed(lambda: convert(source_cache, label == "fast"))
            del cache
            map_only[label].append(milliseconds)
        if (source_cache.get_seq_length() != len(prefix) or target_cache.get_seq_length() != len(prefix)
                or (native_head_hit is not None and native_head_hit.get_seq_length() != native_prefix_tokens)):
            raise RuntimeError("A timed request mutated its warm input cache")
    cache_diagnostics = []
    if diagnose_cache_tensors:
        baseline = convert(source_cache, False)
        for block_tokens in sorted({fast_block_tokens, 8192}):
            candidate = map_affine_fast(mapper, source_cache, runtime.source, runtime.target,
                                        block_tokens=block_tokens)
            for side, name in enumerate(("key", "value")):
                corresponding = [(a[side], b[side]) for a, b in zip(legacy(baseline), legacy(candidate))]
                changed = torch.stack([(a != b).sum() for a, b in corresponding]).sum().item()
                delta = torch.stack([(a.float()-b.float()).abs().max()
                                     for a, b in corresponding]).max().item()
                cache_diagnostics.append({"id": example.id, "side": name, "block_tokens": block_tokens,
                                          "changed_elements": changed, "max_absolute_difference": delta,
                                          "total_elements": sum(a.numel() for a, _ in corresponding)})
            del candidate
    return records, checks, map_only, cache_diagnostics


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.data import load_examples
    from kvverify.evidence import percentile
    from kvverify.hf_backend import DenseRuntime

    cfg = json.loads(Path(os.environ["KV_CONFIG"]).read_text())
    if cfg["mode"] != "ttft" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("TTFT benchmark requires BF16-capable CUDA")
    for name, expected in cfg["input_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input hash mismatch: " + name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    write("config.json", cfg)
    paths = []
    for model in cfg["models"]:
        notice({"phase": "downloading_model", "candidate": model["repo"]})
        paths.append(snapshot_download(model["repo"], revision=model["revision"], token=False,
                     allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], max_workers=4))
    notice({"phase": "loading_models"})
    runtime = DenseRuntime.from_local(*paths, dtype="bfloat16")
    mapper = AffineCacheMapper.load(cfg["input_mapper_dir"], runtime.identities["source"]["sha256"],
                                   runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
    if mapper.manifest["weights_sha256"] != cfg["mapper_weights_sha256"]:
        raise ValueError("Historical adapter bytes differ")
    native_prefix_tokens = cfg.get("native_prefix_tokens", 0)
    arms = timing_arms(native_prefix_tokens)
    expected_policy = str(native_prefix_tokens) if native_prefix_tokens else "none"
    if mapper.manifest.get("native_prefix_policy", "none") != expected_policy:
        raise ValueError("Timing policy differs from the verified adapter manifest")
    mapper = mapper.to("cuda:0").requires_grad_(False)
    examples = load_examples(ROOT/"inputs/quality.jsonl", "eval")
    distinct = {}
    for example in sorted(examples, key=lambda item: item.id):
        distinct.setdefault(example.document_id, example)
    sorted_examples = sorted(distinct.values(), key=lambda example: (len(runtime.tokenize(example)), example.id))
    count = cfg["documents"]
    if not 2 <= count <= len(sorted_examples):
        raise ValueError("Too few distinct development articles")
    selected = [sorted_examples[round(i*(len(sorted_examples)-1)/(count-1))] for i in range(count)]
    write("protocol.json", {"documents": [e.document_id for e in selected], "repeats": cfg["repeats"],
                            "device": torch.cuda.get_device_name(0), "model_identities": runtime.identities,
                            "first_token": "synchronized model-only logits and top-token selection",
                            "warm_cache_setup_in_ttft": False, "transport_or_http_measured": False,
                            "native_prefix_recomputed_per_transfer": native_prefix_tokens,
                            "quality_status": ("development-qualified candidate" if cfg.get("verified_training_job")
                                               else "unchanged archived weights; no quality promotion")})
    rng = random.Random(cfg["seed"])
    rows, parity, cache_diagnostics = [], [], []
    map_timings = defaultdict(list)
    with torch.inference_mode():
        for index, example in enumerate(selected, start=1):
            notice({"phase": "timing", "document": index, "total": len(selected), "id": example.id})
            records, checks, mapping, diagnostics = bench(
                runtime, mapper, example, cfg["repeats"], rng, fast_block_tokens=cfg["fast_block_tokens"],
                diagnose_cache_tensors=cfg["diagnose_cache_tensors"],
                native_prefix_tokens=native_prefix_tokens)
            rows.extend(records)
            parity.extend(checks)
            cache_diagnostics.extend(diagnostics)
            for name, samples in mapping.items():
                map_timings[name].extend(samples)
            write("rows.json", rows)
            write("parity.json", parity)
            if cache_diagnostics:
                write("cache-diagnostics.json", cache_diagnostics)

    def stats(samples):
        return {"p50": median(samples), "p95": percentile(samples, .95), "n": len(samples)}

    summary = {"status": "completed", "documents": len(selected), "timed_requests": len(rows),
               "ttft_ms": {arm: stats([r["ttft_ms"] for r in rows if r["arm"] == arm]) for arm in arms},
               "conversion_ms": {name: stats(samples) for name, samples in map_timings.items()},
               "same_adapter_parity_passed": sum(r["passed"] and r["option_agreement"] for r in parity),
               "same_adapter_parity_total": len(parity),
               "max_forward_kl_nats": max(r["native_to_candidate_kl_nats"] for r in parity),
               "peak_allocated_mib_by_arm": {arm: max(r["peak_allocated_mib"]
                                                    for r in rows if r["arm"] == arm) for arm in arms},
               "fast_block_tokens": cfg["fast_block_tokens"],
               "native_prefix_tokens": native_prefix_tokens,
               "weight_sha256": mapper.manifest["weights_sha256"]}
    write("summary.json", summary)
    write("worker-status.json", summary)
    notice({"phase": "completed", "document": len(selected), "total": len(selected)})


if __name__ == "__main__":
    def interrupt(*_):
        raise TimeoutError("SageMaker worker deadline")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc())
        write("worker-status.json", {"status": "failed"})
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
