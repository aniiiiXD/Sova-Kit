"""Compare a frozen residual and its zero-residual control without training."""

import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.probe_worker import OUT, notice, upload, write

ARMS = ("single_donor", "zero_residual", "trained")
FAST_ARM = "fast_trained"
EVIDENCE_ARM = "fast_trained_with_excerpts"
FRESH_CONTROLS = ("source_hit", "native_hit", "native_excerpts")
FRESH_ARMS = ("single_donor", "trained", FAST_ARM, *FRESH_CONTROLS)
STAGES = ("ttft_ms", "cache_preparation_ms", "mapper_ms", "query_ms")


def replay_arms(compare_fast, reserved_panel, with_excerpts):
    if with_excerpts:
        return (*FRESH_ARMS, EVIDENCE_ARM)
    if reserved_panel:
        return FRESH_ARMS
    return (*ARMS, FAST_ARM) if compare_fast else ARMS


def summarize(rows, cases, repeats, *, compare_fast=False, reserved_panel=False, with_excerpts=False):
    if len(rows) != len(cases) or {row["id"] for row in rows} != set(cases):
        raise ValueError("Incomplete paired replay")
    full_controls = reserved_panel or with_excerpts
    expected_arms = replay_arms(compare_fast, reserved_panel, with_excerpts)

    def margin(logits, gold):
        return logits[gold] - max(value for letter, value in logits.items() if letter != gold)

    scored = []
    for row in rows:
        case = cases[row["id"]]
        gold = case["gold"]
        if set(row["arms"]) != set(expected_arms) or any(len(trials) != repeats for trials in row["arms"].values()):
            raise ValueError("Incomplete arm repetitions")
        native = margin(row["native"]["option_logits"], gold)
        arms = {}
        for arm, trials in row["arms"].items():
            margins = [margin(trial["option_logits"], gold) for trial in trials]
            correct = [trial["answer"] == gold for trial in trials]
            if len(set(correct)) != 1:
                raise ValueError("Paired repetitions disagree on correctness")
            arms[arm] = {
                "correct": correct[0],
                "gold_margin": median(margins),
                **{stage: median(trial[stage] for trial in trials) for stage in STAGES},
            }
        record = {"id": row["id"], "native_correct": row["native"]["answer"] == gold,
                  "native_gold_margin": native, "historical_hard": case["historical_hard"], "arms": arms}
        if compare_fast:
            reference, fast = row["arms"]["trained"], row["arms"][FAST_ARM]
            record["fast_option_agreement"] = all(a["answer"] == b["answer"] for a, b in zip(reference, fast))
            record["fast_max_option_logit_delta"] = max(
                abs(a["option_logits"][label] - b["option_logits"][label])
                for a, b in zip(reference, fast) for label in a["option_logits"])
        if full_controls:
            record["native_cold_ttft_ms"] = row["native"]["ttft_ms"]
        scored.append(record)

    def group(items):
        if not items:
            return {"questions": 0}
        baseline = "single_donor" if full_controls else "zero_residual"
        comparison = "single_donor" if full_controls else "zero"
        pairs = [(row["arms"][baseline], row["arms"]["trained"]) for row in items]
        deltas = [trained["gold_margin"] - zero["gold_margin"] for zero, trained in pairs]
        result = {
            "questions": len(items),
            f"trained_rescues_vs_{comparison}": sum(not old["correct"] and new["correct"] for old, new in pairs),
            f"trained_losses_vs_{comparison}": sum(old["correct"] and not new["correct"] for old, new in pairs),
            "margin_improved": sum(delta > 0 for delta in deltas),
            "margin_worsened": sum(delta < 0 for delta in deltas),
            "median_margin_delta": median(deltas),
            "arms": {
                arm: {
                    "correct": sum(row["arms"][arm]["correct"] for row in items),
                    **{name: median(row["arms"][arm][name] for row in items) for name in ("gold_margin", *STAGES)},
                }
                for arm in expected_arms
            },
        }
        if compare_fast:
            result["fast_option_agreement"] = sum(row["fast_option_agreement"] for row in items)
            result["fast_max_option_logit_delta"] = max(row["fast_max_option_logit_delta"] for row in items)
        if full_controls:
            cheap_misses = [row for row in items if not any(row["arms"][arm]["correct"]
                            for arm in ("single_donor", "source_hit", "native_excerpts"))]
            result.update(
                native_cold_correct=sum(row["native_correct"] for row in items),
                native_cold_ttft_ms=median(row["native_cold_ttft_ms"] for row in items),
                missed_by_cheaper_target_miss_routes=len(cheap_misses),
                fast_rescues_after_cheaper_routes_miss=sum(row["arms"][FAST_ARM]["correct"] for row in cheap_misses),
            )
        if with_excerpts:
            rescue = [row for row in items if not row["arms"][FAST_ARM]["correct"] and row["arms"][EVIDENCE_ARM]["correct"]]
            only_combination = [row for row in rescue if not row["arms"]["native_excerpts"]["correct"]]
            result["excerpt_effect"] = {
                "rescues_vs_fast": len(rescue),
                "losses_vs_fast": sum(row["arms"][FAST_ARM]["correct"] and not row["arms"][EVIDENCE_ARM]["correct"]
                                     for row in items),
                "rescues_beyond_both_inputs": len(only_combination),
                "rescues_beyond_all_target_miss_controls": sum(
                    not (row["arms"]["source_hit"]["correct"] or row["arms"]["single_donor"]["correct"])
                    for row in only_combination),
                "median_margin_delta_vs_fast": median(row["arms"][EVIDENCE_ARM]["gold_margin"]
                                                      - row["arms"][FAST_ARM]["gold_margin"] for row in items),
            }
        return result

    eligible = [row for row in scored if row["native_correct"]]
    count_key = "reserved_documents" if reserved_panel else "historically_native_correct"
    if with_excerpts:
        count_key = "development_documents"
    return {count_key: len(scored), "native_correct_this_run": len(eligible),
            "all": group(scored), "native_correct": group(eligible),
            "historical_hard_native_correct": group([row for row in eligible if row["historical_hard"]]),
            "questions": scored}


def replay_question(runtime, base, zero, trained, item, prefix, suffix, rng, repeats, *,
                    compare_fast=False, reserved_panel=False, with_excerpts=False):
    from time import perf_counter

    from kvverify.evidence_replay import prepare_evidence_replay
    from kvverify.fast_affine import map_affine_fast, map_multi_donor_fast
    from kvverify.hf_backend import legacy
    from kvverify.native_prefix import replace_prefix
    from kvverify.quality import option_token_ids, score_option_logits

    torch = runtime.torch
    tokens = option_token_ids(runtime.tokenizer, item.choices)

    def score(output):
        result = score_option_logits(output.logits[0, -1], tokens, item.answer)
        return {"answer": result["decision_answer"], "option_logits": result["option_logits"]}

    evidence = None
    evidence_ms = 0.
    if with_excerpts:
        start = perf_counter()
        evidence = prepare_evidence_replay(runtime, item, prefix)
        evidence_ms = (perf_counter() - start)*1000
    full_controls = reserved_panel or with_excerpts
    if full_controls:
        output, elapsed = runtime.timed(lambda: runtime.forward(runtime.target, prefix + suffix))
        native = {**score(output), "ttft_ms": elapsed}
        del output
        controls = replay_fresh_controls(runtime, item, prefix, suffix, repeats)
    else:
        native = score(runtime.forward(runtime.target, prefix + suffix))
        controls = {}
    source = runtime.forward(runtime.source, prefix).past_key_values
    paths = {
        "single_donor": lambda: map_affine_fast(base, source, runtime.source, runtime.target),
        "trained": lambda: trained.apply(source, runtime.source, runtime.target),
    }
    if not full_controls:
        paths["zero_residual"] = lambda: zero.apply(source, runtime.source, runtime.target)
        paths = {arm: paths[arm] for arm in ARMS}
    if compare_fast:
        paths[FAST_ARM] = lambda: map_multi_donor_fast(trained, source, runtime.source, runtime.target)
    if evidence is not None:
        def map_with_excerpts():
            cropped = runtime.DynamicCache.from_legacy_cache(legacy(source))
            cropped.crop(evidence.cache_tokens)
            return map_multi_donor_fast(trained, cropped, runtime.source, runtime.target)

        paths[EVIDENCE_ARM] = map_with_excerpts

    def trial(arm):
        start, before_map, after_map, before_query, end = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        query = evidence.augmented_suffix if arm == EVIDENCE_ARM else suffix

        def request():
            start.record()
            head = runtime.forward(runtime.target, prefix[:256]).past_key_values
            before_map.record()
            mapped = paths[arm]()
            after_map.record()
            cache = replace_prefix(mapped, head, 256)
            before_query.record()
            output = runtime.forward(runtime.target, query, cache)
            end.record()
            return output

        output, elapsed = runtime.timed(request)
        excerpt_ms = evidence_ms if arm == EVIDENCE_ARM else 0.
        result = {**score(output), "ttft_ms": elapsed + excerpt_ms, "cache_preparation_ms": start.elapsed_time(before_query),
                  "mapper_ms": before_map.elapsed_time(after_map), "query_ms": before_query.elapsed_time(end)}
        if arm == EVIDENCE_ARM:
            result["excerpt_preparation_ms"] = excerpt_ms
        return result

    arms = tuple(paths)
    rows = {arm: [] for arm in arms}
    for repeat in range(repeats):
        order = rng.sample(arms, len(arms)) if repeat % 2 == 0 else order[::-1]
        for arm in order:
            rows[arm].append(trial(arm))
    record = {"id": item.id, "prefix_tokens": len(prefix), "native": native, "arms": {**rows, **controls}}
    if evidence is not None:
        record["excerpt_source_spans"] = evidence.source_spans
        record["excerpt_query_tokens"] = len(evidence.augmented_suffix)
    return record


def replay_fresh_controls(runtime, item, prefix, suffix, repeats):
    from dataclasses import replace
    from time import perf_counter

    from kvverify.evidence_replay import select_excerpts
    from kvverify.hf_backend import legacy
    from kvverify.quality import option_token_ids, score_option_logits

    tokens = option_token_ids(runtime.tokenizer, item.choices)
    rows = {arm: [] for arm in FRESH_CONTROLS}
    for arm in FRESH_CONTROLS:
        model = runtime.source if arm == "source_hit" else runtime.target
        cache = runtime.forward(model, prefix).past_key_values if arm != "native_excerpts" else None
        for _ in range(repeats):
            preparation_ms = 0.
            if cache is None:
                start = perf_counter()
                chunks, _ = select_excerpts(runtime.tokenizer, item.context, item.question, item.choices)
                excerpt = replace(item, context="\n\n[… omitted …]\n\n".join(chunks))
                ids = list(runtime.tokenize(excerpt))
                preparation_ms = (perf_counter() - start)*1000
                view = None
            else:
                ids = suffix
                view = runtime.DynamicCache.from_legacy_cache(legacy(cache))
            output, elapsed = runtime.timed(lambda: runtime.forward(model, ids, view))
            result = score_option_logits(output.logits[0, -1], tokens, item.answer)
            rows[arm].append({
                "answer": result["decision_answer"], "option_logits": result["option_logits"],
                "ttft_ms": preparation_ms + elapsed, "cache_preparation_ms": 0., "mapper_ms": 0.,
                "query_ms": elapsed, "excerpt_preparation_ms": preparation_ms,
            })
            del output, view
        del cache
    return rows


def main():
    import torch
    from huggingface_hub import snapshot_download

    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.continuation_pilot import MultiDonorMapper
    from kvverify.data import check_heldout, load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.qa_training import split_document_prompt
    import kvverify.shared_cache_replay as shared_cache_replay

    cfg = json.loads(Path(os.environ["KV_CONFIG"]).read_text())
    if cfg["mode"] != "mapper_ablation" or not torch.cuda.is_bf16_supported():
        raise ValueError("The mapper replay requires BF16 CUDA")
    for name, expected in cfg["input_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen replay input changed: " + name)
    dataset = json.loads((ROOT/"inputs/manifest.json").read_text())
    compare_fast = cfg.get("compare_fast", False)
    reserved_panel = cfg.get("reserved_panel", False)
    with_excerpts = cfg.get("with_excerpts", False)
    shared_cache = cfg.get("shared_cache", False)
    cases = cfg["cases"]
    examples = {item.id: item for item in load_examples(ROOT/"inputs/evaluation.jsonl", "eval")}
    selected = [examples[name] for name in cases]
    frozen = {item["id"]: item for item in dataset["selected"]}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = []
    for item in cfg["models"]:
        notice({"phase": "downloading_model", "model": item["repo"]})
        paths.append(snapshot_download(item["repo"], revision=item["revision"], token=False,
                                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
                                      max_workers=4))
    notice({"phase": "loading_models"})
    runtime = DenseRuntime.from_local(*paths, dtype="bfloat16")
    if runtime.tokenizer_sha256 != dataset["tokenizer_fingerprint"]:
        raise ValueError("The replay tokenizer differs from the freeze")
    base = AffineCacheMapper.load(cfg["input_mapper_dir"], runtime.identities["source"]["sha256"],
                                  runtime.identities["target"]["sha256"], runtime.tokenizer_sha256).to("cuda:0")
    if base.manifest["weights_sha256"] != cfg["base_weights_sha256"]:
        raise ValueError("Unexpected affine initializer")
    checkpoint = Path("/opt/ml/input/data/corrected")
    trained_manifest = json.loads((checkpoint/"manifest.json").read_text())
    if trained_manifest != json.loads((ROOT/"inputs/checkpoint-manifest.json").read_text()):
        raise ValueError("Remote residual manifest differs from the frozen checkpoint")
    check_heldout(base.manifest, selected)
    check_heldout(trained_manifest, selected)
    trained = MultiDonorMapper.load(checkpoint, base).requires_grad_(False)
    zero = None
    if not (reserved_panel or with_excerpts or shared_cache):
        zero = MultiDonorMapper(base, rank=trained.down.shape[-1], donors=trained.down.shape[1]).eval()
        zero.requires_grad_(False)
        zero.down.copy_(trained.down)
    arms = shared_cache_replay.ARMS if shared_cache else replay_arms(compare_fast, reserved_panel, with_excerpts)
    scope = ("Development-only replay. Source cache resident. End-to-end option scoring includes excerpt "
             "selection, one mapper call per arm, and both suffix passes for the fused arms."
             if shared_cache else ("Reserved document replication. " if reserved_panel else "Development-only replay. ")
             + "Constrained first option. Excerpt paths charge CPU selection. "
               "Transfer starts from resident source cache and models. Wall TTFT excludes scoring. "
               "CUDA intervals report cache preparation, mapping alone, and query.")
    write("protocol.json", {
        "arms": list(arms),
        "repeats": cfg["repeats"], "cases": cases, "models": runtime.identities,
        "device": torch.cuda.get_device_name(0), "trained_weights_sha256": trained_manifest["weights_sha256"],
        "scope": scope,
    })
    rows = []
    rng = random.Random(cfg["seed"])
    with torch.inference_mode():
        for index, item in enumerate(selected):
            notice({"phase": "paired_replay", "ready": index, "total": len(selected)})
            full, prefix, suffix = split_document_prompt(runtime, item)
            expected = frozen[item.id]
            if (item.answer != cases[item.id]["gold"] or item.context_hash != expected["context_sha256"]
                    or len(full) != expected["prompt_tokens"] or not 256 < len(prefix) < len(full)):
                raise ValueError("Prompt or label drift: " + item.id)
            if shared_cache:
                row = shared_cache_replay.replay_question(runtime, trained, item, prefix, suffix, rng, cfg["repeats"])
            else:
                row = replay_question(runtime, base, zero, trained, item, prefix, suffix, rng, cfg["repeats"],
                                      compare_fast=compare_fast, reserved_panel=reserved_panel,
                                      with_excerpts=with_excerpts)
            rows.append(row)
            write("rows.json", rows)
    result = (shared_cache_replay.summarize(rows, cases, cfg["original_native_correct"], cfg["original_fast_correct"],
                                           cfg["repeats"])
              if shared_cache else summarize(rows, cases, cfg["repeats"], compare_fast=compare_fast,
                                            reserved_panel=reserved_panel, with_excerpts=with_excerpts))
    write("summary.json", result)
    notice({"phase": "completed", "ready": len(rows), "total": len(selected)})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc())
        upload()
        raise
