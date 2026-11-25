"""Replay frozen alternatives and solve the paired mean-latency oracle."""
from collections import Counter, defaultdict
from math import isfinite
from statistics import mean, median

from .evidence import percentile

NATIVE = "native_cold"
CONTROLS = (NATIVE, "source_hit", "native_short256")
TRANSFER = ("pure_transfer", "hybrid_transfer")
ARMS = (*CONTROLS, *TRANSFER, "native_full_hit")
COMPONENTS = (NATIVE, "native_full_hit", "hybrid_transfer", "native_keys", "native_values", "native_both")
TAIL_REPAIR = {"sink1_tail31": 31, "sink1_tail255": 255}
EVIDENCE_ARMS = ("native_evidence_only", "hybrid_evidence_transfer")


def benchmark_question(runtime, mapper, example, prefix, suffix, repeats, rng, *, tail_arms=(), corrected_mapper=None, evidence=None):
    from .fast_affine import map_affine_fast
    from .hf_backend import legacy
    from .native_prefix import replace_prefix
    from .quality import option_token_ids, score_option_logits

    source_cache = runtime.forward(runtime.source, prefix).past_key_values
    native_cache = runtime.forward(runtime.target, prefix).past_key_values
    tokens = option_token_ids(runtime.tokenizer, example.choices)
    labels = sorted(tokens)
    backend = runtime.torch
    indices = backend.tensor([tokens[label] for label in labels],
                             device=next(runtime.target.parameters()).device)
    requested_arms = (*ARMS, *tail_arms)
    if corrected_mapper is not None:
        requested_arms += ("corrected_transfer",)
    if evidence is not None:
        requested_arms += EVIDENCE_ARMS

    def native_head():
        return runtime.forward(runtime.target, prefix[:256]).past_key_values

    def query(model, cache):
        view = runtime.DynamicCache.from_legacy_cache(legacy(cache))
        return runtime.forward(model, suffix, view).logits[0, -1]

    def request(arm):
        if arm == NATIVE:
            logits = runtime.forward(runtime.target, prefix + suffix).logits[0, -1]
        elif arm == "source_hit":
            logits = query(runtime.source, source_cache)
        elif arm == "native_short256":
            logits = query(runtime.target, native_head())
        elif arm == "native_full_hit":
            logits = query(runtime.target, native_cache)
        elif arm == "native_evidence_only":
            logits = runtime.forward(runtime.target, evidence.short_ids).logits[0, -1]
        elif arm == "hybrid_evidence_transfer":
            source_view = runtime.DynamicCache.from_legacy_cache(legacy(source_cache))
            source_view.crop(evidence.cache_tokens)
            mapped = map_affine_fast(mapper, source_view, runtime.source, runtime.target, block_tokens=256)
            mapped = replace_prefix(mapped, native_head(), 256)
            logits = runtime.forward(runtime.target, evidence.augmented_suffix, mapped).logits[0, -1]
        elif arm in TAIL_REPAIR:
            tail = TAIL_REPAIR[arm]
            cut = len(prefix) - tail
            source_view = runtime.DynamicCache.from_legacy_cache(legacy(source_cache))
            source_view.crop(cut)
            mapped = map_affine_fast(mapper, source_view, runtime.source, runtime.target, block_tokens=256)
            first = runtime.forward(runtime.target, prefix[:1]).past_key_values
            cache = replace_prefix(mapped, first, 1)
            # Recompute the tail with 8B conditioned on the translated history.
            logits = runtime.forward(runtime.target, prefix[cut:] + suffix, cache).logits[0, -1]
        else:
            selected = corrected_mapper if arm == "corrected_transfer" else mapper
            mapped = map_affine_fast(selected, source_cache, runtime.source, runtime.target, block_tokens=256)
            if arm in ("hybrid_transfer", "corrected_transfer"):
                mapped = replace_prefix(mapped, native_head(), 256)
            logits = query(runtime.target, mapped)
        return logits, labels[int(logits.index_select(0, indices).argmax().item())]

    warmups = list(requested_arms)
    rng.shuffle(warmups)
    for arm in warmups:
        runtime.timed(lambda: request(arm))
    rows = []
    for repeat in range(repeats):
        arms = list(requested_arms)
        rng.shuffle(arms)
        for arm in arms:
            (logits, decision), elapsed = runtime.timed(lambda: request(arm))
            score = score_option_logits(logits, tokens, example.answer)
            if score["decision_answer"] != decision:
                raise ValueError("Timed option selection differs from the logged logits")
            rows.append({"id": example.id, "document_id": example.document_id,
                         "arm": arm, "repeat": repeat, "ttft_ms": elapsed,
                         "prefix_tokens": len(prefix), "suffix_tokens": len(suffix),
                         "option_logits": score["option_logits"], "answer": decision,
                         "option_mass": score["option_probability_mass"],
                         "unrestricted_first_token": score["full_vocab_top_token_id"],
                         "unrestricted_gold": score["full_vocab_top_token_id"] == tokens[example.answer]})
            if tail_arms:
                rows[-1]["target_history_tokens_recomputed"] = {
                    NATIVE: len(prefix), "hybrid_transfer": 256, "native_short256": 256,
                    **{name: 1 + tail for name, tail in TAIL_REPAIR.items()}}.get(arm, 0)
    if any(cache.get_seq_length() != len(prefix) for cache in (native_cache, source_cache)):
        raise ValueError("Request mutated an input cache")
    return rows


def probe_components(runtime, mapper, example, prefix, suffix, *, layer_groups=None):
    """Hold the first 256 native tokens and intervene on the remaining history."""
    from .boundary_probe import probability_equivalence
    from .fast_affine import map_affine_fast
    from .hf_backend import legacy
    from .native_prefix import oracle_components, replace_prefix
    from .quality import option_token_ids, score_option_logits

    def read(model, ids, cache=None):
        return runtime.forward(model, ids, cache).logits[0, -1].detach().float().cpu()

    def query(cache):
        return read(runtime.target, suffix, runtime.DynamicCache.from_legacy_cache(legacy(cache)))

    native_logits = read(runtime.target, prefix + suffix)
    native = runtime.forward(runtime.target, prefix).past_key_values
    warm_logits = query(native)
    head = runtime.forward(runtime.target, prefix[:256]).past_key_values
    aligned = replace_prefix(native, head, 256)
    del native
    source = runtime.forward(runtime.source, prefix).past_key_values
    mapped = map_affine_fast(mapper, source, runtime.source, runtime.target, block_tokens=256)
    del source
    hybrid = replace_prefix(mapped, head, 256)
    del mapped, head
    vectors = {NATIVE: native_logits, "native_full_hit": warm_logits, "hybrid_transfer": query(hybrid)}
    for arm, keys, values in (("native_keys", True, False), ("native_values", False, True), ("native_both", True, True)):
        # Both inputs already have the identical native 256-token head.
        cache = oracle_components(hybrid, aligned, keys=keys, values=values, retained_native_tokens=0)
        vectors[arm] = query(cache)
        del cache
    if layer_groups:
        mapped_rows = legacy(hybrid)
        native_rows = legacy(aligned)
        for name, layers in layer_groups.items():
            if not layers or min(layers) < 0 or max(layers) >= len(mapped_rows):
                raise ValueError("Invalid native layer intervention")
            cache = runtime.DynamicCache.from_legacy_cache(tuple(
                native_rows[i] if i in layers else pair for i, pair in enumerate(mapped_rows)))
            vectors[name] = query(cache)
    tokens = option_token_ids(runtime.tokenizer, example.choices)
    arms = {}
    for arm, vector in vectors.items():
        score = score_option_logits(vector, tokens, example.answer)
        distance = probability_equivalence(runtime.torch, native_logits, vector)
        arms[arm] = {"answer": score["decision_answer"], "option_logits": score["option_logits"],
                     "option_mass": score["option_probability_mass"],
                     "native_kl": distance["native_to_candidate_kl_nats"]}
    return {"id": example.id, "document_id": example.document_id, "prefix_tokens": len(prefix), "arms": arms}


def trace_readout(runtime, mapper, example, prefix, suffix):
    """Compare actual projected attention outputs along each path's query trajectory."""
    from .fast_affine import map_affine_fast
    from .hf_backend import legacy
    from .native_prefix import replace_prefix

    positions = sorted({0, len(suffix)//2, len(suffix)-1})
    target = runtime.target

    def trace(cache):
        values = []

        def capture(_module, _args, output):
            values.append(output[0][0, positions].detach().float().cpu())

        handles = [layer.self_attn.register_forward_hook(capture) for layer in target.model.layers]
        try:
            runtime.forward(target, suffix, runtime.DynamicCache.from_legacy_cache(legacy(cache)))
        finally:
            for handle in handles:
                handle.remove()
        if len(values) != len(handles):
            raise ValueError("Attention output hooks did not cover the target layers")
        return values

    native = runtime.forward(target, prefix).past_key_values
    native_outputs = trace(native)
    del native
    source = runtime.forward(runtime.source, prefix).past_key_values
    mapped = map_affine_fast(mapper, source, runtime.source, target, block_tokens=256)
    del source
    head = runtime.forward(target, prefix[:256]).past_key_values
    hybrid = replace_prefix(mapped, head, 256)
    del mapped, head
    hybrid_outputs = trace(hybrid)
    torch = runtime.torch
    layers = []
    for index, (reference, candidate) in enumerate(zip(native_outputs, hybrid_outputs)):
        error = (candidate-reference).norm(dim=-1)/reference.norm(dim=-1).clamp_min(1e-8)
        cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
        layers.append({"layer": index, "relative_l2": error.tolist(), "cosine": cosine.tolist()})
    return {"id": example.id, "query_positions": [len(prefix)+i for i in positions], "layers": layers}


def select_component_cases(rows, gold, repeats):
    """Select every case where both cheap paths and both transfers missed native-correct gold."""
    summarize(rows, gold, repeats)
    predictions = defaultdict(dict)
    for row in rows:
        if row["repeat"] == 0:
            predictions[row["id"]][row["arm"]] = row["answer"]
    return {identifier: {NATIVE: arms[NATIVE], "hybrid_transfer": arms["hybrid_transfer"]}
            for identifier, arms in sorted(predictions.items())
            if arms[NATIVE] == gold[identifier]
            and all(arms[arm] != gold[identifier] for arm in (*CONTROLS[1:], *TRANSFER))}


def summarize_components(rows, gold, cases, *, layer_groups=None):
    if len(rows) != len(cases) or {row["id"] for row in rows} != set(cases):
        raise ValueError("Incomplete or duplicated component probe")
    scores = defaultdict(list)
    patterns = Counter()
    for row in rows:
        arms = row["arms"]
        identifier = row["id"]
        expected = cases[identifier]
        if set(arms) != set(COMPONENTS).union(layer_groups or {}) or any(arms[arm]["answer"] != expected[arm] for arm in expected):
            raise ValueError("Probe did not reproduce the parent experiment")
        label = gold[identifier]
        if any(arms[arm]["answer"] != label for arm in (NATIVE, "native_full_hit", "native_both")):
            raise ValueError("Fully native controls did not reproduce the gold answer")
        for arm, score in arms.items():
            options = score["option_logits"]
            if (set(options) != set("ABCD") or not all(isfinite(value) for value in options.values())
                    or score["answer"] != max(sorted(options), key=options.get)
                    or not isfinite(score["native_kl"])):
                raise ValueError("Invalid component score")
            margin = options[label] - max(value for key, value in options.items() if key != label)
            scores[arm].append((score["answer"] == label, margin, score["native_kl"]))
        keys = arms["native_keys"]["answer"] == label
        values = arms["native_values"]["answer"] == label
        patterns[{(True, False): "keys_only", (False, True): "values_only", (True, True): "either",
                  (False, False): "both_required"}[keys, values]] += 1
    result = {"questions": len(cases), "native_controls_passed": True, "recovery_patterns": dict(patterns),
              "arms": {arm: {"correct": sum(correct for correct, _, _ in trace),
                             "median_gold_margin": median(margin for _, margin, _ in trace),
                             "mean_native_kl": mean(kl for _, _, kl in trace)} for arm, trace in scores.items()}}
    if layer_groups:
        baseline = scores["hybrid_transfer"]
        result["layer_groups"] = {name: {"layers": layers,
                                        "mean_paired_gold_margin_gain": mean(
                                            margin - prior[1] for (_, margin, _), prior in zip(scores[name], baseline)),
                                        "rescued": sum(correct and not prior[0]
                                                       for (correct, _, _), prior in zip(scores[name], baseline))}
                                  for name, layers in layer_groups.items()}
    return result


def timing_stats(samples):
    return {"mean": mean(samples), "p50": median(samples), "p95": percentile(samples, .95)}


def oracle(requests, allowed, native_correct, *, preserve_native):
    """Minimize total mean TTFT under aggregate or per-request native gold correctness."""
    chosen, upgrades = [], []
    for index, by_arm in enumerate(requests):
        arms = [(mean(by_arm[arm]["samples"]), not by_arm[arm]["correct"], arm) for arm in allowed]
        cheapest = min(arms)
        correct = min((entry for entry in arms if not entry[1]), default=None)
        if preserve_native and by_arm[NATIVE]["correct"]:
            chosen.append(correct[2])
        else:
            chosen.append(cheapest[2])
            if cheapest[1] and correct is not None:
                upgrades.append((correct[0] - cheapest[0], index, correct[2]))
    if not preserve_native:
        need = max(0, native_correct - sum(row[arm]["correct"] for row, arm in zip(requests, chosen)))
        if need > len(upgrades):
            raise ValueError("Oracle cannot reach the native quality floor")
        for _, index, arm in sorted(upgrades)[:need]:
            chosen[index] = arm
    selected = [row[arm] for row, arm in zip(requests, chosen)]
    return {"correct": sum(row["correct"] for row in selected),
            "native_correct_lost": sum(row[NATIVE]["correct"] and not choice["correct"]
                                       for row, choice in zip(requests, selected)),
            "route_counts": dict(Counter(chosen)),
            "ttft_ms": timing_stats([sample for row in selected for sample in row["samples"]])}


def summarize(rows, gold, repeats, *, tail_cases=None, tail_arms=(), corrected=False, evidence=False):
    corrected_arms = ("corrected_transfer",) if corrected else ()
    required_arms = (*ARMS, *tail_arms, *corrected_arms, *(EVIDENCE_ARMS if evidence else ()))
    groups = defaultdict(dict)
    for row in rows:
        request_id, arm, repeat = row["id"], row["arm"], row["repeat"]
        options = row["option_logits"]
        if (request_id not in gold or arm not in required_arms or repeat not in range(repeats)
                or (arm, repeat) in groups[request_id] or set(options) != set("ABCD")
                or not all(isfinite(v) for v in options.values())
                or row["answer"] != max(sorted(options), key=options.get)
                or not isfinite(row["ttft_ms"]) or row["ttft_ms"] <= 0):
            raise ValueError("Malformed or duplicated paired replay row")
        if arm in EVIDENCE_ARMS:
            cpu_ms, model_ms = row["retrieval_and_tokenization_ms"], row["model_ttft_ms"]
            if not (isfinite(cpu_ms) and cpu_ms >= 0 and isfinite(model_ms) and model_ms > 0
                    and row["ttft_ms"] == cpu_ms + model_ms):
                raise ValueError("Evidence route timing must include separately recorded CPU preparation")
        if arm in tail_arms and row["target_history_tokens_recomputed"] != 1+TAIL_REPAIR[arm]:
            raise ValueError("Tail comparison does not match the declared native token budget")
        groups[request_id][arm, repeat] = row
    if set(groups) != set(gold) or any(len(group) != len(required_arms) * repeats for group in groups.values()):
        raise ValueError("Incomplete paired replay")
    requests = []
    for request_id, group in sorted(groups.items()):
        arms = {}
        for arm in required_arms:
            samples = [group[arm, repeat] for repeat in range(repeats)]
            if len({row["answer"] for row in samples}) != 1:
                raise ValueError("Repeated decoding changed the winning option")
            arms[arm] = {"correct": samples[0]["answer"] == gold[request_id],
                         "answer": samples[0]["answer"],
                         "unrestricted_gold": sum(row["unrestricted_gold"] for row in samples),
                         "samples": [row["ttft_ms"] for row in samples]}
        requests.append(arms)
    summaries = {arm: {"correct": sum(row[arm]["correct"] for row in requests),
                       "unrestricted_gold": sum(row[arm]["unrestricted_gold"] for row in requests),
                       "ttft_ms": timing_stats([sample for row in requests for sample in row[arm]["samples"]])}
                 for arm in required_arms}
    native_correct = summaries[NATIVE]["correct"]
    result = {"questions": len(gold), "timed_requests": len(rows), "repeats": repeats,
              "arms": summaries, "native_warm_agreement": sum(row[NATIVE]["answer"] == row["native_full_hit"]["answer"]
                                                              for row in requests),
              "oracles": {mode: {"without_translation": oracle(requests, CONTROLS, native_correct, preserve_native=preserve),
                                 "with_translation": oracle(requests, (*CONTROLS, *TRANSFER), native_correct, preserve_native=preserve)}
                          for mode, preserve in (("match_native_aggregate", False), ("preserve_native_correct", True))}}
    if evidence:
        cheap = (*CONTROLS, "native_evidence_only")
        prior_transfer = (*cheap, *TRANSFER)
        transferred = (*prior_transfer, "hybrid_evidence_transfer")
        for mode, preserve in (("match_native_aggregate", False), ("preserve_native_correct", True)):
            result["oracles"][mode]["with_native_evidence_only"] = oracle(
                requests, cheap, native_correct, preserve_native=preserve)
            result["oracles"][mode]["with_evidence_and_original_transfer"] = oracle(
                requests, prior_transfer, native_correct, preserve_native=preserve)
            result["oracles"][mode]["with_evidence_transfer"] = oracle(
                requests, transferred, native_correct, preserve_native=preserve)
        result["evidence"] = {
            "native_correct_recovered_by_excerpts": sum(row[NATIVE]["correct"] and row["native_evidence_only"]["correct"]
                                                        for row in requests),
            "native_correct_recovered_by_transfer_and_excerpts": sum(
                row[NATIVE]["correct"] and row["hybrid_evidence_transfer"]["correct"] for row in requests),
            "native_correct_unique_to_transfer_and_excerpts": sum(
                row[NATIVE]["correct"] and row["hybrid_evidence_transfer"]["correct"]
                and all(not row[arm]["correct"] for arm in (*cheap[1:], *TRANSFER)) for row in requests),
        }
    if corrected:
        for mode, preserve in (("match_native_aggregate", False), ("preserve_native_correct", True)):
            result["oracles"][mode]["with_corrected_mapper"] = oracle(
                requests, (*CONTROLS, *TRANSFER, *corrected_arms), native_correct, preserve_native=preserve)
        cases = [row for row in requests if row[NATIVE]["correct"]
                 and all(not row[arm]["correct"] for arm in (*CONTROLS[1:], *TRANSFER))]
        result["correction"] = {
            "known_failures": len(cases),
            "rescued_known_failures": sum(row["corrected_transfer"]["correct"] for row in cases),
            "recovered_native_correct": sum(row[NATIVE]["correct"] and row["corrected_transfer"]["correct"]
                                            for row in requests),
            "old_hybrid_native_correct": sum(row[NATIVE]["correct"] and row["hybrid_transfer"]["correct"]
                                             for row in requests),
        }
        if evidence:
            baseline = (*CONTROLS, *TRANSFER, *EVIDENCE_ARMS)
            for mode, preserve in (("match_native_aggregate", False), ("preserve_native_correct", True)):
                result["oracles"][mode]["with_evidence_and_corrected_mapper"] = oracle(
                    requests, (*baseline, "corrected_transfer"), native_correct, preserve_native=preserve)
            result["correction"]["native_correct_unique_beyond_evidence"] = sum(
                row[NATIVE]["correct"] and row["corrected_transfer"]["correct"]
                and all(not row[arm]["correct"] for arm in baseline if arm != NATIVE) for row in requests)
    if tail_arms:
        for mode, preserve in (("match_native_aggregate", False), ("preserve_native_correct", True)):
            result["oracles"][mode]["with_tail_repair"] = oracle(
                requests, (*CONTROLS, *TRANSFER, *tail_arms), native_correct, preserve_native=preserve)
        if tail_cases:
            chosen = {identifier: requests[index] for index, identifier in enumerate(sorted(groups)) if identifier in tail_cases}
            if set(chosen) != set(tail_cases) or any(row[NATIVE]["answer"] != tail_cases[identifier][NATIVE]
                    or row["hybrid_transfer"]["answer"] != tail_cases[identifier]["hybrid_transfer"]
                    for identifier, row in chosen.items()):
                raise ValueError("Tail replay did not reproduce the six failure controls")
            cases = list(chosen.values())
        else:
            cases = [row for row in requests if row[NATIVE]["correct"]
                     and all(not row[arm]["correct"] for arm in (*CONTROLS[1:], *TRANSFER))]
        result["tail_repair"] = {"known_failures": len(cases), "variants": {
            arm: {"rescued_known_failures": sum(row[arm]["correct"] for row in cases),
                  "rescued_vs_hybrid": sum(row[arm]["correct"] and not row["hybrid_transfer"]["correct"] for row in requests),
                  "lost_vs_hybrid": sum(not row[arm]["correct"] and row["hybrid_transfer"]["correct"] for row in requests)}
            for arm in tail_arms}}
    return result
