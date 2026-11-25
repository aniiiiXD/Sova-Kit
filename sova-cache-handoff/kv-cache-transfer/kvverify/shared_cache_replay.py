"""Score two question suffixes from one immutable mapped cache."""

from math import exp, isfinite
from statistics import median

from .evidence_replay import prepare_evidence_replay
from .hf_backend import legacy
from .quality import option_token_ids

FAST = "fast_only"
FUSED = "shared_lexical"
BALANCED = "shared_balanced"
ARMS = (FAST, FUSED, BALANCED)


def choose_option(*vectors):
    """Average conditional answer probabilities from independently scored suffixes."""
    labels = sorted(vectors[0])
    probabilities = dict.fromkeys(labels, 0.)
    for vector in vectors:
        if set(vector) != set(labels) or not all(isfinite(value) for value in vector.values()):
            raise ValueError("Invalid option logits in shared replay")
        peak = max(vector.values())
        shifted = {label: exp(vector[label] - peak) for label in labels}
        denominator = sum(shifted.values())
        for label in labels:
            probabilities[label] += shifted[label]/(len(vectors)*denominator)
    return max(labels, key=probabilities.__getitem__)


def shared_option_logits(runtime, mapper, source, prefix, suffix, select, evidence=None):
    from .fast_affine import map_multi_donor_fast
    from .native_prefix import replace_prefix

    expected = len(prefix)
    cut = evidence.cache_tokens if evidence else expected
    if source.get_seq_length() != expected or not 256 < cut <= expected:
        raise ValueError("Invalid shared source cache boundary")
    source_view = runtime.DynamicCache.from_legacy_cache(legacy(source))
    if cut < expected:
        source_view.crop(cut)
    head = runtime.forward(runtime.target, prefix[:256]).past_key_values
    mapped = map_multi_donor_fast(mapper, source_view, runtime.source, runtime.target)
    mapped = replace_prefix(mapped, head, 256)
    rows = legacy(mapped)
    suffixes = [prefix[cut:] + suffix]
    if evidence:
        suffixes.append(evidence.augmented_suffix)
    vectors = []
    for query in suffixes:
        view = runtime.DynamicCache.from_legacy_cache(rows)
        output = runtime.forward(runtime.target, query, view)
        vectors.append(select(output))
        del output, view
    if mapped.get_seq_length() != cut or any(k.shape[-2] != cut for k, _ in legacy(mapped)):
        raise ValueError("Suffix scoring changed the shared prefix")
    return vectors


def replay_question(runtime, mapper, item, prefix, suffix, rng, repeats):
    tokens = option_token_ids(runtime.tokenizer, item.choices)
    device = next(runtime.target.parameters()).device
    indices = runtime.torch.tensor(list(tokens.values()), device=device)

    def select(output):
        values = output.logits[0, -1].index_select(0, indices).float().cpu().tolist()
        return dict(zip(tokens, values))

    native, native_ms = runtime.timed(lambda: select(runtime.forward(runtime.target, prefix + suffix)))
    source = runtime.forward(runtime.source, prefix).past_key_values

    def request(arm):
        evidence = None
        if arm != FAST:
            evidence = prepare_evidence_replay(runtime, item, prefix, balanced=arm == BALANCED)
        vectors = shared_option_logits(runtime, mapper, source, prefix, suffix, select, evidence)
        return {"option_logits": vectors, "answer": choose_option(*vectors),
                "excerpt_source_spans": evidence.source_spans if evidence else (),
                "cached_tokens": evidence.cache_tokens if evidence else len(prefix)}

    rows = {arm: [] for arm in ARMS}
    order = rng.sample(ARMS, len(ARMS))
    for repeat in range(repeats):
        for arm in order if repeat % 2 == 0 else reversed(order):
            scored, elapsed = runtime.timed(lambda: request(arm))
            rows[arm].append({**scored, "ttft_ms": elapsed})
    return {"id": item.id, "native": {"answer": choose_option(native), "ttft_ms": native_ms},
            "prefix_tokens": len(prefix), "arms": rows}


def summarize(rows, cases, original_ids, original_fast_correct, repeats):
    identifiers = {row["id"] for row in rows}
    if (len(rows) != len(cases) or identifiers != set(cases)
            or not set(original_fast_correct) <= set(original_ids) <= identifiers):
        raise ValueError("Incomplete shared-cache replay")
    for row in rows:
        if set(row["arms"]) != set(ARMS):
            raise ValueError("Shared-cache arm set changed")
        for arm, trials in row["arms"].items():
            if len(trials) != repeats or any(
                len(trial["option_logits"]) != (1 if arm == FAST else 2)
                or choose_option(*trial["option_logits"]) != trial["answer"]
                or not isfinite(trial["ttft_ms"]) or trial["ttft_ms"] <= 0
                for trial in trials
            ):
                raise ValueError("Shared-cache predictions or timings are incomplete")

    def group(items):
        def correct(row, arm):
            return row["arms"][arm][0]["answer"] == cases[row["id"]]["gold"]

        return {
            "questions": len(items),
            "native_correct": sum(row["native"]["answer"] == cases[row["id"]]["gold"] for row in items),
            "native_ttft_ms": median(row["native"]["ttft_ms"] for row in items),
            "arms": {arm: {
                "correct": sum(correct(row, arm) for row in items),
                "rescues_vs_fast": sum(correct(row, arm) and not correct(row, FAST) for row in items),
                "losses_vs_fast": sum(correct(row, FAST) and not correct(row, arm) for row in items),
                "repeat_answer_agreement": sum(len({trial["answer"] for trial in row["arms"][arm]}) == 1
                                               for row in items),
                "ttft_ms": median(median(trial["ttft_ms"] for trial in row["arms"][arm]) for row in items),
            } for arm in ARMS},
        }

    original = group([row for row in rows if row["id"] in original_ids])
    for arm in ARMS:
        original["arms"][arm]["original_wins_lost"] = sum(
            row["arms"][arm][0]["answer"] != cases[row["id"]]["gold"]
            for row in rows if row["id"] in original_fast_correct)
    return {
        "all": group(rows),
        "original_native_correct": original,
        "hit_11_without_regression": [arm for arm in (FUSED, BALANCED)
                                      if original["arms"][arm]["correct"] >= 11
                                      and original["arms"][arm]["original_wins_lost"] == 0],
    }
