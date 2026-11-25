"""Run the cache agent graph locally and screen the archived SageMaker candidate."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import NamedTemporaryFile

from .agent_graph import Controller, GraphFailed, Node
from .cache_handoff import CacheScope, NativePrefixRegistry
from .handoff import CacheHandoff

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/quality-8b-v1/collected"
QUALITY_MANIFEST = ROOT / "datasets/loop-holdout-v1/manifest.json"
QUALITY_ROWS = QUALITY_MANIFEST.with_name("quality.jsonl")
SCOPES = {"source": CacheScope("fixture-tenant", "tiny-worker-v1", "fixture-token-ids", "dynamic-fp32"),
          "target": CacheScope("fixture-tenant", "tiny-reviewer-v1", "fixture-token-ids", "dynamic-fp32")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_contract(_):
    return {"scope_fields": ["tenant", "model", "tokenizer", "layout"],
            "position_origin": 0, "origin": "native", "branch_state": "private clone"}


def trace(_):
    return {"evidence": [4, 8, 12, 16, 20, 24],
            "worker_suffix": [3, 7],
            "review_suffixes": [[2, 5], [19, 23], [2, 5]],
            "routes": ["native_cold", "native_partial", "native_full"]}


def registry_checks(inputs):
    from copy import deepcopy
    from types import SimpleNamespace

    now = [0.0]
    registry = NativePrefixRegistry(deepcopy, max_bytes=8, ttl_seconds=2, clock=lambda: now[0])

    class FixtureState(SimpleNamespace):
        def get_seq_length(self):
            return len(self.tokens)

    native = SCOPES["target"]
    saved = FixtureState(tokens=[1, 2])
    registry.store_native(native, saved.tokens, saved, byte_count=4)
    saved.tokens[0] = 99
    registry.store_native(native, [1, 2, 3], FixtureState(tokens=[1, 2, 3]), byte_count=4)
    first = registry.longest_native(native, [1, 2, 3, 4])
    first.state.tokens[0] = 80
    checks = {
        "longest_partial_prefix": first.token_count == 3,
        "private_mutable_state": registry.longest_native(native, [1, 2, 3]).state.tokens == [1, 2, 3],
        "producer_mutation_isolated": registry.longest_native(native, [1, 2]).state.tokens == [1, 2],
        "wrong_prefix_misses": registry.longest_native(native, [2, 1]).state is None,
        "wrong_checkpoint_misses": registry.longest_native(SCOPES["source"], [1, 2, 3]).state is None,
        "cross_tenant_misses": registry.longest_native(
            CacheScope("another-tenant", native.model, native.tokenizer, native.layout),
            [1, 2, 3]).state is None,
    }
    registry.store_native(native, [7, 8], FixtureState(tokens=[7, 8]), byte_count=4)
    checks["lru_evicts_to_capacity"] = registry.longest_native(native, [1, 2, 3]).token_count == 2
    now[0] = 2.0
    checks["expiration_misses"] = registry.longest_native(native, [1, 2, 3]).state is None
    checks["empty_lookup_misses"] = registry.longest_native(native, []).state is None
    try:
        registry.longest_native(native, [True, 2])
    except ValueError:
        checks["invalid_token_type_rejected"] = True
    else:
        checks["invalid_token_type_rejected"] = False
    if not all(checks.values()):
        raise ValueError("Registry controls failed: " + str(checks))
    return {"contract": inputs["cache_contract"], "checks": checks}


def replay_native(inputs):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from .hf_backend import DenseRuntime, legacy

    torch.set_num_threads(1)

    def model():
        cfg = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, max_position_embeddings=128,
                          use_sliding_window=False, sliding_window=None)
        cfg._attn_implementation = "sdpa"
        return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        runtime = DenseRuntime(model(), model())
    registry = NativePrefixRegistry(runtime.clone_cache, max_bytes=1_000_000, ttl_seconds=300)
    spec = inputs["trace"]
    evidence = spec["evidence"]

    def advance(scope, tokens, cache):
        model = {SCOPES["source"]: runtime.source, SCOPES["target"]: runtime.target}[scope]
        result = runtime.forward(model, tokens, cache)
        return result.logits, result.past_key_values

    def state_bytes(state):
        return sum(t.numel() * t.element_size() for layer in legacy(state) for t in layer)

    handoff = CacheHandoff(registry, advance, state_bytes)

    source = runtime.forward(runtime.source, evidence).past_key_values
    handoff.remember(SCOPES["source"], evidence, source)
    worker = runtime.forward(runtime.source, spec["worker_suffix"], runtime.clone_cache(source))

    def answer(suffix, expected_count):
        response = handoff.run(SCOPES["target"], evidence, suffix)
        if response.reused_tokens != expected_count:
            raise ValueError("Unexpected native cache match length")
        result = response.output
        ref = runtime.forward(runtime.target, evidence + suffix).logits
        torch.testing.assert_close(result, ref, atol=1e-6, rtol=1e-4)
        return {"native_prefix_tokens": response.reused_tokens, "prefilled_tokens": response.processed_tokens,
                "answer_token": int(result[0, -1].argmax()), "max_logit_delta": float((result-ref).abs().max())}

    suffix_a, suffix_b, suffix_a_again = spec["review_suffixes"]
    if suffix_a != suffix_a_again:
        raise ValueError("Interleaved replay lost the A/B/A suffix sequence")
    cold = answer(suffix_a, 0)
    partial = evidence[:3]
    handoff.prime(SCOPES["target"], partial)
    partial_result = answer(suffix_a, len(partial))
    handoff.prime(SCOPES["target"], evidence)
    first_a = answer(suffix_a, len(evidence))
    branch_b = answer(suffix_b, len(evidence))
    last_a = answer(suffix_a_again, len(evidence))
    if first_a != last_a:
        raise ValueError("Independent suffix branch changed an earlier result")
    return {"evidence_kind": "tiny_random_Qwen3_cpu", "worker_answer_token": int(worker.logits[0, -1].argmax()),
            "source_hit": registry.longest_native(SCOPES["source"], evidence).token_count == len(evidence),
            "paths": {"native_cold": cold, "native_partial": partial_result, "native_full": first_a},
            "interleaved_branch_b": branch_b, "interleaved_a_unchanged": True}


def screen_archived_candidate(files, thresholds):
    def work(_):
        for path, expected in files.items():
            if sha256(Path(path)) != expected:
                raise ValueError("Frozen evidence changed: " + path)
        rows = json.loads((EVIDENCE / "rows.json").read_text())
        timings = json.loads((EVIDENCE / "timings.json").read_text())
        gold = {row["id"]: (row["document_id"], row["answer"])
                for row in map(json.loads, QUALITY_ROWS.read_text().splitlines())}
        manifest = json.loads(QUALITY_MANIFEST.read_text())["files"]["quality.jsonl"]
        docs = {row["document_id"] for row in rows}
        if (sha256(QUALITY_ROWS) != manifest["sha256"] or len(rows) != manifest["examples"]
                or len(docs) != manifest["groups"] or len(timings) != len(docs)
                or {row["document_id"] for row in timings} != docs
                or {row["id"] for row in rows} != set(gold)
                or any((row["document_id"], row["gold"]) != gold[row["id"]] for row in rows)):
            raise ValueError("Rows do not match the frozen test population")
        counts = Counter()
        for row in rows:
            scores = {"native": row["native"],
                      **{name: item["score"] for name, item in row["variants"].items()}}
            for name, result in scores.items():
                counts[name] += result["decision_answer"] == gold[row["id"]][1]
        fractions = {name: count/len(rows) for name, count in counts.items()}
        cold_ms = median(row["native_cold_ms"] for row in timings)
        warm_source_ms = median(row["conversion_ms"] + row["translated_query_ms"] for row in timings)
        gates = {"quality_point_screen": fractions["translated"] >= fractions["native"] - thresholds["quality_margin"],
                 "beats_source": fractions["translated"] > fractions["source"],
                 "ttft_point_screen": warm_source_ms <= thresholds["ttft_ratio"] * cold_ms}
        return {"source": "archived_sagemaker_quality_8b", "questions": len(rows), "documents": len(docs),
                "counts": dict(counts), "accuracy": fractions, "native_cold_ms": cold_ms,
                "source_cache_hit_transfer_ms": warm_source_ms, "gates": gates,
                "qualified_for_confirmation": all(gates.values())}
    return work


def final_report(results):
    samples = results["native_replay"]
    registry = results["registry_checks"]
    if (not all(registry["checks"].values()) or not samples["interleaved_a_unchanged"]
            or not samples["source_hit"]):
        raise ValueError("Local smoke requirements failed")
    candidate = results["archived_screen"]
    return {"local_milestone": "passed", "fixture_evidence": samples["evidence_kind"],
            "product_goal_achieved": False, "new_gpu_jobs": 0,
            "archived_candidate": ("requires_independent_confirmation"
                                   if candidate["qualified_for_confirmation"] else "rejected_on_development"),
            "next_unmet_dependency": ("independent confirmation of the frozen candidate"
                                      if candidate["qualified_for_confirmation"]
                                      else "new trained candidate and measured development pass")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {str(path): sha256(path) for path in (EVIDENCE / "rows.json", EVIDENCE / "timings.json",
                                                  QUALITY_ROWS, QUALITY_MANIFEST)}
    modules = [Path(__file__), Path(__file__).with_name("agent_graph.py"),
               Path(__file__).with_name("cache_handoff.py"), Path(__file__).with_name("hf_backend.py"),
               Path(__file__).with_name("data.py"), Path(__file__).with_name("handoff.py")]
    thresholds = {"quality_margin": .03, "ttft_ratio": .8}
    spec = {"revision": 1, "scope": "offline_first_milestone", "evidence": files,
            "code": {path.name: sha256(path) for path in modules},
            "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
            "thresholds": thresholds}
    nodes = [
        Node("cache_contract", (), cache_contract),
        Node("trace", (), trace),
        Node("registry_checks", ("cache_contract",), registry_checks),
        Node("native_replay", ("cache_contract", "trace"), replay_native),
        Node("archived_screen", (), screen_archived_candidate(files, thresholds)),
        Node("verification", ("registry_checks", "native_replay", "archived_screen"), final_report),
    ]
    try:
        results = asyncio.run(Controller(args.output / "graph.sqlite3", nodes, spec).run())
    except GraphFailed as exc:
        print(f"Graph stopped: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", dir=args.output, prefix=".summary-", delete=False) as temp:
        tmp_path = Path(temp.name)
        try:
            temp.write(payload)
            temp.close()
            tmp_path.replace(args.output / "summary.json")
        finally:
            tmp_path.unlink(missing_ok=True)
    print(json.dumps({"report": str((args.output / "summary.json").resolve()),
                      **results["verification"], "candidate_gates": results["archived_screen"]["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
