"""Evidence contracts and conservative paired summaries. No model imports."""
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ("source", "native_cold", "native_warm", "transfer_cold", "transfer_warm")


def validate_trace(row):
    if row.get("arm") not in ARMS or row.get("evidence_kind") not in ("synthetic_fixture", "hf_model"):
        raise ValueError("Unknown arm/evidence kind")
    for key in ("id", "document_id", "context_hash", "prompt_hash", "run_config_hash"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Missing trace field: {key}")
    if row.get("status") not in ("ok", "error", "over_limit"):
        raise ValueError("Invalid status")
    if not isinstance(row.get("repeat"), int) or row["repeat"] < 0:
        raise ValueError("Invalid repeat")
    if row["status"] != "ok":
        return
    for key in ("ttft_ms", "total_ms"):
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"Invalid timing: {key}")
    if row["total_ms"] + 1e-5 < row["ttft_ms"]:
        raise ValueError("Total duration precedes first token")
    if not isinstance(row.get("correct"), bool):
        raise ValueError("Correctness must be an explicit boolean")
    if row["arm"].startswith("transfer"):
        required = ("source_prefill_ms", "map_ms", "import_ms", "target_first_ms")
        if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] >= 0 for k in required):
            raise ValueError("Transfer lacks finite component timings")
        if row.get("fallback") is not False or row.get("imported_tokens", 0) <= 0:
            raise ValueError("A transfer success requires verified import and no fallback")
        if row.get("target_prefill_tokens") != row.get("prompt_tokens", 0) - row["imported_tokens"]:
            raise ValueError("Imported/recomputed token accounting mismatch")
        if row["target_prefill_tokens"] != 1:
            raise ValueError("This runtime permits exactly one target tail token")
        components = sum(row[k] for k in required if row["arm"] == "transfer_cold" or k != "source_prefill_ms")
        if row["ttft_ms"] + .01 < components:
            raise ValueError("TTFT excludes part of the transfer path")


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def paired_quality(rows, iterations=1000, seed=19):
    groups = defaultdict(list)
    for row in rows:
        if row["repeat"] == 0 and row["arm"] in ("native_cold", "transfer_cold"):
            groups[row["id"]].append(row)
    documents = defaultdict(list)
    for example_id, pair in groups.items():
        byarm = {r["arm"]: r for r in pair}
        if len(pair) != 2 or len(byarm) != 2:
            raise ValueError(f"Unpaired quality example: {example_id}")
        a, b = byarm["native_cold"], byarm["transfer_cold"]
        if any(a[k] != b[k] for k in ("document_id", "context_hash", "prompt_hash", "run_config_hash")):
            raise ValueError("Quality arms have mismatched inputs/configuration")
        # Errors and over-limit outcomes are retained as incorrect.
        delta = int(b.get("correct", False)) - int(a.get("correct", False))
        documents[a["document_id"]].append(delta)
    if not documents:
        return {"examples": 0, "documents": 0, "difference": None, "ci95": None}
    docs = sorted(documents)
    observed = sum(sum(v) for v in documents.values()) / sum(len(v) for v in documents.values())
    ci = None
    if len(docs) >= 2:
        rng = random.Random(seed)
        bootstrap = []
        for _ in range(iterations):
            sampled = [rng.choice(docs) for _ in docs]
            bootstrap.append(sum(sum(documents[d]) for d in sampled) / sum(len(documents[d]) for d in sampled))
        ci = [percentile(bootstrap, .025), percentile(bootstrap, .975)]
    return {"examples": sum(len(v) for v in documents.values()), "documents": len(docs), "difference": observed, "ci95": ci}


def summarize(rows):
    seen = set()
    for row in rows:
        validate_trace(row)
        key = (row["id"], row["repeat"], row["arm"])
        if key in seen:
            raise ValueError("Duplicate trace pairing")
        seen.add(key)
    kinds = {r["evidence_kind"] for r in rows}
    if len(kinds) != 1:
        raise ValueError("Do not mix synthetic and model evidence")
    if len({r["run_config_hash"] for r in rows}) != 1:
        raise ValueError("Do not combine different run configurations")
    arms = {}
    for arm in ARMS:
        allrows = [r for r in rows if r["arm"] == arm]
        good = [r for r in allrows if r["status"] == "ok"]
        arms[arm] = {"attempts": len(allrows), "returned": len(good), "errors_or_over_limit": len(allrows)-len(good),
                     "quality_examples": sum(r["repeat"] == 0 for r in allrows),
                     "accuracy": (sum(r.get("correct", False) for r in allrows if r["repeat"] == 0) / sum(r["repeat"] == 0 for r in allrows)) if any(r["repeat"] == 0 for r in allrows) else None,
                     "p50_ttft_ms": statistics.median([r["ttft_ms"] for r in good]) if good else None,
                     "p95_ttft_ms": percentile([r["ttft_ms"] for r in good], .95)}
    pairs = defaultdict(dict)
    for row in rows:
        if row["arm"] in ("native_cold", "transfer_cold"):
            pairs[(row["id"], row["repeat"])][row["arm"]] = row
    ratios = []
    for key, pair in pairs.items():
        if set(pair) != {"native_cold", "transfer_cold"}:
            raise ValueError(f"Missing latency arm: {key}")
        a, b = pair["native_cold"], pair["transfer_cold"]
        if any(a[k] != b[k] for k in ("prompt_hash", "context_hash", "run_config_hash")):
            raise ValueError("Latency arms have mismatched inputs")
        if a["status"] == b["status"] == "ok" and b["ttft_ms"] > 0:
            ratios.append(a["ttft_ms"]/b["ttft_ms"])
    return {"status": "smoke_only_no_thesis_claim" if kinds == {"synthetic_fixture"} else "exploratory_model_results_not_acceptance_decision",
            "evidence_kind": next(iter(kinds)), "arms": arms, "paired_quality": paired_quality(rows),
            "successful_latency_pairs": len(ratios), "median_paired_cold_speedup": statistics.median(ratios) if ratios else None,
            "acceptance": "not_established", "cost_savings": "not_measured",
            "limitations": ["Latency summaries condition on successful responses; all attempt/error counts remain visible.",
                            "Quality uses one answer per example (repeat zero), not repeated copies as independent observations.",
                            "Confidence intervals are exploratory; no production noninferiority claim is automated."]}


def write_report(directory, rows, metadata, allow_existing=False):
    root = Path(directory)
    if (root/"summary.json").exists():
        raise ValueError("Refusing to overwrite an existing report")
    root.mkdir(parents=True, exist_ok=allow_existing)
    (root/"traces.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    summary = summarize(rows)
    (root/"summary.json").write_text(json.dumps({**summary, "metadata": metadata}, indent=2, allow_nan=False)+"\n")
    lines = ["# SOVA cache-transfer verification", "", f"Status: **{summary['status']}**", "",
             "Synthetic fixtures verify software plumbing only. Model results remain exploratory until the protocol's controls, sample sizes and hardware comparisons pass.", "",
             "| Arm | Attempts | Returned | Errors/over limit | Accuracy (repeat 0) | Median TTFT ms | P95 TTFT ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, stats in summary["arms"].items():
        fmt = lambda v: "-" if v is None else f"{v:.4f}"
        lines.append(f"| {arm} | {stats['attempts']} | {stats['returned']} | {stats['errors_or_over_limit']} | {fmt(stats['accuracy'])} | {fmt(stats['p50_ttft_ms'])} | {fmt(stats['p95_ttft_ms'])} |")
    lines += ["", "Acceptance: not established. Cost savings: not measured.", "", "Quality uncertainty uses paired, document-cluster resampling. Errors count as wrong; repeated timing attempts are not extra quality examples."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    return summary
