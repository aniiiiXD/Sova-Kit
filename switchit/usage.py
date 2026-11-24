"""Usage normalization and accounting, keeping recorded/estimated/unknown separate."""

from __future__ import annotations


def empty_usage() -> dict:
    """Per-basis usage counters. Every call is counted exactly once."""
    return {
        basis: {
            "calls": 0,
            "input_tokens_known": 0,
            "output_tokens_known": 0,
            "input_missing_calls": 0,
            "output_missing_calls": 0,
            "calls_missing_either": 0,
        }
        for basis in ("recorded", "estimated", "unknown")
    }


def normalize_basis(basis, estimator=None):
    basis = (basis or "unknown").strip().lower()
    if basis.startswith("source_reported") or basis in ("recorded", "source"):
        return "recorded", estimator
    if basis.startswith("estimated"):
        return "estimated", estimator or basis
    return "unknown", estimator


def add_usage(usage: dict, input_tokens, output_tokens, basis: str, estimator=None) -> dict:
    """Add one call's usage to a per-basis aggregate. Missing values stay visible."""
    basis, estimator = normalize_basis(basis, estimator)
    bucket = usage.setdefault(
        basis,
        {
            "calls": 0,
            "input_tokens_known": 0,
            "output_tokens_known": 0,
            "input_missing_calls": 0,
            "output_missing_calls": 0,
            "calls_missing_either": 0,
        },
    )
    bucket["calls"] += 1
    for key, value in (("input", input_tokens), ("output", output_tokens)):
        if value is None:
            bucket[f"{key}_missing_calls"] += 1
        else:
            try:
                bucket[f"{key}_tokens_known"] += int(value)
            except (TypeError, ValueError):
                bucket[f"{key}_missing_calls"] += 1
    if input_tokens is None or output_tokens is None:
        bucket["calls_missing_either"] += 1
    return usage


def merge_usage(*usages: dict) -> dict:
    merged = empty_usage()
    for usage in usages:
        if not usage:
            continue
        for basis, bucket in usage.items():
            target = merged.setdefault(basis, empty_usage()[basis])
            for key, value in bucket.items():
                target[key] = target.get(key, 0) + value
    return merged


def usage_from_record(raw: dict):
    """Extract (input_tokens, output_tokens, basis, estimator) from a raw record."""
    usage = raw.get("usage")
    input_tokens = raw.get("input_tokens")
    output_tokens = raw.get("output_tokens")
    basis = raw.get("usage_basis")
    estimator = raw.get("usage_estimator")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", input_tokens)
        output_tokens = usage.get("output_tokens", output_tokens)
        basis = usage.get("basis", basis)
        estimator = usage.get("estimator", estimator)
    elif usage is not None:
        basis = basis or "recorded"
    for key in ("estimated_input_tokens", "est_input_tokens"):
        if input_tokens is None and raw.get(key) is not None:
            input_tokens = raw[key]
            basis = basis or "estimated"
    for key in ("estimated_output_tokens", "est_output_tokens"):
        if output_tokens is None and raw.get(key) is not None:
            output_tokens = raw[key]
            basis = basis or "estimated"
    if basis is None and (input_tokens is not None or output_tokens is not None):
        basis = "recorded"
    if input_tokens is None and output_tokens is None and not basis:
        basis = "unknown"
    return input_tokens, output_tokens, basis, estimator
