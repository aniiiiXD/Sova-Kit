"""Evaluate checks, compare tool proposals, and classify case outcomes."""

from __future__ import annotations

import json

from switchit._run_utils import clip

VERDICTS = ("pass", "fail", "error", "uncertain")


CHECK_TYPES = (
    "no_error", "tool_called", "tool_args_include", "tool_calls_equal",
    "output_contains", "output_equals", "output_regex", "output_min_chars",
    "latency_ms_max", "usage_input_max", "usage_output_max",
)


def _tool_call_names(result: dict) -> list:
    return [call.get("name", "") for call in result.get("tool_calls") or []]


def _deep_subset(superset, subset) -> bool:
    """True when every key/value in subset appears in superset (recursively)."""
    if isinstance(subset, dict):
        if not isinstance(superset, dict):
            return False
        for key, value in subset.items():
            if key not in superset or not _deep_subset(superset[key], value):
                return False
        return True
    if isinstance(subset, list):
        if not isinstance(superset, list) or len(superset) < len(subset):
            return False
        remaining = list(superset)
        for item in subset:
            for index, candidate in enumerate(remaining):
                if _deep_subset(candidate, item):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    return superset == subset


def evaluate_check(check: dict, result: dict) -> dict:
    """Evaluate one check against one replay result.

    Statuses: pass, fail, unknown. A check whose inputs are missing (for
    example usage when the provider reported none) evaluates to unknown
    rather than silently passing.
    """
    check_type = check.get("type")
    evaluated = {
        "type": check_type,
        "required": bool(check.get("required", True)),
        "source": check.get("source", "user"),
        "status": "unknown",
        "detail": "",
    }
    error = result.get("error")
    text = result.get("text") or ""
    tool_calls = result.get("tool_calls") or []
    usage = result.get("usage") or {}

    if error:
        if check_type == "no_error":
            evaluated.update(status="fail", detail=clip(error, 200))
        else:
            evaluated.update(status="unknown", detail="target call errored; check not evaluable")
        return evaluated

    if check_type == "no_error":
        evaluated.update(status="pass", detail="completed without a provider error")

    elif check_type == "tool_called":
        name = check.get("name", "")
        if name in _tool_call_names(result):
            evaluated.update(status="pass", detail=f"proposed tool '{name}'")
        else:
            evaluated.update(
                status="fail",
                detail=f"tool '{name}' was not proposed; proposed: {_tool_call_names(result) or 'none'}",
            )

    elif check_type == "tool_args_include":
        name = check.get("name", "")
        subset = check.get("arguments") or {}
        for call in tool_calls:
            if call.get("name") == name and _deep_subset(call.get("arguments") or {}, subset):
                evaluated.update(status="pass", detail=f"arguments of '{name}' include the required subset")
                break
        else:
            evaluated.update(
                status="fail",
                detail=f"no proposed '{name}' call includes arguments subset {json.dumps(subset, sort_keys=True)}",
            )

    elif check_type == "tool_calls_equal":
        expected = list(check.get("names") or [])
        actual = _tool_call_names(result)
        evaluated.update(
            status="pass" if actual == expected else "fail",
            detail=f"expected {expected} in order; got {actual}",
        )

    elif check_type == "output_equals":
        expected = check.get("text")
        if not isinstance(expected, str):
            evaluated.update(status="unknown", detail="output_equals requires a text string")
        else:
            evaluated.update(status="pass" if text == expected else "fail", detail="exact output text comparison")

    elif check_type == "output_contains":
        needle = check.get("text", "")
        if needle.casefold() in text.casefold():
            evaluated.update(status="pass", detail=f"output contains {needle[:60]!r}")
        else:
            evaluated.update(status="fail", detail=f"output does not contain {clip(needle, 60)!r}")

    elif check_type == "output_regex":
        import re

        try:
            matched = re.search(check.get("pattern", ""), text)
        except re.error as regex_error:
            evaluated.update(status="unknown", detail=f"invalid regex: {regex_error}")
            return evaluated
        if matched:
            evaluated.update(status="pass", detail=f"regex matched {matched.group(0)[:60]!r}")
        else:
            evaluated.update(status="fail", detail="regex did not match the output")

    elif check_type == "output_min_chars":
        minimum = int(check.get("value", 0))
        evaluated.update(
            status="pass" if len(text) >= minimum else "fail",
            detail=f"output length {len(text)} vs minimum {minimum}",
        )

    elif check_type == "latency_ms_max":
        latency = result.get("latency_ms")
        if latency is None:
            evaluated.update(status="unknown", detail="no latency recorded")
        else:
            maximum = int(check.get("value", 0))
            evaluated.update(
                status="pass" if latency <= maximum else "fail",
                detail=f"latency {latency} ms vs maximum {maximum} ms",
            )

    elif check_type in ("usage_input_max", "usage_output_max"):
        key = "input" if check_type == "usage_input_max" else "output"
        value = usage.get(f"{key}_tokens")
        if value is None or usage.get("basis") == "unknown":
            evaluated.update(
                status="unknown",
                detail=f"target usage for '{key}' is unknown (provenance: {usage.get('basis', 'unknown')})",
            )
        else:
            maximum = int(check.get("value", 0))
            evaluated.update(
                status="pass" if value <= maximum else "fail",
                detail=f"{key} tokens {value} vs maximum {maximum} (provenance: {usage.get('basis')})",
            )

    else:
        evaluated.update(status="unknown", detail=f"unknown check type {check_type!r}")
    return evaluated


def evaluate_checks(checks: list, result: dict) -> list:
    return [evaluate_check(check, result) for check in checks or []]


def historical_result(case: dict) -> dict:
    """The case's historical response shaped like a replay result."""
    historical = case.get("historical") or {}
    usage = dict(historical.get("usage") or {})
    usage.setdefault("basis", "unknown")
    usage.setdefault("estimator", None)
    return {
        "text": historical.get("text") or "",
        "tool_calls": historical.get("tool_calls") or [],
        "finish_reason": historical.get("finish_reason"),
        "error": historical.get("error"),
        "latency_ms": historical.get("latency_ms"),
        "usage": usage,
        "adapter": "historical",
        "model": historical.get("model"),
    }


def compare_tools(baseline_tools: list, target_tools: list) -> list:
    """Compare every occurrence in order, including duplicate tool names."""
    findings = []
    baseline, target = baseline_tools or [], target_tools or []
    for position in range(max(len(baseline), len(target))):
        left = baseline[position] if position < len(baseline) else None
        right = target[position] if position < len(target) else None
        if left and right and left.get("name") == right.get("name"):
            same = left.get("arguments", {}) == right.get("arguments", {})
            findings.append({"kind": "match" if same else "arg_drift", "tool": left.get("name", ""),
                             "detail": f"occurrence {position}: " + ("same tool, equivalent arguments" if same else "arguments differ")})
        else:
            if left:
                findings.append({"kind": "missing_tool", "tool": left.get("name", ""),
                                 "detail": f"reference occurrence {position} missing or reordered"})
            if right:
                findings.append({"kind": "added_tool", "tool": right.get("name", ""),
                                 "detail": f"target occurrence {position} added or reordered"})
    return findings or [{"kind": "match", "tool": "", "detail": "no proposed tool calls on either side"}]


def finalize_case(case: dict, baseline_result: dict, target_result: dict) -> dict:
    """Evaluate checks, compare tools, and classify the case verdict."""
    checks_target = evaluate_checks(case.get("checks"), target_result)
    checks_reference = evaluate_checks(case.get("checks"), historical_result(case))
    tool_findings = compare_tools(
        historical_result(case).get("tool_calls"),
        target_result.get("tool_calls"),
    )

    uncertainty = []
    failures = []
    for check, reference in zip(checks_target, checks_reference):
        reference_failed = bool(reference and reference["status"] == "fail")
        if check["status"] == "fail":
            failures.append(
                {
                    "kind": "required_check_failed" if check["required"] else "optional_check_failed",
                    "check": check,
                    "case_id": case["case_id"],
                }
            )
            if reference_failed:
                uncertainty.append(
                    f"check '{check['type']}' also fails against the historical reference; "
                    "the check may be mis-specified rather than the target broken"
                )
        elif check["status"] == "unknown" and check["required"]:
            uncertainty.append(f"required check '{check['type']}' could not be evaluated: {check['detail']}")

    target_error = target_result.get("error")
    if target_error:
        failures.append({"kind": "target_error", "check": None, "case_id": case["case_id"]})

    explicit_tool_checks = {
        check.get("name") for check in case.get("checks") or [] if check.get("type") == "tool_called"
    }
    for finding in tool_findings:
        if finding["kind"] == "missing_tool":
            uncertainty.append(
                f"reference proposed tool '{finding['tool']}' but the target did not, and no explicit check covers it"
            )
        elif finding["kind"] == "added_tool":
            uncertainty.append(f"target proposed an additional tool '{finding['tool']}' absent from the reference")
        elif finding["kind"] == "arg_drift":
            uncertainty.append(f"arguments drifted for tool '{finding['tool']}' while no strict check covers them")

    historical_basis = (case.get("historical", {}).get("usage") or {}).get("basis")
    target_basis = (target_result.get("usage") or {}).get("basis")
    notes = []
    if historical_basis != target_basis:
        notes.append(
            f"usage provenance differs: historical '{historical_basis}' vs target '{target_basis}'; "
            "token totals are reported separately by basis and are not directly comparable"
        )

    if target_error:
        verdict = "error"
    elif any(f["kind"] == "required_check_failed" for f in failures):
        verdict = "fail"
    elif uncertainty:
        verdict = "uncertain"
    else:
        verdict = "pass"

    return {
        "case_id": case["case_id"],
        "workflow_id": case.get("workflow_id"),
        "trace_id": case.get("trace_id"),
        "origins": case.get("origins") or [],
        "baseline": baseline_result,
        "target": target_result,
        "checks_target": checks_target,
        "checks_reference": checks_reference,
        "tool_findings": tool_findings,
        "failures": failures,
        "uncertainty": uncertainty,
        "notes": notes,
        "verdict": verdict,
    }
