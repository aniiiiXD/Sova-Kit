"""Aggregate evaluated runs into usage, failure groups, and prioritized fixes."""

from __future__ import annotations

from switchit.evaluation import VERDICTS


def _merge_usage_results(results: list) -> dict:
    merged = {
        basis: {"calls": 0, "input_tokens_known": 0, "output_tokens_known": 0, "input_missing_calls": 0,
                "output_missing_calls": 0, "calls_missing_either": 0}
        for basis in ("recorded", "estimated", "unknown")
    }
    without_usage = 0
    for result in results:
        if not result:
            continue
        usage = result.get("usage") or {}
        basis = usage.get("basis") or "unknown"
        if basis not in merged:
            basis = "unknown"
        bucket = merged[basis]
        bucket["calls"] += 1
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is None and output_tokens is None:
            without_usage += 1
        for key, value in (("input", input_tokens), ("output", output_tokens)):
            if value is None:
                bucket[f"{key}_missing_calls"] += 1
            else:
                bucket[f"{key}_tokens_known"] += int(value)
        if input_tokens is None or output_tokens is None:
            bucket["calls_missing_either"] += 1
    merged["requests_without_usage"] = without_usage
    return merged


def aggregate(results_doc: dict, cases_doc: dict, capabilities_baseline: dict,
              capabilities_target: dict) -> dict:
    """Summarize a run: verdict counts, failure groups, gaps, fixes, usage."""
    case_results = results_doc.get("case_results") or []
    cases_by_id = {case["case_id"]: case for case in cases_doc.get("cases") or []}

    verdict_counts = {verdict: 0 for verdict in VERDICTS}
    for result in case_results:
        verdict_counts[result["verdict"]] = verdict_counts.get(result["verdict"], 0) + 1

    groups = {}
    for result in case_results:
        for failure in result.get("failures") or []:
            kind = failure["kind"]
            if kind == "required_check_failed" and failure.get("check"):
                key = f"required_check_failed:{failure['check']['type']}"
            elif kind == "optional_check_failed" and failure.get("check"):
                key = f"optional_check_failed:{failure['check']['type']}"
            elif kind == "target_error":
                key = "target_error"
            else:
                key = kind
            group = groups.setdefault(key, {"count": 0, "case_ids": [], "kinds": set()})
            group["count"] += 1
            if failure["case_id"] not in group["case_ids"]:
                group["case_ids"].append(failure["case_id"])
            group["kinds"].add(kind)

    for result in case_results:
        for finding in result.get("tool_findings") or []:
            if finding["kind"] in ("missing_tool", "added_tool", "arg_drift"):
                key = f"tool_{finding['kind']}"
                group = groups.setdefault(key, {"count": 0, "case_ids": [], "kinds": set()})
                group["count"] += 1
                if result["case_id"] not in group["case_ids"]:
                    group["case_ids"].append(result["case_id"])
                group["kinds"].add(f"tool_{finding['kind']}")

    failure_groups = [
        {
            "key": key,
            "count": group["count"],
            "case_ids": group["case_ids"][:12],
            "kinds": sorted(group["kinds"]),
        }
        for key, group in sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    ]

    needed = {"tool_calling": False}
    for case in cases_doc.get("cases") or []:
        historical_tools = (case.get("historical") or {}).get("tool_calls") or []
        request_tools = (case.get("request") or {}).get("tools") or []
        if historical_tools or request_tools:
            needed["tool_calling"] = True
    capability_gaps = []
    if needed["tool_calling"] and capabilities_target.get("tool_calling") is False:
        capability_gaps.append(
            {
                "capability": "tool_calling",
                "detail": "cases replay tool-using workflows, but the target adapter declares no tool-calling support",
            }
        )

    fixes = []
    if verdict_counts["error"]:
        fixes.append({
            "priority": 1,
            "title": f"{verdict_counts['error']} case(s) errored on the target provider",
            "case_ids": [r["case_id"] for r in case_results if r["verdict"] == "error"][:12],
            "suggestion": "Inspect the recorded error strings and the adapter configuration (model id, endpoint, credentials, region) before anything else.",
        })
    if capability_gaps:
        fixes.append({
            "priority": 2,
            "title": "Declared target capabilities do not cover what the cases need",
            "case_ids": sorted({cid for gap in capability_gaps for cid in _ids_for_capability(case_results, cases_by_id)}),
            "suggestion": "Check the target provider's current documentation and either enable the capability or pick workflows that do not need it.",
        })
    required_failures = [g for g in failure_groups if g["key"].startswith("required_check_failed")]
    if required_failures:
        fixes.append({
            "priority": 3,
            "title": "Required success checks failed on the target",
            "case_ids": sorted({cid for group in required_failures for cid in group["case_ids"]}),
            "suggestion": "For each failing check, decide whether the prompt, the tool schema mapping, or the check itself needs to change; then replay.",
        })
    tool_groups = [g for g in failure_groups if g["key"].startswith("tool_missing_tool")]
    if tool_groups:
        fixes.append({
            "priority": 4,
            "title": "Target stopped proposing tools the reference proposed",
            "case_ids": sorted({cid for group in tool_groups for cid in group["case_ids"]}),
            "suggestion": "Map tool schemas and system prompts explicitly for the target; missing tool proposals usually trace to prompt or schema translation.",
        })
    mis_specified_cases = sorted({
        result["case_id"]
        for result in case_results
        for note in result.get("uncertainty") or []
        if "mis-specified" in note
    })
    if mis_specified_cases:
        fixes.append({
            "priority": 5,
            "title": "Some checks fail even against the historical reference",
            "case_ids": mis_specified_cases,
            "suggestion": "Fix or relax these checks in cases.json; they cannot distinguish a broken target from a broken expectation.",
        })
    latency_groups = [g for g in failure_groups if g["key"] == "optional_check_failed:latency_ms_max"]
    if latency_groups:
        fixes.append({
            "priority": 6,
            "title": "Latency exceeded the default budget on some cases",
            "case_ids": sorted({cid for group in latency_groups for cid in group["case_ids"]}),
            "suggestion": "Compare target latency against historical latency by workflow; tune the budget or investigate the provider.",
        })
    drift_groups = [g for g in failure_groups if g["key"] == "tool_arg_drift"]
    if drift_groups:
        fixes.append({
            "priority": 7,
            "title": "Tool arguments drifted from the reference",
            "case_ids": sorted({cid for group in drift_groups for cid in group["case_ids"]}),
            "suggestion": "Add explicit tool_args_include checks for the arguments that matter, then replay.",
        })

    usage = {
        "baseline": _merge_usage_results([r.get("baseline") for r in case_results]),
        "target": _merge_usage_results([r.get("target") for r in case_results]),
    }

    return {
        "verdicts": verdict_counts,
        "failure_groups": failure_groups,
        "capability_gaps": capability_gaps,
        "needed_capabilities": needed,
        "prioritized_fixes": fixes,
        "usage": usage,
        "cases_with_uncertainty": sorted({
            result["case_id"] for result in case_results if result.get("uncertainty")
        }),
    }


def _ids_for_capability(case_results, cases_by_id):
    for result in case_results:
        case = cases_by_id.get(result["case_id"]) or {}
        historical_tools = (case.get("historical") or {}).get("tool_calls") or []
        request_tools = (case.get("request") or {}).get("tools") or []
        if historical_tools or request_tools:
            yield result["case_id"]
