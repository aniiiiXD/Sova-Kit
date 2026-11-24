"""Select representative replay cases, historical fixtures, and default checks."""

from __future__ import annotations

import json

from switchit.storage import atomic_write_json

CASES_VERSION = 1


ORIGIN_PRIORITY = ("failed", "risky", "costly", "common")


def _case_usage_value(usage: dict, key: str):
    """Prefer recorded usage for budget math; fall back to estimated; else None."""
    for basis in ("recorded", "estimated"):
        bucket = (usage or {}).get(basis) or {}
        if bucket.get(f"{key}_tokens_known"):
            return bucket[f"{key}_tokens_known"]
    return None


def _seed_call(store: dict, trace: dict):
    """Pick the call whose response best represents the trace outcome.

    Preference: last call with a non-error response, then last call with any
    response, then simply the last call (error-only traces replay their error).
    """
    calls_by_id = {call["call_id"]: call for call in store["calls"]}
    members = [calls_by_id[cid] for cid in trace["call_ids"] if cid in calls_by_id]
    if not members:
        return store["calls"][0] if store["calls"] else None
    with_response = [call for call in members if call["response"]["text"] or call["response"]["tool_calls"]]
    without_error = [call for call in with_response if not call["response"]["error"]]
    if without_error:
        return without_error[-1]
    if with_response:
        return with_response[-1]
    return members[-1]


def default_checks(case: dict) -> list:
    """Deterministic default checks derived from reference behavior.

    Defaults are deliberately conservative: tool presence, no error, and loose
    latency/usage budgets. They are starting points for a reviewer to edit —
    a historical model response alone is never treated as proof of success.
    """
    checks = [{"type": "no_error", "required": True, "source": "default"}]
    historical_tools = [call["name"] for call in case["historical"]["tool_calls"] if call["name"]]
    for name in sorted(set(historical_tools)):
        checks.append({"type": "tool_called", "name": name, "required": True, "source": "default"})
    historical_text = case["historical"]["text"]
    if historical_text and len(historical_text) > 40:
        checks.append(
            {"type": "output_min_chars", "value": max(20, int(len(historical_text) * 0.25)),
             "required": False, "source": "default"}
        )
    latency = case["historical"].get("latency_ms")
    if latency:
        checks.append(
            {"type": "latency_ms_max", "value": int(max(5000, latency * 3)),
             "required": False, "source": "default"}
        )
    output_tokens = _case_usage_value(case["historical"].get("usage") or {}, "output")
    if output_tokens:
        checks.append(
            {"type": "usage_output_max", "value": int(max(256, output_tokens * 3)),
             "required": False, "source": "default"}
        )
    return checks


def select_cases(store: dict, workflow_map: dict, max_cases: int = 24) -> dict:
    """Select representative replay cases across workflows.

    Coverage goals from the plan: common, costly, risky, and failed workflows,
    plus at least one case from traces outside any confident workflow when they
    exist. One case per trace maximum; per-workflow cap keeps the set small.
    """
    traces_by_id = {trace["trace_id"]: trace for trace in store["traces"]}
    cases = []
    used_traces = set()
    per_workflow_cap = 2

    def base_usage(trace):
        return trace["usage"]

    def make_case(trace, workflow_id, origins):
        seed = _seed_call(store, trace)
        if seed is None:
            return None
        # Tool behavior belongs to the whole trace, not just the final answer:
        # merge every proposed tool call across the trace's calls, preserving
        # every occurrence in order.
        calls_by_id = {call["call_id"]: call for call in store["calls"]}
        merged_tools = []
        for call_id in trace["call_ids"]:
            call = calls_by_id.get(call_id)
            if not call:
                continue
            for tool_call in call["response"]["tool_calls"]:
                name = tool_call.get("name", "")
                if name:
                    merged_tools.append({"name": name, "arguments": tool_call.get("arguments") or {}})
        # Recorded tool fixtures: what the business tool actually returned in
        # history (paired proposal -> following call's tool-result messages).
        # These let a chained replay complete the workflow offline without
        # ever executing a real business tool.
        tool_fixtures = {}
        members = [calls_by_id[cid] for cid in trace["call_ids"] if cid in calls_by_id]
        for position, call in enumerate(members[:-1]):
            proposed = [tc.get("name", "") for tc in call["response"]["tool_calls"] if tc.get("name")]
            if not proposed:
                continue
            followup = members[position + 1]
            results = [
                message.get("content") for message in followup["request"]["messages"]
                if message.get("role") == "tool" and message.get("content")
            ]
            for name, result in zip(proposed, results):
                if name and result and name not in tool_fixtures:
                    tool_fixtures[name] = result
        historical = {
            "text": seed["response"]["text"],
            "tool_calls": merged_tools,
            "finish_reason": seed["response"]["finish_reason"],
            "error": seed["response"]["error"],
            "usage": dict(seed["usage"]),
            "latency_ms": seed["latency_ms"],
            "model": seed["model"],
            "call_id": seed["call_id"],
            "usage_note": (
                "Reference behavior captured from history. It is what the current "
                "provider did, not proof of what success looks like."
            ),
        }
        case = {
            "case_id": f"case_{len(cases) + 1:03d}",
            "workflow_id": workflow_id,
            "trace_id": trace["trace_id"],
            "origins": origins,
            "request": seed["request"],
            "historical": historical,
        }
        if tool_fixtures:
            case["tool_fixtures"] = tool_fixtures
        case["checks"] = default_checks(case)
        case["notes"] = [
            "The historical response is reference behavior, not a success guarantee.",
            "Tool calls proposed by the target are captured and compared, never executed.",
        ]
        if tool_fixtures:
            case["notes"].append(
                "Recorded tool fixtures are attached (tool_fixtures). A chained replay "
                "(replay --chain) can use them to complete the workflow offline — the "
                "harness never executes a real business tool."
            )
        return case

    def add_candidate(trace, workflow_id, origin):
        if trace["trace_id"] in used_traces:
            for existing in cases:
                if existing["trace_id"] == trace["trace_id"] and origin not in existing["origins"]:
                    existing["origins"].append(origin)
                    existing["origins"].sort(key=ORIGIN_PRIORITY.index)
                    return
            return
        case = make_case(trace, workflow_id, [origin])
        if case is not None:
            cases.append(case)
            used_traces.add(trace["trace_id"])

    for workflow in workflow_map["workflows"]:
        members = [traces_by_id[tid] for tid in workflow["trace_ids"] if tid in traces_by_id]
        if not members:
            continue
        picks = []
        failed = [trace for trace in members if trace["error_count"]]
        risky = sorted(members, key=lambda trace: (-len(trace["tool_names"]), trace["trace_id"]))
        costly = sorted(
            members,
            key=lambda trace: (-(_case_usage_value(base_usage(trace), "input") or 0), trace["trace_id"]),
        )
        common = sorted(members, key=lambda trace: (-trace["call_count"], trace["trace_id"]))
        if failed:
            picks.append((failed[0], "failed"))
        if risky:
            picks.append((risky[0], "risky"))
        if costly:
            picks.append((costly[0], "costly"))
        if common:
            picks.append((common[0], "common"))
        added = 0
        for trace, origin in picks:
            if added >= per_workflow_cap:
                break
            existed_before = trace["trace_id"] in used_traces
            add_candidate(trace, workflow["id"], origin)
            # The cap limits distinct cases per workflow, not the origins a
            # single trace collects when every pick lands on the same trace.
            if not existed_before:
                added += 1

    unassigned_ids = list(workflow_map.get("unassigned_trace_ids") or [])
    if unassigned_ids:
        trace = traces_by_id.get(unassigned_ids[0])
        if trace:
            add_candidate(trace, "unassigned", "common")

    # Deterministic ordering: workflow id, then case id; hard cap with failed
    # cases kept preferentially so the small set still sees failures.
    cases.sort(key=lambda case: (case["workflow_id"], case["case_id"]))
    if len(cases) > max_cases:
        failures = [case for case in cases if "failed" in case["origins"]]
        others = [case for case in cases if "failed" not in case["origins"]]
        keep = failures[:max_cases] if len(failures) >= max_cases else failures + others[: max_cases - len(failures)]
        cases = sorted(keep, key=lambda case: (case["workflow_id"], case["case_id"]))
    for position, case in enumerate(cases, start=1):
        case["case_id"] = f"case_{position:03d}"
    return {
        "cases_version": CASES_VERSION,
        "params": {"max_cases": max_cases, "per_workflow_cap": per_workflow_cap},
        "cases": cases,
        "notes": [
            "Checks are defaults generated from reference behavior. Edit cases.json to add explicit success checks where outcomes matter.",
            "Deterministic checks only. An LLM judge, if added by hand, is a fallible measurement tool, not ground truth.",
        ],
    }


def save_cases(cases_doc: dict, path) -> None:
    atomic_write_json(path, cases_doc)


def load_cases(path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)
