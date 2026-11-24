"""Existing call-to-trace-to-session aggregation.

This preserves the current flat containment model; it does not reconstruct
parent-agent, subagent, retry, or causal dependency edges."""

from __future__ import annotations

from collections import defaultdict

from switchit.usage import add_usage, empty_usage

MAX_USER_REQUESTS_PER_TRACE = 4


def build_traces(calls: list, session_meta: dict | None = None) -> list:
    """Aggregate canonical calls into traces with usage and evidence summaries."""
    grouped = defaultdict(list)
    for call in calls:
        grouped[call["trace_id"]].append(call)
    traces = []
    for trace_id in sorted(grouped):
        members = sorted(grouped[trace_id], key=lambda c: (c["call_index"], c["call_id"]))
        usage = empty_usage()
        error_count = 0
        tool_names = []
        user_requests = []
        latency_values = []
        session_id = None
        meta = (session_meta or {}).get(trace_id) or {}
        for call in members:
            add_usage(
                usage,
                call["usage"]["input_tokens"],
                call["usage"]["output_tokens"],
                call["usage"]["basis"],
                call["usage"]["estimator"],
            )
            response = call["response"]
            if response["error"]:
                error_count += 1
            for tool_call in response["tool_calls"]:
                if tool_call["name"] and tool_call["name"] not in tool_names:
                    tool_names.append(tool_call["name"])
            for tool_call in call["request"]["tools"]:
                if tool_call["name"] and tool_call["name"] not in tool_names:
                    tool_names.append(tool_call["name"])
            for message in call["request"]["messages"]:
                if message.get("role") == "user" and message.get("content"):
                    user_requests.append(message["content"])
            if call["latency_ms"] is not None:
                latency_values.append(call["latency_ms"])
            if call["session_id"]:
                session_id = call["session_id"]
        deduped_requests = []
        seen = set()
        for request in user_requests:
            key = request.casefold()[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped_requests.append(request)
        traces.append(
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "call_ids": [call["call_id"] for call in members],
                "call_count": len(members),
                "error_count": error_count,
                "tool_names": tool_names,
                "user_requests": deduped_requests[:MAX_USER_REQUESTS_PER_TRACE],
                "latency_ms_max": max(latency_values) if latency_values else None,
                "usage": usage,
            }
        )
        if meta:
            for key in ("title", "origin", "cwd"):
                if meta.get(key):
                    traces[-1][key] = meta[key]
    return traces


def build_sessions(traces: list) -> list:
    sessions = defaultdict(lambda: {"session_id": None, "trace_ids": [], "call_count": 0})
    for trace in traces:
        if not trace.get("session_id"):
            continue
        session = sessions[trace["session_id"]]
        session["session_id"] = trace["session_id"]
        session["trace_ids"].append(trace["trace_id"])
        session["call_count"] += trace["call_count"]
    return [sessions[key] for key in sorted(sessions)]
