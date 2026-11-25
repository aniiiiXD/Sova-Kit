"""Infer prompts, tool schemas, provider mix, and dependencies from the store."""

from __future__ import annotations

import json

from switchit.redaction import excerpt, redact_value


def _json_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def infer_tool_schema(observations: list) -> dict:
    """Infer a JSON schema for a tool from its observed argument dicts.

    The result is an inference from recorded behavior — a starting point for
    the reviewer, not a recovered ground truth. Keys present in at least 80%
    of observed calls are marked required.
    """
    observations = [obs for obs in observations if isinstance(obs, dict)]
    if not observations:
        return {"type": "object", "properties": {}, "inferred_from": 0}
    total = len(observations)
    keys = {}
    for obs in observations:
        for key, value in obs.items():
            entry = keys.setdefault(key, {"types": set(), "count": 0})
            entry["count"] += 1
            entry["types"].add(_json_type(value))
    properties = {}
    required = []
    for key, entry in sorted(keys.items()):
        types = entry["types"] - {"unknown"}
        if len(types) == 1:
            property_type = next(iter(types))
        elif "string" in types or not types:
            property_type = "string"
        else:
            property_type = sorted(types)[0]
        properties[key] = {"type": property_type}
        if entry["count"] >= 0.8 * total:
            required.append(key)
    schema = {"type": "object", "properties": properties, "inferred_from": total}
    if required:
        schema["required"] = required
    return schema


def build_profile(store: dict) -> dict:
    """Dig through every session and surface: system prompts, a tool inventory
    with schemas inferred from observed calls, provider/model mix, error
    samples, and usage by provenance. Everything inside was redacted at
    import; excerpts stay capped."""
    from collections import Counter, defaultdict

    tool_observations = defaultdict(list)
    tool_call_counts = Counter()
    tool_traces = defaultdict(set)
    system_prompts = Counter()
    provider_models = Counter()
    error_samples = []
    calls_with_errors = 0

    for call in store["calls"]:
        provider_models[(call["provider"], call["model"])] += 1
        response = call["response"]
        if response["error"]:
            calls_with_errors += 1
            if len(error_samples) < 5:
                error_samples.append({"trace_id": call["trace_id"], "error": response["error"]})
        for message in call["request"]["messages"]:
            if message.get("role") == "system" and message.get("content"):
                system_prompts[message["content"]] += 1
        for tool_call in response["tool_calls"]:
            name = tool_call.get("name", "")
            if not name:
                continue
            tool_call_counts[name] += 1
            tool_observations[name].append(tool_call.get("arguments") or {})

    # Which traces use each tool.
    for trace in store["traces"]:
        for name in trace["tool_names"]:
            tool_traces[name].add(trace["trace_id"])

    tools = []
    for name, count in sorted(tool_call_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        observations = tool_observations[name]
        examples = []
        for obs in observations[:3]:
            examples.append(json.dumps(redact_value(obs), sort_keys=True)[:300])
        tools.append(
            {
                "name": name,
                "proposals": count,
                "traces": len(tool_traces[name]),
                "schema_inferred": infer_tool_schema(observations),
                "example_arguments": examples,
            }
        )

    # Also surface tools declared in the request but never proposed.
    for trace in store["traces"]:
        for name in trace["tool_names"]:
            if name not in tool_call_counts:
                tool_call_counts[name] = 0

    prompts = [
        {"excerpt": excerpt(prompt, 600), "calls_using": count}
        for prompt, count in system_prompts.most_common(5)
    ]

    providers = [
        {"provider": provider, "model": model, "calls": count}
        for (provider, model), count in provider_models.most_common(8)
    ]

    # Session origins — the cloud/local split, when the source recorded it.
    origin_counts = Counter()
    titled_sessions = []
    for trace in store["traces"]:
        origin = trace.get("origin")
        if origin:
            origin_counts[origin] += 1
        if trace.get("title"):
            titled_sessions.append(
                {"trace_id": trace["trace_id"], "title": excerpt(trace["title"], 120),
                 "origin": trace.get("origin"), "calls": trace["call_count"]}
            )
    titled_sessions = titled_sessions[:8]

    primary = providers[0] if providers else None
    inferred_config = {
        "name": "inferred-from-traces",
        "inferred": True,
        "prompts": [
            {"id": f"system_inferred_{position}", "role": "system", "template": prompt["excerpt"]}
            for position, prompt in enumerate(prompts[:1])
        ],
        "tools": [
            {
                "name": tool["name"],
                "description": "(inferred from observed calls; edit before use)",
                "parameters": {
                    key: value for key, value in tool["schema_inferred"].items()
                },
            }
            for tool in tools
        ],
        "provider": (
            {"provider": primary["provider"], "model": primary["model"], "base_url": None}
            if primary else None
        ),
    }

    sessions = len(store.get("sessions") or [])
    profile = {
        "profile_version": 1,
        "totals": {
            "traces": store["totals"]["traces"],
            "calls": store["totals"]["calls"],
            "sessions": sessions,
            "calls_with_errors": calls_with_errors,
        },
        "usage": store["totals"]["usage"],
        "providers": providers,
        "session_origins": dict(origin_counts),
        "named_sessions": titled_sessions,
        "system_prompts": prompts,
        "tools": tools,
        "error_samples": error_samples,
        "notes": [
            "Tool schemas are inferred from observed arguments; a reviewer should confirm them before reuse.",
            "All content was redacted at import; excerpts are capped.",
        ],
    }
    return {"profile": profile, "app_config_inferred": inferred_config}
