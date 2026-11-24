"""Convert raw records to the existing canonical call format."""

from __future__ import annotations

import hashlib
import json

from switchit.redaction import (
    excerpt,
    redact_messages,
    redact_literal_value,
    redact_text,
    redact_tools,
    redact_value,
)
from switchit.usage import normalize_basis, usage_from_record


def _stable_id(*parts) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return digest.hexdigest()[:24]


def normalize_tool_call(raw) -> dict:
    """Coerce a historical tool call into {name, arguments-dict}."""
    if not isinstance(raw, dict):
        return {"name": redact_text(str(raw))[:200], "arguments": {}}
    function = raw.get("function") if isinstance(raw.get("function"), dict) else None
    name = raw.get("name") or (function or {}).get("name") or ""
    if raw.get("type") == "custom":
        # _custom_input is a transparent storage envelope, not an inferred JSON schema.
        value = redact_literal_value(raw.get("input"))
        result = {"name": redact_text(str(name))[:200], "arguments": {"_custom_input": value},
                  "type": "custom", "input_format": "freeform" if isinstance(value, str) else "unknown", "input": value}
        identity = raw.get("id") or raw.get("call_id") or raw.get("tool_call_id")
        if identity:
            result["id"] = str(identity)
        return result
    arguments = raw.get("arguments", raw.get("args", (function or {}).get("arguments", {})))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"_raw": arguments[:400]}
    if not isinstance(arguments, dict):
        arguments = {}
    result = {"name": redact_text(str(name))[:200], "arguments": redact_value(arguments)}
    identity = raw.get("id") or raw.get("call_id") or raw.get("tool_call_id")
    if identity:
        result["id"] = str(identity)
    return result


def normalize_response(response) -> dict:
    """Coerce any historical output shape into the canonical response dict."""
    result = {"text": "", "tool_calls": [], "finish_reason": None, "error": None}
    if response is None:
        return result
    if isinstance(response, str):
        result["text"] = excerpt(response)
        return result
    if not isinstance(response, dict):
        result["text"] = excerpt(str(response))
        return result

    def absorb_message(message: dict) -> None:
        role = str(message.get("role") or "")
        if role and role != "assistant" and not message.get("tool_calls"):
            return
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    result["text"] = excerpt((result["text"] + " " + block["text"]).strip())
                elif isinstance(block, str):
                    result["text"] = excerpt((result["text"] + " " + block).strip())
        elif isinstance(content, str) and content:
            result["text"] = excerpt((result["text"] + " " + content).strip())
        for call in message.get("tool_calls") or []:
            if call:
                result["tool_calls"].append(normalize_tool_call(call))

    messages = response.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                absorb_message(message)
    message = response.get("message")
    if isinstance(message, dict):
        absorb_message(message)
        if result["finish_reason"] is None:
            result["finish_reason"] = message.get("finish_reason", message.get("stop_reason"))
        if result["error"] is None and message.get("error"):
            result["error"] = redact_text(str(message["error"]))[:400]

    if not result["tool_calls"]:
        for call in response.get("tool_calls") or []:
            if call:
                result["tool_calls"].append(normalize_tool_call(call))
    if not result["text"]:
        content = response.get("content", response.get("text", response.get("output", "")))
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    if block.get("type") in ("tool_use", "function_call") or block.get("name"):
                        result["tool_calls"].append(normalize_tool_call(block))
                elif isinstance(block, str):
                    parts.append(block)
            result["text"] = excerpt(" ".join(p for p in parts if p))
        else:
            result["text"] = excerpt(content if content is not None else "")
    if result["finish_reason"] is None:
        result["finish_reason"] = response.get("finish_reason", response.get("stop_reason"))
    if result["error"] is None and response.get("error") is not None:
        result["error"] = redact_text(str(response["error"]))[:400]
    return result


def normalize_request(request) -> dict:
    """Coerce any historical input shape into {"messages": [...], "tools": [...]}."""
    messages = []
    tools = []
    if isinstance(request, dict):
        raw_messages = request.get("messages", request.get("input_messages"))
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except ValueError:
                raw_messages = [{"role": "user", "content": raw_messages}]
        if isinstance(raw_messages, list):
            messages = raw_messages
        raw_tools = request.get("tools", request.get("tool_definitions"))
        if isinstance(raw_tools, list):
            tools = raw_tools
        if not messages and "prompt" in request:
            messages = [{"role": "user", "content": str(request.get("prompt"))}]
    elif isinstance(request, str):
        try:
            parsed = json.loads(request)
            return normalize_request(parsed)
        except ValueError:
            messages = [{"role": "user", "content": request}]
    elif isinstance(request, list):
        messages = request
    return {"messages": redact_messages(messages), "tools": redact_tools(tools)}


def make_call(raw: dict, importer: str, source_location: str) -> dict:
    """Build one canonical call from a raw record. Original IDs are preserved."""
    trace_id = raw.get("trace_id") or raw.get("traceId") or raw.get("conversation_id")
    if not trace_id:
        raise ValueError(f"{source_location}: record has no trace_id")
    call_id = raw.get("call_id") or raw.get("id") or raw.get("span_id") or raw.get("observation_id")
    call_index = raw.get("call_index")
    if call_index is None:
        call_index = raw.get("index")
    span_id = raw.get("span_id") or raw.get("spanId")
    session_id = raw.get("session_id") or raw.get("sessionId") or raw.get("session")
    input_tokens, output_tokens, basis, estimator = usage_from_record(raw)
    response = normalize_response(raw.get("response"))
    error = raw.get("error")
    if error is not None:
        if isinstance(error, str):
            response["error"] = redact_text(error)[:400] or response["error"]
        else:
            response["error"] = json.dumps(redact_value(error))[:400]
    result = {
        "call_id": str(call_id) if call_id else _stable_id("call", importer, source_location, trace_id, call_index),
        "trace_id": str(trace_id),
        "session_id": str(session_id) if session_id else None,
        "span_id": str(span_id) if span_id else None,
        "call_index": int(call_index) if call_index is not None else 0,
        "provider": redact_text(str(raw.get("provider") or raw.get("gen_ai_system") or "unknown"))[:120],
        "model": redact_text(str(raw.get("model") or "unknown"))[:160],
        "request": normalize_request(raw.get("request")),
        "response": response,
        "latency_ms": _coerce_latency(raw.get("latency_ms")),
        "usage": {
            "input_tokens": _coerce_int(input_tokens),
            "output_tokens": _coerce_int(output_tokens),
            "basis": normalize_basis(basis, estimator)[0],
            "estimator": estimator,
        },
        "source": {"importer": importer, "location": str(source_location)},
    }
    aliases = {"timestamp": ("timestamp", "start_time", "startTimeUnixNano"),
               "parent_id": ("parent_id", "parentId", "parentUuid"),
               "parent_span_id": ("parent_span_id", "parentSpanId"),
               "agent_id": ("agent_id", "agentId"),
               "parent_agent_id": ("parent_agent_id", "parentAgentId"),
               "retry_of": ("retry_of",), "handoff_from": ("handoff_from",),
               "source_record_id": ("source_record_id",)}
    for field, names in aliases.items():
        for name in names:
            if raw.get(name) is not None:
                result[field] = raw[name]
                break
    if isinstance(raw.get("context"), dict):
        result["context"] = redact_value(raw["context"])
    return result


def _coerce_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_latency(value):
    if value is None:
        return None
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return None
