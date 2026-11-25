"""Parse copilot cli session records into raw calls."""

from __future__ import annotations

from pathlib import Path

from switchit.importers.common import (
    _content_text,
    _iter_jsonl,
    _parse_time,
    _tool_result_call,
)


def import_copilot_cli(path) -> list:
    """Parse one Copilot CLI ``events.jsonl`` (``~/.copilot/session-state/<id>/``).

    Documented via the shipped event schema (session-events.schema.json): every
    line is {id, parentId, timestamp, type, data}; a parser reads
    ``session.start`` (cwd), ``system.message`` (system prompt),
    ``user.message`` (prompts), ``assistant.message`` (content, toolRequests,
    model), ``tool.execution_complete`` (results), ``session.usage_record``
    (token usage).
    """
    path = Path(path)
    raw_calls = []
    session_id = path.parent.name if path.parent.name != "session-state" else path.stem
    trace_id = f"copilot-cli:{session_id}"
    system_text = None
    system_attached = False
    pending_tools = {}
    last_usage = None
    for record in _iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        event_type = record.get("type")
        data = record.get("data") or {}
        timestamp = _parse_time(record.get("timestamp"))
        if event_type == "session.start":
            session_id = data.get("sessionId") or session_id
            trace_id = f"copilot-cli:{session_id}"
        elif event_type == "system.message":
            if data.get("content"):
                system_text = _content_text(data.get("content"))
        elif event_type == "user.message":
            text = _content_text(data.get("content"))
            if text.strip():
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": data.get("messageId") or f"{session_id}:u:{len(raw_calls)}",
                        "session_id": session_id,
                        "call_index": len(raw_calls),
                        "provider": "copilot-cli",
                        "model": "user-turn",
                        "request": {"messages": [{"role": "user", "content": text}]},
                        "response": None,
                        "usage": None,
                    }
                )
        elif event_type == "assistant.message":
            requests = [
                {"name": request.get("name", ""),
                 "arguments": request.get("arguments") if isinstance(request.get("arguments"), dict) else {}}
                for request in data.get("toolRequests") or []
                if isinstance(request, dict) and request.get("name")
            ]
            for request in data.get("toolRequests") or []:
                if isinstance(request, dict) and request.get("toolCallId") and request.get("name"):
                    pending_tools[request["toolCallId"]] = request["name"]
            request_messages = []
            if system_text and not system_attached:
                request_messages.append({"role": "system", "content": system_text})
                system_attached = True
            usage = last_usage if last_usage else None
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": data.get("messageId") or f"{session_id}:a:{len(raw_calls)}",
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": "github-copilot",
                    "model": data.get("model"),
                    "request": {"messages": request_messages},
                    "response": {
                        "text": _content_text(data.get("content")),
                        "tool_calls": requests,
                        "finish_reason": "tool_calls" if requests else "stop",
                    },
                    "usage": usage,
                }
            )
            last_usage = None
        elif event_type == "tool.execution_complete":
            data_tool_id = data.get("toolCallId")
            tool_name = pending_tools.get(data_tool_id) or data.get("toolName") or "unknown_tool"
            result = data.get("result") or {}
            result_text = _content_text(result.get("content") if isinstance(result, dict) else result)
            if result_text:
                raw_calls.append(_tool_result_call(
                    trace_id, session_id, len(raw_calls), tool_name, result_text,
                ))
        elif event_type == "session.usage_record":
            usage = data.get("usage") or {}
            if usage.get("inputTokens") is not None or usage.get("outputTokens") is not None:
                last_usage = {"input_tokens": usage.get("inputTokens"),
                              "output_tokens": usage.get("outputTokens"), "basis": "recorded"}
    return raw_calls
