"""Parse codex session records into raw calls."""

from __future__ import annotations

from pathlib import Path

from switchit.importers.common import (
    attach_record_metadata,
    _content_text,
    _iter_jsonl,
    _safe_json,
    _tool_result_call,
)


def import_codex(path) -> list:
    """Parse one Codex rollout file (.jsonl)."""
    raw_calls = []
    session_id = None
    system_text = ""
    last_assistant_usage = None
    index = 0
    tool_names = {}
    for line in _iter_jsonl(Path(path)):
        record_start = len(raw_calls)
        line_type = line.get("type")
        payload = line.get("payload") or {}
        if line_type == "session_meta":
            session_id = payload.get("session_id") or payload.get("id") or Path(path).stem
            instructions = payload.get("base_instructions")
            if isinstance(instructions, dict):
                instructions = instructions.get("text")
            system_text = instructions or ""
            continue
        if line_type == "event_msg":
            info = payload.get("info") or {}
            usage = info.get("total_token_usage") or info.get("token_usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", usage.get("input"))
                output_tokens = usage.get("output_tokens", usage.get("output"))
                if input_tokens is not None or output_tokens is not None:
                    last_assistant_usage = {
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "basis": "recorded",
                    }
            continue
        if line_type != "response_item":
            continue
        payload_type = payload.get("type")
        trace_id = f"codex:{session_id or Path(path).stem}"
        if payload_type == "message":
            role = payload.get("role") or "user"
            text = _content_text(payload.get("content"))
            if not text:
                continue
            if role == "assistant":
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": f"{session_id}:assistant:{index}",
                        "session_id": str(session_id) if session_id else None,
                        "call_index": index,
                        "provider": "openai",
                        "model": None,
                        "request": {"messages": (
                            [{"role": "system", "content": system_text}]
                            if system_text and not any(
                                c["response"] and c["response"].get("text") for c in raw_calls
                            )
                            else []
                        )},
                        "response": {"text": text, "tool_calls": [], "finish_reason": "stop"},
                        "usage": last_assistant_usage,
                    }
                )
                index += 1
                last_assistant_usage = None
            elif role == "user" and raw_calls is not None:
                # Record the user turn as its own lightweight call so requests
                # stay linked to the answers that followed them.
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": f"{session_id}:user:{index}",
                        "session_id": str(session_id) if session_id else None,
                        "call_index": index,
                        "provider": "openai",
                        "model": "user-turn",
                        "request": {"messages": [{"role": "user", "content": text}]},
                        "response": None,
                        "usage": None,
                    }
                )
                index += 1
        elif payload_type == "function_call":
            tool_names[payload.get("call_id")] = payload.get("name", "")
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": payload.get("call_id") or f"{session_id}:fn:{index}",
                    "session_id": str(session_id) if session_id else None,
                    "call_index": index,
                    "provider": "openai",
                    "model": "tool-call",
                    "request": {"messages": []},
                    "response": {"text": "", "finish_reason": "tool_calls",
                                 "tool_calls": [{"id": payload.get("call_id"), "name": payload.get("name", ""),
                                                 "arguments": _safe_json(payload.get("arguments"))}]},
                    "usage": None,
                }
            )
            index += 1
        elif payload_type == "custom_tool_call":
            identity = payload.get("call_id")
            tool_names[identity] = payload.get("name", "")
            raw_calls.append({
                "trace_id": trace_id,
                "call_id": identity or f"{session_id}:custom:{index}",
                "source_record_id": payload.get("id"),
                "session_id": str(session_id) if session_id else None,
                "call_index": index, "provider": "openai", "model": "custom-tool-call",
                "request": {"messages": []},
                "response": {"text": "", "finish_reason": "tool_calls", "tool_calls": [{
                    "id": identity, "name": payload.get("name", ""),
                    "type": "custom", "input": payload.get("input"),
                }]},
                "usage": None,
                "context": {"source_record_type": "custom_tool_call", "custom_tool_requires_adapter": True},
            })
            index += 1
        elif payload_type in ("function_call_output", "custom_tool_call_output"):
            output = payload.get("output")
            custom = payload_type == "custom_tool_call_output"
            text = _content_text(output) if custom and isinstance(output, list) else str(output or "")
            result = _tool_result_call(trace_id, str(session_id), index, tool_names.get(payload.get("call_id")),
                                       text, payload.get("call_id"))
            result["provider"] = "openai"
            if payload.get("id"):
                result["source_record_id"] = payload["id"]
            if custom:
                message = result["request"]["messages"][0]
                message.update(tool_format="custom", output_format="content_blocks" if isinstance(output, list) else "text")
                message["content"] = output
                if isinstance(output, list):
                    message["content_blocks"] = output
                result["context"] = {"source_record_type": "custom_tool_call_output"}
            raw_calls.append(result)
            index += 1
        attach_record_metadata(raw_calls[record_start:], line)
    return raw_calls
