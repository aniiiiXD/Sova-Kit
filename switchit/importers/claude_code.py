"""Parse claude code session records into raw calls."""

from __future__ import annotations

from pathlib import Path

from switchit.importers.common import (
    attach_record_metadata,
    _content_text,
    _iter_jsonl,
    _parse_time,
    _tool_result_call,
    _usage_recorded,
)


def import_claude_code(path) -> list:
    """Parse one Claude Code session transcript (.jsonl)."""
    raw_calls = []
    session_id = Path(path).stem
    pending_tools = []  # tool_use proposals awaiting their results
    tools_by_id = {}
    last_time = None
    for line in _iter_jsonl(Path(path)):
        record_start = len(raw_calls)
        line_type = line.get("type")
        if line_type not in ("user", "assistant"):
            continue
        session_id = line.get("sessionId") or session_id
        message = line.get("message") or {}
        role = message.get("role") or line_type
        timestamp = _parse_time(line.get("timestamp"))
        if role == "assistant":
            blocks = message.get("content")
            if isinstance(blocks, str):
                text, tool_blocks = blocks, []
            else:
                text = _content_text(
                    [b for b in (blocks or []) if isinstance(b, dict) and b.get("type") == "text"]
                )
                tool_blocks = [b for b in (blocks or [])
                               if isinstance(b, dict) and b.get("type") == "tool_use"]
            tool_calls = [
                {"id": block.get("id"), "name": block.get("name", ""), "arguments": block.get("input") or {}}
                for block in tool_blocks
            ]
            pending_tools = [call["name"] for call in tool_calls if call["name"]]
            tools_by_id.update({call["id"]: call["name"] for call in tool_calls if call.get("id")})
            raw_calls.append(
                {
                    "trace_id": f"claude-code:{session_id}",
                    "call_id": message.get("id"),
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": "anthropic",
                    "model": message.get("model"),
                    "request": {"messages": []},
                    "response": {
                        "text": text,
                        "tool_calls": tool_calls,
                        "finish_reason": message.get("stop_reason"),
                    },
                    "latency_ms": int((timestamp - last_time) * 1000)
                    if timestamp and last_time and timestamp > last_time else None,
                    "usage": _usage_recorded(message.get("usage")),
                }
            )
            last_time = timestamp or last_time
        elif role == "user":
            blocks = message.get("content")
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_text = _content_text(block.get("content"))
                        if result_text:
                            raw_calls.append(_tool_result_call(
                                f"claude-code:{session_id}", session_id, len(raw_calls),
                                tools_by_id.get(block.get("tool_use_id")) or (pending_tools.pop(0) if pending_tools else None), result_text,
                                block.get("tool_use_id"),
                            ))
            elif isinstance(blocks, str) and blocks.strip():
                raw_calls.append(
                    {
                        "trace_id": f"claude-code:{session_id}",
                        "call_id": f"{session_id}:user:{len(raw_calls)}",
                        "session_id": session_id,
                        "call_index": len(raw_calls),
                        "provider": "anthropic",
                        "model": "user-turn",
                        "request": {"messages": [{"role": "user", "content": blocks}]},
                        "response": None,
                        "usage": None,
                    }
                )
        attach_record_metadata(raw_calls[record_start:], line)
    return raw_calls
