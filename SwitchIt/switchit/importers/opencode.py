"""Parse opencode session records into raw calls."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import sqlite3

from switchit.importers.common import _usage_recorded


def import_opencode(path) -> list:
    """Parse an OpenCode storage database (message + part tables)."""
    path = Path(path)
    if not path.exists():
        return []
    raw_calls = []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        parts_by_message = defaultdict(list)
        try:
            for row in connection.execute("SELECT message_id, data FROM part"):
                try:
                    parts_by_message[row["message_id"]].append(json.loads(row["data"]))
                except ValueError:
                    continue
        except sqlite3.OperationalError:
            pass  # older storage layouts keep parts inline
        counters = defaultdict(int)
        for row in connection.execute("SELECT id, session_id, data FROM message ORDER BY rowid"):
            try:
                data = json.loads(row["data"])
            except ValueError:
                continue
            role = data.get("role")
            session_id = row["session_id"]
            trace_id = f"opencode:{session_id}"
            parts = parts_by_message.get(row["id"]) or data.get("parts") or []
            if role == "assistant":
                text_parts = [part.get("text", "") for part in parts
                              if isinstance(part, dict) and part.get("type") == "text"]
                tool_calls = [
                    {"name": part.get("tool") or part.get("name", ""),
                     "arguments": (part.get("state") or {}).get("input") or part.get("input") or {}}
                    for part in parts
                    if isinstance(part, dict) and part.get("type") in ("tool", "tool-invocation")
                ]
                usage = data.get("tokens") or {}
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": row["id"],
                        "session_id": session_id,
                        "call_index": counters[session_id],
                        "provider": data.get("providerID") or (data.get("model") or {}).get("providerID") or "opencode",
                        "model": data.get("modelID") or (data.get("model") or {}).get("modelID"),
                        "request": {"messages": []},
                        "response": {
                            "text": "\n".join(t for t in text_parts if t),
                            "tool_calls": tool_calls,
                            "finish_reason": data.get("finish"),
                        },
                        "usage": _usage_recorded(usage) or (
                            {"input_tokens": usage.get("input"), "output_tokens": usage.get("output"),
                             "basis": "recorded"}
                            if usage.get("input") is not None or usage.get("output") is not None else None
                        ),
                    }
                )
                counters[session_id] += 1
            elif role == "user":
                text_parts = [part.get("text", "") for part in parts
                              if isinstance(part, dict) and part.get("type") == "text"]
                text = "\n".join(t for t in text_parts if t)
                if text:
                    raw_calls.append(
                        {
                            "trace_id": trace_id,
                            "call_id": row["id"],
                            "session_id": session_id,
                            "call_index": counters[session_id],
                            "provider": "opencode",
                            "model": "user-turn",
                            "request": {"messages": [{"role": "user", "content": text}]},
                            "response": None,
                            "usage": None,
                        }
                    )
                    counters[session_id] += 1
    finally:
        connection.close()
    return raw_calls
