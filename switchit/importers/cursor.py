"""Parse cursor session records into raw calls."""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3

from switchit.importers.common import _content_text, _safe_json


def import_cursor(path) -> list:
    """Parse one Cursor chat (a folder containing store.db + meta.json).

    Blobs are JSON messages {role, content}; usage is not reported, so these
    calls land in the `unknown` provenance bucket. Tool calls appear only in
    agentic chats and are extracted tolerantly from whatever block shape the
    version used.
    """
    path = Path(path)
    store_db = path if path.name == "store.db" else path / "store.db"
    if not store_db.exists():
        return []
    chat_id = store_db.parent.name
    trace_id = f"cursor:{chat_id}"
    raw_calls = []
    system_text = None
    index = 0
    connection = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT data FROM blobs ORDER BY rowid")
        for (blob,) in rows:
            try:
                message = json.loads(blob)
            except (ValueError, TypeError):
                continue
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system_text = _content_text(content)
                continue
            if role == "assistant":
                text, tool_calls = _cursor_assistant_content(content)
                request_messages = []
                if system_text:
                    request_messages.append({"role": "system", "content": system_text})
                    system_text = None  # attach once, to the first assistant call
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": message.get("id") or f"{chat_id}:assistant:{index}",
                        "session_id": chat_id,
                        "call_index": index,
                        "provider": "cursor",
                        "model": message.get("model") or None,
                        "request": {"messages": request_messages},
                        "response": {
                            "text": text,
                            "tool_calls": tool_calls,
                            "finish_reason": "stop" if not tool_calls else "tool_calls",
                        },
                        "usage": None,
                    }
                )
                index += 1
            elif role == "user":
                text = _content_text(content)
                if text.strip():
                    raw_calls.append(
                        {
                            "trace_id": trace_id,
                            "call_id": f"{chat_id}:user:{index}",
                            "session_id": chat_id,
                            "call_index": index,
                            "provider": "cursor",
                            "model": "user-turn",
                            "request": {"messages": [{"role": "user", "content": text}]},
                            "response": None,
                            "usage": None,
                        }
                    )
                    index += 1
    except sqlite3.Error:
        return raw_calls
    finally:
        connection.close()
    return raw_calls


def _cursor_assistant_content(content) -> tuple:
    """Extract (text, tool_calls) tolerantly from a Cursor assistant blob."""
    tool_calls = []
    if isinstance(content, str):
        return content, tool_calls
    texts = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            texts.append(block["text"])
        wrapped = block.get("toolCall") or block.get("tool_use") or block.get("function_call")
        candidate = wrapped if isinstance(wrapped, dict) else block
        name = candidate.get("name") or (candidate.get("function") or {}).get("name")
        if name:
            arguments = (candidate.get("arguments") or candidate.get("input")
                         or (candidate.get("function") or {}).get("arguments") or {})
            if isinstance(arguments, str):
                arguments = _safe_json(arguments)
            tool_calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return "\n".join(t for t in texts if t), tool_calls
