"""Parse cline session records into raw calls."""

from __future__ import annotations

from pathlib import Path
import json

from switchit.importers.common import _tool_result_call


def import_cline(path) -> list:
    """Parse a Cline store: SDK sessions, legacy task dirs, or one file.

    Shapes (documented in Cline's source, sdk/packages/shared/src/storage):
    - ``<dir>/sessions/<id>/<id>.messages.json`` — SDK session transcripts.
    - ``<dir>/tasks/<id>/api_conversation_history.json`` — legacy Anthropic-style
      conversation arrays, next to ``ui_messages.json`` for usage.
    Both go through the same redaction pipeline as every other importer.
    """
    path = Path(path)
    raw_calls = []
    if path.is_file():
        if path.name.endswith(".messages.json"):
            raw_calls.extend(_cline_sdk_messages_file(path))
        elif path.name == "api_conversation_history.json":
            raw_calls.extend(_cline_legacy_task(path.parent))
        return raw_calls
    if (path / "api_conversation_history.json").exists():
        return _cline_legacy_task(path)
    sessions_dir = path / "data" / "sessions" if (path / "data").exists() else path / "sessions"
    if sessions_dir.exists():
        for messages_file in sorted(sessions_dir.glob("*/*.messages.json")):
            raw_calls.extend(_cline_sdk_messages_file(messages_file))
    if (path / "tasks").exists():
        for task_dir in sorted(candidate for candidate in (path / "tasks").iterdir() if candidate.is_dir()):
            history_file = task_dir / "api_conversation_history.json"
            if history_file.exists():
                raw_calls.extend(_cline_legacy_task(task_dir))
    return raw_calls


def _cline_sdk_messages_file(messages_file: Path) -> list:
    try:
        document = json.loads(messages_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(document, dict):
        return []
    session_id = document.get("sessionId") or messages_file.parent.name
    trace_id = f"cline:{session_id}"
    system_text = document.get("system_prompt") or ""
    raw_calls = []
    system_attached = False
    for message in document.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text_parts = [b.get("text", "") for b in blocks
                      if isinstance(b, dict) and b.get("type") == "text"]
        tool_calls = [
            {"name": b.get("name", ""), "arguments": b.get("input") or {}}
            for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
        ]
        metrics = message.get("metrics") or {}
        usage = None
        if metrics.get("inputTokens") is not None or metrics.get("outputTokens") is not None:
            usage = {"input_tokens": metrics.get("inputTokens"),
                     "output_tokens": metrics.get("outputTokens"), "basis": "recorded"}
        if role == "assistant":
            request_messages = []
            if system_text and not system_attached:
                request_messages.append({"role": "system", "content": system_text})
                system_attached = True
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": message.get("id") or f"{session_id}:a:{len(raw_calls)}",
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": (message.get("modelInfo") or {}).get("provider") or "cline",
                    "model": (message.get("modelInfo") or {}).get("id"),
                    "request": {"messages": request_messages},
                    "response": {"text": "\n".join(t for t in text_parts if t),
                                 "tool_calls": tool_calls, "finish_reason": "stop"},
                    "usage": usage,
                }
            )
        elif role == "user":
            tool_results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for result in tool_results:
                    result_text = _cline_result_text(result.get("content"))
                    if result_text:
                        raw_calls.append(_tool_result_call(
                            trace_id, session_id, len(raw_calls),
                            result.get("name") or "unknown_tool", result_text,
                        ))
            else:
                text = "\n".join(t for t in text_parts if t)
                if text.strip():
                    raw_calls.append(
                        {
                            "trace_id": trace_id,
                            "call_id": f"{session_id}:u:{len(raw_calls)}",
                            "session_id": session_id,
                            "call_index": len(raw_calls),
                            "provider": "cline",
                            "model": "user-turn",
                            "request": {"messages": [{"role": "user", "content": text}]},
                            "response": None,
                            "usage": None,
                        }
                    )
    return raw_calls


def _cline_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_cline_result_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("output") or "")
    return ""


def _cline_legacy_task(task_dir: Path) -> list:
    history_file = task_dir / "api_conversation_history.json"
    try:
        conversation = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(conversation, list):
        return []
    task_id = task_dir.name
    trace_id = f"cline:{task_id}"
    # Usage lives in ui_messages.json say/api_req_started entries, in order.
    usage_events = []
    ui_file = task_dir / "ui_messages.json"
    if ui_file.exists():
        try:
            ui_messages = json.loads(ui_file.read_text(encoding="utf-8"))
            for entry in ui_messages or []:
                if not isinstance(entry, dict) or entry.get("say") != "api_req_started":
                    continue
                try:
                    info = json.loads(entry.get("text") or "{}")
                except ValueError:
                    continue
                if info.get("tokensIn") is not None or info.get("tokensOut") is not None:
                    usage_events.append({"input_tokens": info.get("tokensIn"),
                                         "output_tokens": info.get("tokensOut"),
                                         "basis": "recorded"})
        except (ValueError, OSError):
            usage_events = []
    raw_calls = []
    usage_index = 0
    for position, message in enumerate(conversation):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        if role == "assistant":
            tool_calls = [
                {"name": b.get("name", ""), "arguments": b.get("input") or {}}
                for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
            ]
            text = _cline_result_text([b.get("text", "") for b in blocks
                                       if isinstance(b, dict) and b.get("type") == "text"])
            usage = usage_events[usage_index] if usage_index < len(usage_events) else None
            usage_index += 1
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": f"{task_id}:a:{position}",
                    "session_id": task_id,
                    "call_index": position,
                    "provider": "cline",
                    "model": None,
                    "request": {"messages": []},
                    "response": {"text": text, "tool_calls": tool_calls, "finish_reason": "stop"},
                    "usage": usage,
                }
            )
        elif role == "user":
            tool_results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for result in tool_results:
                    result_text = _cline_result_text(result.get("content"))
                    if result_text:
                        raw_calls.append(_tool_result_call(
                            trace_id, task_id, len(raw_calls),
                            result.get("name") or "unknown_tool", result_text,
                        ))
            else:
                text = _cline_result_text(content if isinstance(content, list)
                                          else [{"type": "text", "text": content or ""}])
                if text.strip():
                    raw_calls.append(
                        {
                            "trace_id": trace_id,
                            "call_id": f"{task_id}:u:{position}",
                            "session_id": task_id,
                            "call_index": position,
                            "provider": "cline",
                            "model": "user-turn",
                            "request": {"messages": [{"role": "user", "content": text}]},
                            "response": None,
                            "usage": None,
                        }
                    )
    return raw_calls
