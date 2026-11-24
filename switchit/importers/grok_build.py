"""Parse grok build session records into raw calls."""

from __future__ import annotations

from pathlib import Path
import json
import re

from switchit.importers.common import (
    _gemini_text,
    _iter_jsonl,
    _safe_json,
    _tool_result_call,
)


def import_grok_build(path) -> list:
    """Parse one Grok Build session directory (``~/.grok/sessions/<cwd>/<id>/``).

    Grok Build is open source (xai-org/grok-build); the format below is read
    from its storage module (crates/codegen/xai-grok-shell/src/session/):

    - ``chat_history.jsonl`` — one ``ConversationItem`` per line, ``type`` of
      ``user`` (text wrapped in ``<user_query>`` tags), ``assistant`` (text +
      ``tool_calls`` with ``id``/``name``/``arguments`` as a JSON string),
      ``tool_result`` (``tool_call_id`` + ``content``), ``reasoning``
      (skipped — opaque), ``system`` (injected system prompt).
    - ``summary.json`` — {info: {id, cwd}, generated_title, current_model_id,
      session_kind, created_at, updated_at}.
    - ``signals.json`` — session-level token/turn counters (attached to the
      last assistant call, estimator-labeled: these are session totals).
    - ``system_prompt.txt`` — the rendered system prompt.
    - ``updates.jsonl`` — ACP display stream; ``turn_completed`` backfills the
      stop reason, ``tool_call_update`` marks failed results.
    """
    directory = Path(path)
    if directory.name in ("events.jsonl", "chat_history.jsonl", "updates.jsonl"):
        directory = directory.parent
    session_id = directory.name
    trace_id = f"grok-build:{session_id}"

    summary = {}
    summary_file = directory / "summary.json"
    if summary_file.exists():
        try:
            loaded = json.loads(summary_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (ValueError, OSError):
            summary = {}
    info = summary.get("info") or {}
    cwd = info.get("cwd") or summary.get("cwd")
    title = summary.get("generated_title") or summary.get("session_summary") or summary.get("title") or ""
    model = summary.get("current_model_id") or None
    session_kind = summary.get("session_kind")

    system_text = ""
    system_file = directory / "system_prompt.txt"
    if system_file.exists():
        try:
            system_text = system_file.read_text(encoding="utf-8", errors="replace")[:8000]
        except OSError:
            system_text = ""

    USER_QUERY = re.compile(r"<user_query>(.*)</user_query>", re.S)
    items = []
    history_file = directory / "chat_history.jsonl"
    if history_file.exists():
        for record in _iter_jsonl(history_file):
            if isinstance(record, dict) and record.get("type"):
                items.append(record)

    # updates.jsonl backfills: stop reason for the last turn, failed tool calls.
    stop_reason = None
    failed_tool_ids = set()
    updates_file = directory / "updates.jsonl"
    if updates_file.exists():
        for record in _iter_jsonl(updates_file):
            if not isinstance(record, dict):
                continue
            params = record.get("params") or {}
            update = params.get("update")
            if update is None and isinstance(params.get("update"), str):
                try:
                    update = json.loads(params["update"])
                except ValueError:
                    update = None
            if not isinstance(update, dict):
                continue
            kind = update.get("sessionUpdate")
            if kind == "turn_completed":
                stop_reason = update.get("stopReason") or stop_reason
            elif kind == "tool_call_update":
                if str(update.get("status")) == "failed" and update.get("toolCallId"):
                    failed_tool_ids.add(str(update["toolCallId"]))

    raw_calls = []
    tool_name_by_id = {}
    usage_attached = False
    signals = {}
    signals_file = directory / "signals.json"
    if signals_file.exists():
        try:
            loaded = json.loads(signals_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                signals = loaded.get("tokens") or loaded.get("usage") or loaded
        except (ValueError, OSError):
            signals = {}

    for item in items:
        item_type = str(item.get("type") or "").lower()
        if item_type == "system":
            if not system_text:
                system_text = _gemini_text(item.get("content") or item.get("text"))
            continue
        if item_type == "reasoning":
            continue  # opaque encrypted_content; nothing readable to import
        if item_type == "user":
            text = _gemini_text(item.get("content"))
            match = USER_QUERY.search(text)
            if match:
                text = match.group(1).strip()
            if not text.strip():
                continue
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": "grok-build",
                    "model": "user-turn",
                    "request": {"messages": [{"role": "user", "content": text}]},
                    "response": None,
                    "usage": None,
                }
            )
        elif item_type == "assistant":
            tool_calls = []
            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                call_id = str(call.get("id") or "")
                if name:
                    tool_calls.append({"name": name, "arguments": _safe_json(call.get("arguments")), "id": call_id})
                    if call_id:
                        tool_name_by_id[call_id] = name
            request_messages = []
            if system_text:
                request_messages.append({"role": "system", "content": system_text})
            usage = None
            usage_record = item.get("usage") or {}
            if usage_record.get("input_tokens") is not None or usage_record.get("output_tokens") is not None:
                usage = {"input_tokens": usage_record.get("input_tokens"),
                         "output_tokens": usage_record.get("output_tokens"), "basis": "recorded"}
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": "grok-build",
                    "model": item.get("model_id") or model,
                    "request": {"messages": request_messages},
                    "response": {
                        "text": _gemini_text(item.get("content")),
                        "tool_calls": tool_calls,
                        "finish_reason": stop_reason or ("tool_calls" if tool_calls else "stop"),
                    },
                    "usage": usage,
                }
            )
        elif item_type == "tool_result":
            call_id = str(item.get("tool_call_id") or "")
            content = item.get("content")
            text = content if isinstance(content, str) else _gemini_text(content)
            if not text:
                continue
            marker = " [tool failed]" if call_id in failed_tool_ids else ""
            raw_calls.append(_tool_result_call(
                trace_id, session_id, len(raw_calls),
                tool_name_by_id.get(call_id, "unknown_tool"),
                text + marker, call_id,
            ))

    # Session-level token totals from signals.json go on the last assistant
    # call, estimator-labeled: these are session totals, not per-call usage.
    total_input = signals.get("input_tokens") or signals.get("input")
    total_output = signals.get("output_tokens") or signals.get("output")
    if (total_input is not None or total_output is not None) and raw_calls:
        for raw in reversed(raw_calls):
            if raw["response"] is not None and raw["response"]["text"]:
                raw["usage"] = {
                    "input_tokens": total_input, "output_tokens": total_output,
                    "basis": "recorded", "estimator": "grok_signals_session_totals",
                }
                break
    if session_kind:
        pass  # session_kind is metadata; traces stay flat
    return raw_calls
