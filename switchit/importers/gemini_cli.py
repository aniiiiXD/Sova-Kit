"""Parse gemini cli session records into raw calls."""

from __future__ import annotations

from pathlib import Path

from switchit.importers.common import _gemini_text, _iter_jsonl, _tool_result_call


def import_gemini_cli(path) -> list:
    """Parse one Gemini CLI session transcript (``~/.gemini/tmp/<slug>/chats/*.jsonl``).

    Documented in gemini-cli's source (chatRecordingTypes.ts): a metadata line,
    then message records {id, timestamp, type: user|gemini|info|error, content,
    toolCalls[{id, name, args, result, status}], tokens{input, output}, model},
    plus update records ``$set`` / ``$patch`` applied by id.
    """
    path = Path(path)
    messages_by_id = {}
    order = []
    session_id = path.stem
    kind = "main"
    for record in _iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        if record.get("sessionId") and "id" not in record:
            session_id = record["sessionId"]  # the metadata line
            continue
        if "$set" in record:
            payload = record["$set"] or {}
            if isinstance(payload.get("messages"), list):
                messages_by_id.clear()
                order.clear()
                for message in payload["messages"]:
                    if isinstance(message, dict) and message.get("id"):
                        messages_by_id[message["id"]] = message
                        order.append(message["id"])
            for key in ("sessionId", "kind"):
                if payload.get(key):
                    if key == "sessionId":
                        session_id = payload[key]
                    else:
                        kind = payload[key]
            continue
        if "$patch" in record:
            payload = record["$patch"] or {}
            for update in payload.get("updates") or []:
                if not isinstance(update, dict) or not update.get("id"):
                    continue
                existing = messages_by_id.setdefault(update["id"], {"id": update["id"]})
                if update["id"] not in order:
                    order.append(update["id"])
                for key in ("content", "toolCalls", "tokens", "model", "type", "timestamp"):
                    if update.get(key) is not None:
                        existing[key] = update[key]
            continue
        if "$rewindTo" in record:
            rewind_id = record["$rewindTo"]
            if rewind_id in order:
                position = order.index(rewind_id)
                for dropped in order[position + 1:]:
                    messages_by_id.pop(dropped, None)
                order = order[: position + 1]
            continue
        if record.get("id") is not None:
            message_id = record["id"]
            if message_id in messages_by_id:
                messages_by_id[message_id].update(record)
            else:
                messages_by_id[message_id] = record
                order.append(message_id)

    trace_id = f"gemini-cli:{session_id}"
    raw_calls = []
    for message_id in order:
        message = messages_by_id[message_id]
        message_type = message.get("type")
        if message_type == "user":
            text = _gemini_text(message.get("content"))
            if text.strip():
                raw_calls.append(
                    {
                        "trace_id": trace_id,
                        "call_id": message_id,
                        "session_id": session_id,
                        "call_index": len(raw_calls),
                        "provider": "gemini-cli",
                        "model": "user-turn",
                        "request": {"messages": [{"role": "user", "content": text}]},
                        "response": None,
                        "usage": None,
                    }
                )
        elif message_type == "gemini":
            tokens = message.get("tokens") or {}
            usage = None
            if tokens.get("input") is not None or tokens.get("output") is not None:
                usage = {"input_tokens": tokens.get("input"),
                         "output_tokens": tokens.get("output"), "basis": "recorded"}
            tool_calls = [
                {"name": call.get("name", ""), "arguments": call.get("args") or {}}
                for call in message.get("toolCalls") or []
                if isinstance(call, dict) and call.get("name")
            ]
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": message_id,
                    "session_id": session_id,
                    "call_index": len(raw_calls),
                    "provider": "google",
                    "model": message.get("model"),
                    "request": {"messages": []},
                    "response": {
                        "text": _gemini_text(message.get("content")),
                        "tool_calls": tool_calls,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    },
                    "usage": usage,
                }
            )
            # Gemini records each tool result inside the same message; emit
            # them as tool-result calls so chained fixtures can pair them.
            for call in message.get("toolCalls") or []:
                if not isinstance(call, dict) or not call.get("name"):
                    continue
                result = call.get("result")
                result_text = _gemini_text(result)
                if result_text:
                    raw_calls.append(_tool_result_call(
                        trace_id, session_id, len(raw_calls), call["name"], result_text,
                    ))
    return raw_calls
