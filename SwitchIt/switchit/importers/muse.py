"""Parse muse session records into raw calls."""

from __future__ import annotations

from pathlib import Path

from switchit.importers.common import _iter_jsonl, _safe_json, _tool_result_call


def import_muse(path) -> list:
    """Parse one Muse Code ``session.jsonl`` (Meta's coding agent CLI).

    Layout documented via Meta's own session-evidence reader and the binary's
    serde field names: every line is an envelope with ``recorded_at``
    (microseconds), ``stream.id`` (the session id) and a ``payload`` keyed by
    ``payload_type``. The conversation lives in
    ``payload_type:"runtime.session"`` with ``payload.kind:"run"``:
    ``event.kind`` ``started`` (the user prompt), ``assistant_message_committed``,
    ``assistant_tool_calls_committed``, ``tool_result_batch_committed``,
    ``model_completed`` (usage). Fact records nest values under
    ``payload.record``. Subagent logs share the same format and import as
    their own traces.
    """
    path = Path(path)
    raw_calls = []
    session_id = path.parent.name
    trace_id = f"muse:{session_id}"
    model_info = {}
    runs = {}
    run_order = []

    def run_slot(run_id):
        if run_id not in runs:
            runs[run_id] = {"prompt": None, "text_parts": [], "tool_calls": [],
                            "tool_results": [], "usage": None, "model": None,
                            "finish": None, "error": None, "timestamp": None}
            run_order.append(run_id)
        return runs[run_id]

    for record in _iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        if "retained_frame" in record or "retained_marker" in record:
            continue  # permission frames and live-only markers carry no content
        payload_type = record.get("payload_type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("stream", {}).get("id"):
            session_id = str(record["stream"]["id"])

        if payload_type == "runtime.session.metadata":
            inner = payload.get("record") or payload
            if isinstance(inner, dict) and inner.get("model_id"):
                model_info = {"model": inner.get("model_id"), "provider": inner.get("provider_id")}

        elif payload_type == "runtime.session":
            kind = payload.get("kind")
            event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
            event_kind = event.get("kind")
            run = run_slot(payload.get("run_id") or f"run-{len(run_order)}")

            if run["timestamp"] is None:
                run["timestamp"] = record.get("recorded_at")
            if kind == "run" and event_kind == "started":
                prompt = event.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    run["prompt"] = prompt
            elif kind == "run" and event_kind == "assistant_message_committed":
                text = event.get("text")
                if isinstance(text, str) and text.strip():
                    run["text_parts"].append(text)
            elif kind == "run" and event_kind == "assistant_tool_calls_committed":
                for call in event.get("tool_calls") or []:
                    if not isinstance(call, dict) or not call.get("name"):
                        continue
                    args = call.get("args")
                    if isinstance(args, str):
                        args = _safe_json(args)
                    run["tool_calls"].append({"id": call.get("id") or call.get("tool_call_id"), "name": str(call["name"]), "arguments": args if isinstance(args, dict) else {}})
            elif kind == "run" and event_kind == "tool_result_batch_committed":
                for result in event.get("results") or []:
                    if isinstance(result, dict) and result.get("text"):
                        run["tool_results"].append({"tool_call_id": result.get("tool_call_id"), "text": str(result["text"])})
            elif kind == "run" and event_kind == "model_completed":
                usage = event.get("usage") or {}
                run["usage"] = {"input_tokens": usage.get("input_tokens"),
                                "output_tokens": usage.get("output_tokens"), "basis": "recorded"}
                if event.get("model"):
                    run["model"] = str(event["model"])
                if event.get("finish_reason"):
                    run["finish"] = str(event["finish_reason"])

    for run_id in run_order:
        run = runs[run_id]
        messages = []
        if run["prompt"]:
            messages.append({"role": "user", "content": run["prompt"]})
        raw_calls.append(
            {
                "trace_id": trace_id,
                "call_id": f"{session_id}:{run_id}",
                "timestamp": run["timestamp"],
                "session_id": session_id,
                "call_index": len(raw_calls),
                "provider": (model_info or {}).get("provider") or "muse",
                "model": run["model"] or (model_info or {}).get("model") or "muse",
                "request": {"messages": messages},
                "response": {
                    "text": "\n\n".join(run["text_parts"]).strip(),
                    "tool_calls": run["tool_calls"],
                    "finish_reason": run["finish"] or ("tool_calls" if run["tool_calls"] else "stop"),
                },
                "usage": run["usage"],
            }
        )
        for result in run["tool_results"]:
            names = [call["name"] for call in run["tool_calls"] if call["name"]]
            raw_calls.append(_tool_result_call(
                trace_id, session_id, len(raw_calls),
                next((c["name"] for c in run["tool_calls"] if c.get("id") == result.get("tool_call_id")), names[0] if len(names) == 1 else "unknown_tool"), result["text"], result.get("tool_call_id"),
            ))
    return raw_calls


def import_muse_tree(path) -> list:
    """Walk a Muse sessions tree (or one session.jsonl), including subagent logs."""
    target = Path(path)
    if target.is_file() and target.name == "session.jsonl":
        return import_muse(target)
    if not target.exists():
        return []
    raw_calls = []
    for session_file in sorted(target.glob("**/session.jsonl")):
        raw_calls.extend(import_muse(session_file))
    return raw_calls
