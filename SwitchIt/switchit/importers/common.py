"""Shared parsing helpers for agent transcripts; no source discovery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

REDACT_MARKER = "[REDACTED]"


def _parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _open_text(path: Path):
    """Open a text file, transparently decompressing .zst (Codex compresses
    older rollouts). Uses the stdlib zstd on Python 3.14+, else the optional
    `zstandard` package."""
    if path.suffix == ".zst":
        try:
            from compression import zstd  # Python 3.14+
        except ImportError:
            try:
                import zstandard

                return zstandard.open(path, "rt", encoding="utf-8")
            except ImportError as error:
                raise ImportError(
                    f"{path} is zstd-compressed; reading it needs Python 3.14+ "
                    "or the optional zstandard package (pip install zstandard)"
                ) from error
        return zstd.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(path: Path):
    try:
        with _open_text(path) as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _content_text(content) -> str:
    """Flatten Anthropic/OpenAI-style content blocks to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                for key in ("text", "thinking", "input_text", "output_text"):
                    if isinstance(block.get(key), str):
                        parts.append(block[key])
                        break
    return "\n".join(p for p in parts if p)


def _usage_recorded(usage) -> dict:
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "basis": "recorded"}


def _tool_result_call(trace_id, session_id, call_index, tool_name, result_text, tool_call_id=None):
    return {
        "trace_id": trace_id,
        "call_id": f"{session_id}:tool:{call_index}",
        "session_id": session_id,
        "call_index": call_index,
        "provider": "anthropic",
        "model": "tool-result",
        "request": {"messages": [{"role": "tool", "content": result_text,
                                  "name": tool_name or "unknown_tool",
                                  **({"tool_call_id": str(tool_call_id)} if tool_call_id else {})}]},
        "response": None,
        "usage": None,
        "usage_basis": None,
    }


def _safe_json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_raw": value[:400]}
        except ValueError:
            return {"_raw": value[:400]}
    return {}


def _gemini_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                response = item.get("functionResponse")
                if isinstance(response, dict):
                    inner = response.get("response")
                    if isinstance(inner, dict) and inner.get("output") is not None:
                        parts.append(str(inner["output"]))
                        continue
                parts.append(str(item.get("text") or item.get("output") or ""))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("output") or "")
    return ""


def attach_record_metadata(calls, record):
    """Preserve explicit source relationships; never infer a parent from a filename."""
    for call in calls:
        for field in ('timestamp', 'parent_id', 'parentId', 'parentUuid', 'agent_id',
                      'parent_agent_id', 'retry_of', 'handoff_from'):
            if record.get(field) is not None:
                call[field] = record[field]
        if record.get('uuid') or record.get('id'):
            call['source_record_id'] = record.get('uuid') or record['id']
