"""Parse aider session records into raw calls."""

from __future__ import annotations

from pathlib import Path
import re

from switchit.importers.common import _tool_result_call

AIDER_HEADER_PATTERN = re.compile(r"^# aider chat started at (.+)$")


def import_aider(path) -> list:
    """Parse ``.aider.chat.history.md`` — one file holds every session.

    Documented in aider's source (aider/io.py, aider/utils.py): ``# aider chat
    started at`` headers date sessions, ``#### `` lines are user prompts,
    ``> `` blockquotes are tool/system output, bare markdown is the assistant.
    No session ids exist, so one is synthesised per header. No usage, no
    structured tool calls — this is task-level evidence, honestly weaker.
    """
    path = Path(path)
    raw_calls = []
    session_stamp = None
    call_index = 0
    assistant_lines = []
    tool_lines = []

    def flush_assistant():
        nonlocal assistant_lines, call_index
        if session_stamp is not None and assistant_lines:
            raw_calls.append(
                {
                    "trace_id": f"aider:{path}#{session_stamp}",
                    "call_id": f"{session_stamp}:a:{call_index}",
                    "session_id": session_stamp,
                    "call_index": call_index,
                    "provider": "aider",
                    "model": None,
                    "request": {"messages": []},
                    "response": {"text": "\n".join(assistant_lines).strip(),
                                 "tool_calls": [], "finish_reason": "stop"},
                    "usage": None,
                }
            )
            call_index += 1
            assistant_lines = []

    def flush_tool():
        nonlocal tool_lines, call_index
        if session_stamp is not None and tool_lines:
            raw_calls.append(_tool_result_call(
                f"aider:{path}#{session_stamp}", session_stamp, call_index,
                None, "\n".join(tool_lines).strip(),
            ))
            call_index += 1
            tool_lines = []

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        header = AIDER_HEADER_PATTERN.match(line.strip())
        if header:
            flush_assistant()
            flush_tool()
            session_stamp = header.group(1).strip()
            call_index = 0
        elif line.startswith("#### "):
            flush_assistant()
            flush_tool()
            if session_stamp is not None:
                prompt = line[5:].rstrip().rstrip("\\").strip()
                if prompt:
                    raw_calls.append(
                        {
                            "trace_id": f"aider:{path}#{session_stamp}",
                            "call_id": f"{session_stamp}:u:{call_index}",
                            "session_id": session_stamp,
                            "call_index": call_index,
                            "provider": "aider",
                            "model": "user-turn",
                            "request": {"messages": [{"role": "user", "content": prompt}]},
                            "response": None,
                            "usage": None,
                        }
                    )
                    call_index += 1
        elif line.startswith("> "):
            flush_assistant()
            tool_lines.append(line[2:].rstrip())
        elif line.strip():
            flush_tool()
            assistant_lines.append(line)
    flush_assistant()
    flush_tool()
    return raw_calls
