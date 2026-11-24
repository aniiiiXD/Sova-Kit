"""Dispatch explicit source paths to their agent-specific parsers."""

from __future__ import annotations

import sqlite3

from switchit.importers.claude_code import import_claude_code
from switchit.importers.cline import import_cline
from switchit.importers.codex import import_codex
from switchit.importers.copilot_cli import import_copilot_cli
from switchit.importers.cursor import import_cursor
from switchit.importers.gemini_cli import import_gemini_cli
from switchit.importers.grok_build import import_grok_build
from switchit.importers.muse import import_muse_tree
from switchit.importers.opencode import import_opencode

IMPORTERS_BY_AGENT = {
    "codex": import_codex,
    "claude-code": import_claude_code,
    "opencode": import_opencode,
    "cursor": import_cursor,
    "cline": import_cline,
    "gemini-cli": import_gemini_cli,
    "copilot-cli": import_copilot_cli,
    "muse": import_muse_tree,
    "grok-build": import_grok_build,
}


def import_agent_sessions(sessions: list, *, importers=None) -> list:
    """Run the per-agent importers over discovered or explicitly given sessions."""
    importers = IMPORTERS_BY_AGENT if importers is None else importers
    raw_calls = []
    for entry in sessions:
        kind = entry["agent"]
        importer = importers.get(kind)
        if importer is None:
            continue
        try:
            raw_calls.extend(importer(entry["path"]))
        except (OSError, sqlite3.Error) as error:
            # One unreadable session must not sink the import.
            print(f"switchit: skipping unreadable {kind} session {entry['path']}: {error}",
                  file=__import__("sys").stderr)
    return raw_calls
