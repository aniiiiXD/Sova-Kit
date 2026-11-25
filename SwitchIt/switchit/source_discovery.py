"""Find supported local agent session files without parsing their contents."""

from __future__ import annotations

from pathlib import Path

_VSCODE_GLOBAL_STORAGE_CANDIDATES = [
    Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev",
    Path.home() / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev",
    Path.home() / ".cline" / "data",
]


AGENT_DEFAULTS = {
    "codex": [Path.home() / ".codex" / "sessions"],
    "claude-code": [Path.home() / ".claude" / "projects"],
    "opencode": [
        Path.home() / ".local" / "share" / "opencode" / "opencode.db",
        Path.home() / ".local" / "share" / "opencode" / "storage",
    ],
    "cursor": [Path.home() / ".cursor" / "chats"],
    "cline": _VSCODE_GLOBAL_STORAGE_CANDIDATES,
    "gemini-cli": [Path.home() / ".gemini" / "tmp"],
    "copilot-cli": [Path.home() / ".copilot" / "session-state"],
    "muse": [Path.home() / ".local" / "share" / "muse" / "sessions"],
    "grok-build": [Path.home() / ".grok" / "sessions"],
}


def discover_agent_sessions(kinds=None, limit: int = 200, *, defaults=None) -> list:
    """Find existing coding-agent session files on this machine.

    Returns a list of {"agent", "path"} entries, newest-ish first where the
    layout allows. Never raises for missing folders — an agent that is not
    installed simply contributes nothing.
    """
    defaults = AGENT_DEFAULTS if defaults is None else defaults
    kinds = list(kinds or defaults)
    sessions = []
    for kind in kinds:
        for base in defaults.get(kind, []):
            if not base.exists():
                continue
            if kind == "opencode":
                if base.suffix == ".db":
                    sessions.append({"agent": kind, "path": str(base)})
                continue
            if kind == "cursor":
                files = sorted(
                    (p for p in base.glob("*/*/store.db")),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
            elif kind == "claude-code":
                files = [p for p in base.glob("**/*.jsonl")
                         if not p.name.startswith("skill-")]
            elif kind == "codex":
                files = sorted(
                    (p for p in base.glob("**/*.jsonl*") if p.suffix in (".jsonl", ".zst")),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
            elif kind == "gemini-cli":
                files = sorted(base.glob("*/chats/session-*.jsonl"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            elif kind == "copilot-cli":
                files = sorted(base.glob("*/events.jsonl"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            elif kind == "muse":
                files = sorted(base.glob("**/session.jsonl"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            elif kind == "grok-build":
                files = sorted((d for d in base.glob("*/*") if d.is_dir() and ((d / "events.jsonl").exists() or (d / "chat_history.jsonl").exists())),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            elif kind == "cline":
                files = []
                for messages_file in sorted(base.glob("**/sessions/*/*.messages.json")):
                    files.append(messages_file)
                for task_dir in sorted(base.glob("**/tasks/*")):
                    if (task_dir / "api_conversation_history.json").exists():
                        files.append(task_dir)
            else:
                files = list(base.glob("**/*.jsonl"))
            if kind == "claude-code":
                files = [p for p in files if not p.name.startswith("skill-")]
            for file_path in files[: max(0, limit)]:
                sessions.append({"agent": kind, "path": str(file_path)})
    return sessions
