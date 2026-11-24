"""Read session catalog metadata and create discovery-only task records."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from switchit.importers.common import _iter_jsonl


def codex_thread_catalog(db_path=None) -> dict:
    """Read the ChatGPT-desktop thread catalog (thread_id -> title, kind, cwd).

    Codex writes this for every thread the desktop app knows about;
    ``source_kind`` distinguishes local ``codex-cli`` threads from ``chatgpt``
    cloud conversations (whose transcripts stay server-side). Read-only; WAL
    sidecars are read as-is. Missing database means no metadata.
    """
    db_path = Path(db_path or Path.home() / ".codex" / "sqlite" / "codex-dev.db")
    if not db_path.exists():
        return {}
    catalog = {}
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT thread_id, display_title, source_kind, cwd FROM local_thread_catalog"
        )
        for thread_id, title, source_kind, cwd in rows:
            catalog[str(thread_id)] = {
                "title": title,
                "source_kind": source_kind,
                "cwd": cwd,
                "origin": "cloud (ChatGPT)" if "chatgpt" in str(source_kind) else "local",
            }
        connection.close()
    except sqlite3.Error:
        return {}
    return catalog


def catalog_stub_calls(catalog: dict, existing_trace_ids=None) -> list:
    """Title-level task traces for catalog threads with no local transcript.

    ChatGPT-cloud conversations (and Dot sessions that never spawned a local
    Codex rollout) exist server-side; only their titles are known locally.
    These stubs surface them in the workflow map as task-level evidence —
    same honesty rules as Devin: no transcript, no replay, discovery only.
    """
    existing_trace_ids = set(existing_trace_ids or [])
    raw_calls = []
    for thread_id, info in catalog.items():
        # A local rollout of the same thread carries the full transcript,
        # whichever prefix it was imported under (codex:, chatgpt:, ...).
        if any(f"{prefix}:{thread_id}" in existing_trace_ids for prefix in ("codex", "chatgpt", "cli", "vscode")):
            continue
        title = info.get("title") or ""
        if not title:
            continue
        trace_id = f"chatgpt:{thread_id}"
        raw_calls.append(
            {
                "trace_id": trace_id,
                "call_id": f"{thread_id}:task",
                "session_id": thread_id,
                "call_index": 0,
                "provider": "chatgpt-cloud",
                "model": "cloud-thread",
                "request": {"messages": [{"role": "user", "content": f"[cloud thread] {title}"}]},
                "response": {"text": "[session outcome] transcript stored server-side by OpenAI; only the title is available locally.",
                             "tool_calls": [], "finish_reason": "stop"},
                "usage": None,
                "origin": info.get("origin") or "cloud (ChatGPT)",
            }
        )
    return raw_calls


def collect_session_meta(sessions: list, catalog_db=None) -> dict:
    """Gather per-trace session metadata (title, origin, cwd) for imported sessions.

    Returns {trace_id: {"title", "origin", "cwd"}} — currently only Codex
    carries a real cloud/local distinction; every other agent's on-disk
    sessions are simply local. ``catalog_db`` overrides the Codex thread
    catalog location (used by tests; production reads ~/.codex).
    """
    meta = {}
    codex_catalog = None
    for entry in sessions:
        agent = entry.get("agent")
        if agent == "codex":
            if codex_catalog is None:
                codex_catalog = codex_thread_catalog(catalog_db)
            path = Path(entry["path"])
            # Rollout files embed their session id; read it cheaply.
            for record in _iter_jsonl(path):
                if isinstance(record, dict) and record.get("type") == "session_meta":
                    payload = record.get("payload") or {}
                    session_id = payload.get("session_id") or payload.get("id")
                    if session_id and str(session_id) in codex_catalog:
                        info = codex_catalog[str(session_id)]
                        meta[f"codex:{session_id}"] = {
                            "title": info["title"],
                            "origin": info["origin"],
                            "cwd": info["cwd"],
                        }
                    break
        else:
            source_path = Path(entry["path"])
            trace_hint = f"{agent}:"
            # Origin is local by construction: the transcript is a local file.
            meta.setdefault(trace_hint + source_path.name, {"origin": "local"})
    return meta
