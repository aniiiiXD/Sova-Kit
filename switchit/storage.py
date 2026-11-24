"""Canonical store assembly and local JSON persistence."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import tempfile

from switchit.hierarchy import build_sessions, build_traces
from switchit.redaction import redact_value
from switchit.usage import merge_usage

STORE_VERSION = 1


def build_store(calls: list, app_config=None, source_files=None, notes=None,
                session_meta: dict | None = None) -> dict:
    """Assemble the canonical store from canonical calls.

    ``session_meta`` optionally maps trace_id to {"title", "origin", "cwd"}
    (e.g. Codex's cloud/local thread catalog) and is merged into the traces.
    """
    traces = build_traces(calls, session_meta=session_meta)
    return {
        "store_version": STORE_VERSION,
        "app_config": redact_value(app_config) if app_config else None,
        "sessions": build_sessions(traces),
        "traces": traces,
        "calls": calls,
        "totals": {
            "traces": len(traces),
            "calls": len(calls),
            "usage": merge_usage(*[trace["usage"] for trace in traces]),
        },
        "source_files": list(source_files or []),
        "notes": list(notes or []),
    }


def atomic_write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=1, sort_keys=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_store(store: dict, path) -> None:
    atomic_write_json(path, store)


def load_store(path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def digest_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_data(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
