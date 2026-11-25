"""Orchestrate file importers, normalize records, and assemble a store."""

from __future__ import annotations

import json

from switchit.importers.files import IMPORTERS
from switchit.normalization import make_call
from switchit.storage import build_store


def import_store(inputs: dict, app_config_path=None, notes=None) -> dict:
    """Run the requested importers and assemble the canonical store.

    ``inputs`` maps importer kind ("jsonl", "otlp", "langfuse", "parquet") to a
    list of paths. Every call is imported exactly once, redacted at creation.
    """
    app_config = None
    if app_config_path:
        with open(app_config_path, "r", encoding="utf-8") as stream:
            app_config = json.load(stream)
    calls = []
    source_files = []
    sequence = 0
    for kind, paths in inputs.items():
        importer = IMPORTERS[kind]
        for path in paths:
            raw_calls = importer(path)
            for raw in raw_calls:
                raw["call_index"] = raw.get("call_index", sequence)
                sequence += 1
                calls.append(make_call(raw, kind, str(path)))
            source_files.append({"importer": kind, "path": str(path), "records": len(raw_calls)})
    calls.sort(key=lambda call: (call["trace_id"], call["call_index"], call["call_id"]))
    all_notes = [
        "All message content was redacted at import; redacted content is the only content stored, replayed, or reported.",
    ] + list(notes or [])
    if app_config and isinstance(app_config, dict):
        all_notes.append(
            "App configuration supplies prompts and tool definitions; it is descriptive, not evidence of behavior."
        )
    return build_store(calls, app_config=app_config, source_files=source_files, notes=all_notes)
