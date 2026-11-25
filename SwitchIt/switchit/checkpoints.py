"""Run identity and checkpoint loading, preserving existing resume semantics."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

RUNNER_VERSION = "0.2.0"


def run_fingerprint(cases_doc: dict, baseline_spec: dict, target_spec: dict) -> str:
    """Bind the run to its cases, adapter specs, and runner version."""
    payload = {
        "runner_version": RUNNER_VERSION,
        "cases_digest": hashlib.sha256(
            json.dumps(cases_doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "baseline": baseline_spec,
        "target": target_spec,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_checkpoint(run_dir):
    path = Path(run_dir) / "run_state.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (ValueError, OSError):
        return None


def _checkpoint_path(run_dir) -> Path:
    return Path(run_dir) / "run_state.json"
