"""Persist workflow maps and preserve reviewer edits.

This is the existing review layer, not a new workflow specification builder."""

from __future__ import annotations

import json

from switchit.storage import atomic_write_json


def save_workflows(workflow_map: dict, path) -> None:
    atomic_write_json(path, workflow_map)


def load_workflows(path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def apply_edits(workflow_map: dict, previous: dict) -> dict:
    """Re-run grouping while preserving reviewer edits.

    A workflow the reviewer marked ``edited`` keeps its id, name, description,
    confidence, and explicit ``trace_ids``; those traces are removed from the
    automatically regenerated groups. Merges and splits are expressed by
    editing ``trace_ids`` of two workflows. Trace ids that no longer exist in
    the store are dropped.
    """
    if not previous:
        return workflow_map
    previous_edited = [wf for wf in previous.get("workflows", []) if wf.get("edited")]
    if not previous_edited:
        return workflow_map
    known_trace_ids = set(workflow_map.get("unassigned_trace_ids") or [])
    for workflow in workflow_map["workflows"]:
        known_trace_ids.update(workflow["trace_ids"])
    reserved_ids = set()
    preserved = []
    for workflow in previous_edited:
        trace_ids = [tid for tid in workflow.get("trace_ids", []) if tid in known_trace_ids]
        if not trace_ids:
            continue
        reserved_ids.update(trace_ids)
        preserved.append({**workflow, "trace_ids": sorted(trace_ids)})
    if not preserved:
        return workflow_map
    for workflow in workflow_map["workflows"]:
        workflow["trace_ids"] = [tid for tid in workflow["trace_ids"] if tid not in reserved_ids]
    workflow_map["workflows"] = [wf for wf in workflow_map["workflows"] if wf["trace_ids"]] + preserved
    workflow_map["unassigned_trace_ids"] = [
        tid for tid in workflow_map["unassigned_trace_ids"] if tid not in reserved_ids
    ]
    for position, workflow in enumerate(workflow_map["workflows"], start=1):
        workflow["id"] = f"wf_{position:03d}"
    workflow_map.setdefault("notes", []).append(
        "Reviewer edits were preserved from the previous workflow map on this run."
    )
    return workflow_map
