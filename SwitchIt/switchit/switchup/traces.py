"""Public trace import and holdout APIs for SwitchUP."""
from switchit.contracts import require
from switchit.events import events_from_calls
from switchit.import_pipeline import active_events
from switchit.workflow_commands import read_json

from .capture import chat_suite, invoke
from .store import Workspace

__all__ = ["Workspace", "read_events", "ingest_events", "pin_chat_suite", "invoke"]


def read_events(path):
    """Read a canonical event list, a SwitchIt event store, or normalized calls."""
    document = read_json(path)
    if isinstance(document, list):
        return document
    require(isinstance(document, dict), "provide an events array or a SwitchIt evidence object")
    if "event_store_version" in document:
        return active_events(document)
    if "calls" in document:
        return events_from_calls(document["calls"])
    return document.get("events")


def ingest_events(workspace, project, events):
    """Store redacted evidence in batches and return added and duplicate counts."""
    require(isinstance(events, list) and events, "provide a nonempty array of evidence events")
    totals = {"added": 0, "duplicates": 0}
    for offset in range(0, len(events), 3000):
        counts = workspace.ingest(project, events[offset:offset + 3000])
        for key in totals:
            totals[key] += counts[key]
    return totals


def pin_chat_suite(workspace, project, holdout_traces, *, expected_generation, max_cases=30):
    """Pin replay cases built from complete requests in separate sessions."""
    suite = chat_suite(workspace, workspace.project(project), holdout_traces, max_cases=max_cases)
    return workspace.install_suite(project, suite, expected_generation=expected_generation)
