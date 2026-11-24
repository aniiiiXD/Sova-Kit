"""Incremental immutable evidence ingestion; conflicting IDs never overwrite history."""
from copy import deepcopy
from pathlib import Path
import json

from switchit.contracts import ContractError, fingerprint, require, version
from switchit.events import normalize_event, validate_event
from switchit.storage import atomic_write_json


def empty_event_store():
    return {'event_store_version': 1, 'revision': 0, 'events': [], 'sources': {}}


def merge_events(store, events, sources=None):
    version(store, 'event_store_version')
    result = deepcopy(store)
    existing = {}
    for event in result['events']:
        validate_event(event)
        require(event['event_id'] not in existing, 'duplicate event in stored evidence')
        existing[event['event_id']] = event
    added = 0
    for raw in events:
        event = normalize_event(raw)
        key = event['event_id']
        if key in existing:
            if existing[key] != event:
                raise ContractError(f'conflicting event ID {key}; retain both source revisions with distinct IDs')
            continue
        existing[key] = event
        added += 1
    result['events'] = list(existing.values())
    result['sources'].update(deepcopy(sources or {}))
    if added or result['sources'] != store['sources']:
        result['revision'] += 1
    result['digest'] = fingerprint(result['events'])
    return result


def ingest_events(path, events, sources=None):
    """Single-writer local transaction: validate the entire batch before replacing disk."""
    path = Path(path)
    store = json.loads(path.read_text()) if path.exists() else empty_event_store()
    result = merge_events(store, events, sources)
    atomic_write_json(path, result)
    return result
