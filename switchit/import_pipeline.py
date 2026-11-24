"""Audited source ingestion into revisioned events; no network or runtime execution."""
from pathlib import Path
import json
import sqlite3

from switchit.contracts import fingerprint, require, version
from switchit.event_store import empty_event_store, merge_events
from switchit.events import events_from_calls, normalize_event
from switchit.importers.common import _open_text
from switchit.importers.registry import IMPORTERS_BY_AGENT
from switchit.normalization import make_call
from switchit.source_inventory import source_digest


def _audit_jsonl(path):
    issues = []
    files = [path] if path.is_file() else sorted(path.rglob('*.jsonl'))
    for file in files:
        if file.suffix not in ('.jsonl', '.zst'):
            continue
        with _open_text(file) as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError('not an object')
                except ValueError:
                    issues.append({'file': str(file), 'line': number, 'reason': 'invalid_json_record'})
    return issues


def import_inventory(inventory, store=None, importers=None):
    version(inventory, 'inventory_version')
    store = store if store is not None else empty_event_store()
    importers = IMPORTERS_BY_AGENT if importers is None else importers
    all_events, source_updates, diagnostics = [], {}, []
    for source in inventory['sources']:
        sid = source['source_id']
        if source['status'] not in ('ready', 'unchanged'):
            diagnostics.append({'source_id': sid, 'status': source['status']})
            continue
        if store.get('sources', {}).get(sid, {}).get('digest') == source.get('digest'):
            diagnostics.append({'source_id': sid, 'status': 'unchanged'})
            continue
        parser = importers.get(source['agent'])
        if parser is None:
            diagnostics.append({'source_id': sid, 'status': 'unsupported'})
            continue
        try:
            path = Path(source['path'])
            require(source_digest(path) == source['digest'], 'source changed since discovery; rescan')
            issues = _audit_jsonl(path)
            raw_calls = parser(path)
            require(raw_calls, "source has no recognized records; check its schema")
            require(source_digest(path) == source['digest'], 'source changed during import; rescan')
            calls = [make_call(raw, source['agent'], str(path)) for raw in raw_calls]
            events = events_from_calls(calls)
            # Pin a complete source revision. Old revisions remain available for audit.
            mapping = {e['event_id']: fingerprint([sid, source['digest'], e['event_id']])[:32] for e in events}
            for event in events:
                event['event_id'] = mapping[event['event_id']]
                for field in ('parent_id', 'retry_of', 'handoff_from'):
                    event[field] = mapping.get(event.get(field), event.get(field))
                event['source'].update(source_id=sid, revision=source['digest'])
                if issues:
                    event['uncertainty'].append('source_has_invalid_records')
                all_events.append(normalize_event(event))
            source_updates[sid] = {'digest': source['digest'], 'event_ids': [mapping[k] for k in mapping],
                                   'agent': source['agent'], 'issues': issues}
            diagnostics.append({'source_id': sid, 'status': 'partial' if issues else 'imported',
                                'events': len(events), 'issues': issues})
        except (OSError, ValueError, ImportError, sqlite3.Error) as error:
            diagnostics.append({'source_id': sid, 'status': 'error', 'error': type(error).__name__,
                                'message': str(error)[:300]})
    return {'store': merge_events(store, all_events, source_updates), 'diagnostics': diagnostics}


def active_events(store):
    version(store, 'event_store_version')
    ids = {eid for source in store['sources'].values() for eid in source.get('event_ids', [])}
    return [e for e in store['events'] if not e['source'].get('source_id') or e['event_id'] in ids]
