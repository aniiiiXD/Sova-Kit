"""Aegis task bundles and daemon snapshots → observed, non-replayable evidence."""
from switchit.contracts import fingerprint, require
from switchit.events import normalize_event


LIMITATIONS = [
    'Aegis messages and command observations do not include complete provider requests or tool results; exact replay is unavailable.',
    'Verification results and review findings are historical observations, not fresh checks or SwitchIt approvals.',
    'Working profiles are advisory context, not executable workflow routing.',
    'Raw terminal transport and hidden provider state are not collected.',
]


def _string(row, key):
    value = row.get(key)
    require(isinstance(value, str) and bool(value.strip()), f'Aegis {key} must be a nonempty string')
    return value


def _rows(document, key):
    rows = document.get(key, [])
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            f'Aegis {key} must be an object array')
    seen = set()
    for row in rows:
        identity = _string(row, 'id')
        require(identity not in seen, f'duplicate Aegis {key} identity')
        seen.add(identity)
    return rows


def import_aegis(document):
    require(isinstance(document, dict), 'Aegis document must be an object')
    require(type(document.get('version')) is int and document['version'] == 1,
            'unsupported Aegis version')
    snapshot = document.get('format') == 'aegis-snapshot'
    require(snapshot or document.get('format') == 'aegis-task', 'unsupported Aegis format')
    bundle = document.get('bundle') if snapshot else document
    require(isinstance(bundle, dict) and bundle.get('format') == 'aegis-task'
            and type(bundle.get('version')) is int and bundle['version'] == 1, 'invalid Aegis task bundle')
    origin, task_id = _string(bundle, 'source_origin'), _string(bundle, 'source_task_id')
    require(isinstance(bundle.get('prompt'), str) and isinstance(bundle.get('source_repository'), str),
            'Aegis prompt and source_repository must be strings')
    omissions = bundle.get('omissions', [])
    require(isinstance(omissions, list) and all(isinstance(v, str) for v in omissions), 'invalid Aegis omissions')
    coverage = document.get('coverage', []) if snapshot else []
    require(isinstance(coverage, list) and all(isinstance(c, dict) and
            c.get('status') in ('complete', 'partial', 'limited', 'bounded') for c in coverage), 'invalid Aegis coverage')
    runs = _rows(document, 'runs') if snapshot else []
    messages = _rows(document, 'events') if snapshot else _rows(bundle, 'messages')
    verifications = _rows(document if snapshot else bundle, 'verifications')
    findings = _rows(bundle, 'findings')
    source_id = fingerprint(['aegis', origin, task_id, document['format']])[:24]
    revision = fingerprint(document)
    task_trace = 'aegis:' + fingerprint([origin, task_id])[:24]
    events, diagnostics, identities = [], [], {}

    def emit(category, identity, row, *, kind='state', parent=None, trace=None, index=0, uncertainty=()):
        record_id = fingerprint([origin, task_id, category, identity])[:32]
        eid = fingerprint([source_id, revision, record_id])[:32]
        identities[(category, identity)] = eid
        event = normalize_event({
            'event_id': eid, 'record_id': record_id, 'trace_id': trace or task_trace,
            'session_id': task_trace, 'index': index, 'kind': kind,
            'timestamp': row.get('timestamp', row.get('started_at', row.get('created_at'))),
            'parent_id': parent, 'agent_id': row.get('run_id'),
            'source': {'importer': 'aegis', 'source_id': source_id, 'revision': revision,
                       'record_type': category, 'format': document['format'],
                       'repository': bundle['source_repository']},
            'payload': {'aegis_type': category, **row}, 'evidence': 'metadata_only',
            'uncertainty': ['exact_replay_unavailable', *uncertainty],
        })
        events.append(event)
        return eid

    root = emit('task', task_id, {'id': task_id, 'text': bundle['prompt'],
        'context': bundle.get('context', {}), 'profile': bundle.get('profile', bundle.get('working_profile', {})),
        'omissions': omissions, 'coverage': coverage, 'limitations': LIMITATIONS},
        kind='user_input', uncertainty=['task_bundle_is_bounded'])
    run_ids = set()
    for run in runs:
        require(run.get('task_id') == task_id, 'Aegis run belongs to another task')
        rid = run['id']; run_ids.add(rid)
        emit('run', rid, run, kind='agent_start', parent=root,
             trace=task_trace + ':run:' + fingerprint(rid)[:16])

    def location(row):
        require(row.get('task_id', task_id) == task_id, 'Aegis record belongs to another task')
        rid = row.get('run_id')
        require(rid is None or isinstance(rid, str), 'Aegis run_id must be a string or null')
        parent = identities.get(('run', rid), root)
        missing = bool(rid and rid not in run_ids)
        if missing:
            diagnostics.append({'record_id': row['id'], 'reason': 'run_not_imported', 'run_id': rid})
        return parent, (task_trace + ':run:' + fingerprint(rid)[:16] if rid else task_trace), missing

    for row in messages:
        require(isinstance(row.get('type'), str) and isinstance(row.get('content'), str),
                'Aegis event requires type and content strings')
        sequence = row.get('sequence')
        require(type(sequence) is int and sequence >= 0, 'Aegis event requires a nonnegative sequence')
    # Only persisted sequence is used for event ordering, never API page order.
    for row in sorted(messages, key=lambda r: (r['sequence'], r['id'])):
        parent, trace, missing = location(row)
        uncertainty = ['provider_request_and_tool_results_missing']
        if missing:
            uncertainty.append('run_not_imported')
        emit('event', row['id'], row, parent=parent, trace=trace, index=row['sequence'] + 1,
             kind='user_input' if row['type'] == 'user.message' else 'state', uncertainty=uncertainty)
    for category, rows in (('verification', verifications), ('finding', findings)):
        for row in rows:
            parent, _, missing = location(row)
            if category == 'verification':
                require(isinstance(row.get('command'), list) and row['command']
                        and all(isinstance(arg, str) for arg in row['command']), 'Aegis verification requires command argv')
                require(type(row.get('exit_code')) is int and isinstance(row.get('output'), str),
                        'Aegis verification requires exit_code and output')
            emit(category, row['id'], {**row, 'observed_freshness': row.get('freshness'), 'freshness': 'historical'},
                 parent=parent, trace=task_trace + ':' + category + ':' + fingerprint(row['id'])[:16],
                 uncertainty=['historical_observation', *(['run_not_imported'] if missing else [])])
    return {'events': events, 'source_id': source_id, 'revision': revision, 'coverage': coverage,
            'diagnostics': diagnostics, 'omissions': omissions, 'limitations': LIMITATIONS,
            'replay_ready': False}
