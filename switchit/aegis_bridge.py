"""Local JSON bridge between Aegis tasks and SwitchIt evidence/workflow storage.

The daemon supplies --store (and optional --sources) as trusted startup settings.
One request is read from stdin or --request-file; one versioned response is written
on stdout. No provider calls or agent commands are executed by this bridge.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from .contracts import fingerprint, require
from .event_store import empty_event_store, merge_events
from .events import normalize_event
from .execution_graph import build_execution_graph
from .import_pipeline import active_events, import_inventory
from .importers.aegis import import_aegis, LIMITATIONS
from .redaction import redact_value
from .storage import atomic_write_json

BRIDGE_VERSION = 1
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_EVENTS = 3000
ARTIFACTS = frozenset({'store', 'graph', 'candidates', 'drafts', 'metadata'})


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON object field')
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError('non-finite JSON numbers are not supported')
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        require(depth <= 50, 'JSON nesting exceeds 50 levels')
        if isinstance(current, dict):
            pending.extend((v, depth + 1) for v in current.values())
        elif isinstance(current, list):
            pending.extend((v, depth + 1) for v in current)
    return value


def _read(path):
    return _json(Path(path).read_text(encoding='utf-8'))


def _identity(value, name='task_id'):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 200,
            name + ' must be a nonempty string of at most 200 characters')
    return value


class EvidenceBridge:
    def __init__(self, store, sources=None):
        configured_root = Path(store).expanduser().absolute()
        require(not configured_root.is_symlink(), 'store root must not be a symlink')
        self.root = configured_root.resolve()
        self.sources = deepcopy(sources or {})
        require(isinstance(self.sources, dict), 'sources configuration must be an object')

    def _checked(self, path):
        """Reject linked components before every store read/write, including new paths."""
        path = Path(path)
        require(path.is_relative_to(self.root), 'storage path escapes configured root')
        current = self.root
        require(not current.is_symlink(), 'store root must not be a symlink')
        for part in path.relative_to(self.root).parts:
            require(part not in ('.', '..'), 'invalid storage path')
            current = current / part
            require(not current.is_symlink(), 'store paths must not contain symlinks')
        require(path.resolve().is_relative_to(self.root), 'storage path escapes configured root')
        if path.is_file():
            require(path.stat().st_nlink == 1, 'store files must not be hard-linked')
        return path

    def _read_store(self, path):
        return _read(self._checked(path))

    def _write_store(self, path, value):
        atomic_write_json(self._checked(path), value)

    @contextmanager
    def _lock(self):
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._checked(self.root / '.bridge.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _task_dir(self, task_id, exists=True):
        _identity(task_id)
        path = self._checked(self.root / 'tasks' / fingerprint(task_id)[:32])
        require(path.resolve().is_relative_to(self.root), 'task storage escapes configured root')
        if exists:
            require(self._checked(path / 'metadata.json').is_file(), 'task evidence has not been captured')
        return path

    def _source_list(self):
        result = []
        for sid, source in sorted(self.sources.items()):
            require(isinstance(source, dict), 'invalid configured source')
            result.append({'source_id': sid, 'agent': source.get('agent'),
                           'label': source.get('label', sid)})
        return result

    def _store(self, folder):
        path = self._checked(folder / 'store.json')
        return self._read_store(path) if path.exists() else empty_event_store()

    def _workflow_docs(self, folder):
        index = self._checked(folder / 'workflows.json')
        return self._read_store(index) if index.exists() else {'workflows': []}

    def _summary(self, folder):
        metadata = self._read_store(folder / 'metadata.json')
        store = self._store(folder)
        events = active_events(store)
        native = [e for e in events if e['evidence'] != 'metadata_only']
        workflows = self._workflow_docs(folder)['workflows']
        return {'task_id': metadata['task_id'], 'prompt': metadata['prompt'],
                'run_count': len(metadata['run_ids']), 'event_count': len(events),
                'native_event_count': len(native),
                'evidence_status': 'trace_attached' if native else 'evidence_only',
                'workflow_count': len(workflows),
                'reviewed_count': sum(w['workflow']['status'] == 'reviewed' for w in workflows),
                'updated_at': metadata['updated_at'], 'revision': store.get('digest', fingerprint([]))}

    def _detail(self, folder):
        summary = self._summary(folder)
        store = self._store(folder)
        events = active_events(store)
        graph = build_execution_graph(events)
        visible = {e['event_id'] for e in graph['nodes'][:200]}
        graph = {**graph, 'nodes': graph['nodes'][:200],
                 'edges': [edge for edge in graph['edges'] if edge['source'] in visible and edge['target'] in visible],
                 'order': [eid for eid in graph['order'] if eid in visible],
                 'issues': graph['issues'][:200],
                 'truncated': len(events) > 200}
        entries = []
        for entry in self._workflow_docs(folder)['workflows']:
            spec = entry['workflow']
            evaluation = entry.get('evaluation')
            counts = {key: 0 for key in ('pass', 'fail', 'error', 'uncertain')}
            if evaluation:
                for case in evaluation['cases']:
                    counts[case['evaluation']['verdict']] += 1
            entries.append({'workflow_id': spec['workflow_id'], 'name': spec['name'],
                            'status': spec['status'], 'revision': spec['revision'],
                            'evidence_revision': entry['evidence_revision'],
                            'stale': entry['evidence_revision'] != summary['revision'],
                            'case_count': len(entry.get('cases', {}).get('cases', [])),
                            'verdicts': counts, 'exported': entry.get('exported', False),
                            'workflow': spec, 'evaluation': evaluation})
        return {'task': summary, 'events': events[:200], 'graph': graph,
                'workflows': entries, 'sources': self._source_list(),
                'limitations': LIMITATIONS + ['Fixture results do not establish live cross-agent portability.']}

    def _touch(self, folder):
        metadata = self._read_store(folder / 'metadata.json')
        metadata['updated_at'] = int(time.time() * 1000)
        self._write_store(folder / 'metadata.json', metadata)

    def _merge(self, folder, events, sources):
        store = merge_events(self._store(folder), events, sources)
        require(len(active_events(store)) <= MAX_EVENTS, 'task active evidence exceeds event limit')
        build_execution_graph(active_events(store))  # Validate the entire transaction before disk replacement.
        self._write_store(folder / 'store.json', store)
        return store

    def capture(self, request):
        snapshot = request.get('snapshot', request.get('document'))
        imported = import_aegis(snapshot)
        require(len(imported['events']) <= MAX_EVENTS, 'Aegis evidence exceeds event limit')
        bundle = snapshot['bundle'] if snapshot['format'] == 'aegis-snapshot' else snapshot
        task_id = bundle['source_task_id']
        folder = self._task_dir(task_id, exists=False)
        previous = self._read_store(folder / 'metadata.json') if (folder / 'metadata.json').exists() else None
        if previous:
            require(previous['source_origin'] == bundle['source_origin'] and
                    previous['source_repository'] == bundle['source_repository'], 'task source identity changed')
        # Save an immutable redacted source revision before updating the active pointer.
        revision_file = self._checked(folder / 'captures' / (imported['revision'] + '.json'))
        if not revision_file.exists():
            self._write_store(revision_file, redact_value(snapshot))
        self._merge(folder, imported['events'], {imported['source_id']: {
            'digest': imported['revision'], 'event_ids': [e['event_id'] for e in imported['events']],
            'agent': 'aegis', 'issues': imported['diagnostics']}})
        run_ids = {r['id'] for r in snapshot.get('runs', [])}
        raw_rows = snapshot.get('events', bundle.get('messages', []))
        run_ids.update(row['run_id'] for row in raw_rows if row.get('run_id'))
        metadata = {'bridge_task_version': 1, 'task_id': task_id,
                    'source_origin': bundle['source_origin'], 'source_repository': bundle['source_repository'],
                    'prompt': redact_value(bundle['prompt']), 'run_ids': sorted(run_ids),
                    'updated_at': int(time.time() * 1000), 'capture_revision': imported['revision'],
                    'coverage': imported['coverage'], 'diagnostics': imported['diagnostics']}
        self._write_store(folder / 'metadata.json', metadata)
        return self._detail(folder)

    def attach_trace(self, request):
        folder = self._task_dir(request.get('task_id'))
        run_id = request.get('run_id')
        if run_id is not None:
            _identity(run_id, 'run_id')
            require(run_id in self._read_store(folder / 'metadata.json')['run_ids'], 'run_id is not part of the captured task')
        if request.get('source_id') is not None:
            sid = _identity(request['source_id'], 'source_id')
            require(sid in self.sources, 'trace source is not configured')
            config = self.sources[sid]
            require(isinstance(config, dict) and config.get('agent') in ('codex', 'claude-code', 'cursor'),
                    'configured trace importer must be codex, claude-code or cursor')
            path = Path(config['path']).expanduser().resolve()
            require(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024,
                    'configured source must be a session file no larger than 32 MiB')
            from .source_inventory import source_digest
            source_key = fingerprint(['aegis-bridge-native', sid, config['agent'], str(path), run_id])[:24]
            inventory = {'inventory_version': 1, 'sources': [{'source_id': source_key,
                'path': str(path), 'agent': config['agent'], 'status': 'ready', 'digest': source_digest(path)}]}
            result = import_inventory(inventory)
            require(result['store']['events'], 'configured source has no recognized native trace records')
            events = active_events(result['store'])
            source_updates = result['store']['sources']
        else:
            supplied = request.get('events')
            require(isinstance(supplied, list) and 0 < len(supplied) <= MAX_EVENTS,
                    'supply a nonempty canonical events array or a configured source_id')
            events = [normalize_event(event) for event in supplied]
            sid = fingerprint(['aegis-bridge-upload', request.get('attachment_id', run_id or 'task')])[:24]
            revision = fingerprint(events)
            mapping = {e['event_id']: fingerprint([sid, revision, e['event_id']])[:32] for e in events}
            require(len(mapping) == len(events), 'duplicate uploaded event IDs')
            for event in events:
                event['event_id'] = mapping[event['event_id']]
                for field in ('parent_id', 'retry_of', 'handoff_from'):
                    event[field] = mapping.get(event.get(field), event.get(field))
                # Preserve the claimed source/evidence, while making upload provenance explicit.
                event['source'].update(source_id=sid, revision=revision, bridge_attachment='caller_supplied')
                event['uncertainty'] = sorted(set(event['uncertainty'] + ['caller_supplied_trace']))
            source_updates = {sid: {'digest': revision, 'event_ids': [e['event_id'] for e in events],
                                    'agent': 'caller-supplied', 'issues': []}}
        require(len(events) <= MAX_EVENTS, 'native trace exceeds event limit')
        if run_id:
            for event in events:
                event['source']['aegis_run_id'] = run_id
        # Source revisions are namespaced; uploaded native events cannot replace Aegis metadata.
        self._merge(folder, events, source_updates)
        self._touch(folder)
        return self._detail(folder)

    def build_workflow(self, request):
        from .workflow_builder import build_workflow
        from .workflow_patterns import discover_workflows
        folder = self._task_dir(request.get('task_id'))
        events = active_events(self._store(folder))
        require(any(e['kind'] == 'model_call' and e['evidence'] != 'metadata_only' for e in events),
                'native model/tool trace evidence is required; Aegis activity alone is not executable')
        # Discovery operates only on attached native evidence. Aegis context remains in the task graph.
        graph = build_execution_graph([e for e in events if e['evidence'] != 'metadata_only'])
        candidates = discover_workflows(graph)
        revision = self._summary(folder)['revision']
        entries, diagnostics = [], []
        for candidate in candidates['candidates']:
            try:
                spec = build_workflow(candidate, graph)
                require(len(spec['steps']) <= 100, 'candidate exceeds 100-step limit')
                entries.append({'workflow': spec, 'evidence_revision': revision})
            except ValueError as error:
                diagnostics.append({'candidate_id': candidate['candidate_id'], 'message': str(error)})
        require(entries, 'no executable workflow candidates could be reconstructed')
        self._write_store(folder / 'graph.json', graph)
        self._write_store(folder / 'candidates.json', candidates)
        self._write_store(folder / 'drafts.json', {'workflows': [e['workflow'] for e in entries], 'diagnostics': diagnostics})
        previous = self._workflow_docs(folder)['workflows']
        # Keep reviewed contracts as historical revisions when rebuilding newer evidence.
        reviewed = [e for e in previous if e['workflow']['status'] == 'reviewed']
        reviewed_current = {(e['workflow']['workflow_id'], e['evidence_revision']) for e in reviewed}
        # Rebuilding identical evidence must not demote an existing reviewed contract
        # or create a draft with the same workflow ID that shadows it during export.
        self._write_store(folder / 'workflows.json', {'workflows': reviewed + [e for e in entries
            if (e['workflow']['workflow_id'], e['evidence_revision']) not in reviewed_current]})
        self._touch(folder)
        return self._detail(folder)

    def _select(self, folder, request):
        doc = self._workflow_docs(folder)
        workflow_id = _identity(request.get('workflow_id'), 'workflow_id')
        matches = [e for e in doc['workflows'] if e['workflow']['workflow_id'] == workflow_id
                   and (not request.get('revision') or e['workflow']['revision'] == request['revision'])]
        # Without a revision select the latest saved instance, never a random historical contract.
        require(matches, 'workflow is not stored for this task')
        return doc, matches[-1]

    def review_workflow(self, request):
        from .workflow_builder import review_workflow
        from .workflow_cases import build_workflow_cases
        from .workflow_commands import execute_case
        folder = self._task_dir(request.get('task_id'))
        doc, entry = self._select(folder, request)
        require(entry['evidence_revision'] == self._summary(folder)['revision'],
                'workflow evidence changed; rebuild the draft before review')
        reviewer = _identity(request.get('reviewer'), 'reviewer')
        checks = request.get('checks')
        require(isinstance(checks, list) and checks and len(checks) <= 30, 'checks must be a nonempty array of at most 30 checks')
        allowed = {'no_error', 'output_equals', 'output_contains', 'output_min_chars', 'tool_called',
                   'tool_args_include', 'tool_calls_equal', 'tool_sequence', 'allowed_tools', 'step_before',
                   'state_equals', 'artifact_equals', 'artifact_exists', 'latency_ms_max', 'usage_input_max',
                   'usage_output_max', 'cost_max'}
        require(all(isinstance(c, dict) and c.get('type') in allowed for c in checks), 'unsupported outcome check')
        bindings = request.get('bindings', {})
        require(isinstance(bindings, dict) and set(bindings) <= {'inputs', 'outputs', 'steps', 'allowed_tools'}, 'unsupported workflow binding')
        spec = review_workflow(entry['workflow'], reviewer=reviewer, success_checks=checks, **bindings)
        graph, candidates = self._read_store(folder / 'graph.json'), self._read_store(folder / 'candidates.json')
        candidate = next((c for c in candidates['candidates'] if c['candidate_id'] == spec['workflow_id']), None)
        require(candidate is not None, 'workflow candidate is unavailable')
        suite = build_workflow_cases(spec, candidate, graph)
        require(len(suite['cases']) <= 100, 'workflow suite exceeds 100-case limit')
        cases = [{'case_id': case['case_id'], **execute_case(spec, case)} for case in suite['cases']]
        evaluation = {'execution': 'recorded_fixtures_only', 'cases': cases,
                      'limitations': ['Recorded fixtures do not prove live agent portability.']}
        replacement = {'workflow': spec, 'cases': suite, 'evaluation': evaluation,
                       'evidence_revision': entry['evidence_revision'], 'exported': False}
        doc['workflows'][doc['workflows'].index(entry)] = replacement
        self._write_store(folder / 'workflows.json', doc)
        # Pin reviewed contracts independently of future draft changes.
        self._write_store(folder / 'reviews' / (spec['revision'] + '.json'), replacement)
        self._touch(folder)
        return self._detail(folder)

    def export_skill(self, request):
        from .skill_export import export_skill
        folder = self._task_dir(request.get('task_id'))
        doc, entry = self._select(folder, request)
        spec = entry['workflow']
        require(spec['status'] == 'reviewed', 'review a workflow before exporting it')
        require(entry['evidence_revision'] == self._summary(folder)['revision'],
                'workflow evidence changed; review current evidence before export')
        include_cases = request.get('include_cases', False)
        require(type(include_cases) is bool, 'include_cases must be a boolean')
        destination = folder / 'exports' / (spec['revision'] + ('-with-cases' if include_cases else '-contract'))
        if not destination.exists():
            self._checked(destination)
            export_skill(spec, destination, suite=entry.get('cases') if include_cases else None)
        self._checked(destination)
        for exported_file in destination.rglob('*'):
            self._checked(exported_file)
        manifest = self._read_store(destination / 'manifest.json')
        entry['exported'] = True
        self._write_store(folder / 'workflows.json', doc)
        self._touch(folder)
        return {**self._detail(folder), 'export': {'path': str(destination), 'manifest': manifest,
                'includes_private_evidence': include_cases}}

    def get(self, request):
        folder = self._task_dir(request.get('task_id'))
        artifact = request.get('artifact')
        if artifact is None:
            return self._detail(folder)
        require(artifact in ARTIFACTS, 'unknown evidence artifact')
        if artifact == 'graph':
            document = build_execution_graph(active_events(self._store(folder)))
        else:
            path = folder / (artifact + '.json')
            require(path.is_file(), 'artifact is not available yet')
            document = self._read_store(path)
        return {'artifact': artifact, 'document': document}

    def handle(self, request):
        require(isinstance(request, dict) and request.get('bridge_version') == BRIDGE_VERSION, 'unsupported bridge_version')
        operation = request.get('operation')
        require(operation in {'capture', 'attach_trace', 'build_workflow', 'review_workflow', 'export_skill', 'list', 'get', 'sources'},
                'unsupported bridge operation')
        with self._lock():
            if operation == 'list':
                folders = sorted((self.root / 'tasks').glob('*/metadata.json'))
                tasks = [self._summary(path.parent) for path in folders]
                return {'tasks': sorted(tasks, key=lambda task: task['updated_at'], reverse=True)}
            if operation == 'sources':
                return {'sources': self._source_list()}
            return getattr(self, operation)(request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True)
    parser.add_argument('--sources', help='Trusted startup JSON mapping source IDs to explicit native trace files')
    parser.add_argument('--request-file', help='Read one request from a daemon-created input file instead of stdin')
    args = parser.parse_args(argv)
    try:
        if args.request_file:
            with Path(args.request_file).open('rb') as stream:
                raw = stream.read(MAX_REQUEST_BYTES + 1)
        else:
            raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        require(len(raw) <= MAX_REQUEST_BYTES, 'bridge request exceeds size limit')
        request = _json(raw)
        sources = _read(args.sources) if args.sources else None
        result = EvidenceBridge(args.store, sources).handle(request)
        response = {'bridge_version': BRIDGE_VERSION, 'ok': True, 'result': result}
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        response = {'bridge_version': BRIDGE_VERSION, 'ok': False,
                    'error': {'code': 'invalid_request' if isinstance(error, (ValueError, TypeError, KeyError)) else 'storage_error',
                              'message': str(error)[:400]}}
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return 0 if response['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
