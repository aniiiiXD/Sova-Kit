"""Durable, bounded local jobs over the workflow modules; no remote runtimes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import uuid

from switchit.contracts import ContractError, require
from switchit.dependencies import profile_dependencies
from switchit.events import events_from_calls, normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.module_demo import synthetic_events
from switchit.normalization import make_call
from switchit.runtime_adapters import FixtureRuntime
from switchit.storage import atomic_write_json
from switchit.workflow_builder import build_workflow, review_workflow
from switchit.workflow_cases import build_workflow_cases
from switchit.workflow_evaluation import evaluate_workflow
from switchit.workflow_patterns import discover_workflows
from switchit.workflow_runner import run_workflow

MAX_RECORDS = 3000
MAX_ACTIVE = 24
ARTIFACTS = frozenset({'input.json', 'graph.json', 'candidates.json', 'draft.json',
                       'dependencies.json', 'workflow.json', 'cases.json', 'evaluation.json'})
CHECK_TYPES = frozenset({'no_error', 'tool_called', 'tool_args_include', 'tool_calls_equal',
    'output_contains', 'output_equals', 'output_min_chars', 'latency_ms_max', 'usage_input_max', 'usage_output_max',
    'state_equals', 'artifact_equals', 'artifact_exists', 'step_before', 'allowed_tools',
    'tool_sequence', 'cost_max'})
# Regex is intentionally not exposed over HTTP: arbitrary patterns can block a worker.
ACTIVE = frozenset({'queued', 'running', 'cancelling'})


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class JobConflict(ValueError):
    pass


class JobCancelled(Exception):
    pass


def validate_submission(payload):
    require(isinstance(payload, dict), 'Request must be a JSON object.')
    require(set(payload) <= {'name', 'events', 'calls', 'demo', 'aegis'}, 'Unsupported submission field.')
    name = payload.get('name', 'Imported workflow')
    require(isinstance(name, str) and 0 < len(name.strip()) <= 120, 'Name must contain 1–120 characters.')
    selected = [key for key in ('events', 'calls', 'demo', 'aegis') if key in payload]
    require(len(selected) == 1, 'Provide exactly one of events, calls, demo, or aegis.')
    mode = selected[0]
    if mode == 'demo':
        require(payload['demo'] is True, 'demo must be true.')
        events = synthetic_events()
    elif mode == 'aegis':
        from switchit.importers.aegis import import_aegis
        events = import_aegis(payload['aegis'])['events']
    else:
        rows = payload[mode]
        require(isinstance(rows, list) and 0 < len(rows) <= MAX_RECORDS, f'Provide 1–{MAX_RECORDS} records.')
        require(all(isinstance(row, dict) for row in rows), 'Each record must be a JSON object.')
        if mode == 'events':
            events = [normalize_event(row, row.get('source') or {'importer': 'platform-upload'}) for row in rows]
        else:
            calls = [make_call(row, 'platform-upload', f'upload:{index}') for index, row in enumerate(rows)]
            events = events_from_calls(calls)
    require(events and len(events) <= MAX_RECORDS, f'Normalized input must contain 1–{MAX_RECORDS} events.')
    require(len({event['event_id'] for event in events}) == len(events), 'Input contains duplicate event IDs.')
    return name.strip(), events, mode == 'demo'


def validate_review(payload):
    require(isinstance(payload, dict), 'Review must be a JSON object.')
    require(set(payload) <= {'reviewer', 'checks', 'candidate_id', 'inputs', 'outputs', 'steps', 'allowed_tools'}, 'Unsupported review field.')
    reviewer = payload.get('reviewer')
    require(isinstance(reviewer, str) and 0 < len(reviewer.strip()) <= 100, 'Enter a reviewer name (1–100 characters).')
    checks = payload.get('checks')
    require(isinstance(checks, list) and 0 < len(checks) <= 30, 'Provide 1–30 explicit checks.')
    for check in checks:
        require(isinstance(check, dict) and check.get('type') in CHECK_TYPES, 'Unsupported check type.')
        require(set(check) <= {'type', 'required', 'source', 'name', 'names', 'arguments', 'text',
                             'value', 'path', 'before', 'after', 'calls'}, 'Unsupported check field.')
        if 'required' in check:
            require(isinstance(check['required'], bool), 'required must be a boolean.')
        for key in ('text', 'path', 'name', 'before', 'after'):
            if key in check:
                require(isinstance(check[key], str) and len(check[key]) <= 2000, f'{key} must be a short string.')
        if check['type'] in ('output_contains', 'output_equals', 'artifact_exists', 'artifact_equals', 'state_equals'):
            key = 'text' if check['type'] in ('output_contains', 'output_equals') else 'path'
            require(bool(check.get(key)), f'{key} is required for {check["type"]}.')
        if check['type'] in ('output_min_chars', 'latency_ms_max', 'usage_input_max', 'usage_output_max', 'cost_max'):
            value = check.get('value')
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1e12,
                    'Numeric check value must be nonnegative and finite.')
        if 'names' in check:
            require(isinstance(check['names'], list) and all(isinstance(name, str) for name in check['names']), 'names must be a list of strings.')
        if 'arguments' in check:
            require(isinstance(check['arguments'], dict), 'arguments must be an object.')
        if 'calls' in check:
            require(isinstance(check['calls'], list) and len(check['calls']) <= 200 and
                    all(isinstance(call, dict) and isinstance(call.get('name'), str) and
                        isinstance(call.get('arguments', {}), dict) for call in check['calls']), 'calls must contain named tool-call objects.')
    require(any(check.get('required', True) and check['type'] != 'no_error' for check in checks),
            'Add at least one required outcome check beyond no_error.')
    if 'candidate_id' in payload:
        require(isinstance(payload['candidate_id'], str), 'candidate_id must be a string.')
    for key, expected in (('inputs', dict), ('outputs', list), ('steps', list), ('allowed_tools', list)):
        if key in payload:
            require(isinstance(payload[key], expected), f'{key} has an invalid shape.')
    if 'steps' in payload:
        require(0 < len(payload['steps']) <= 300 and all(isinstance(s, dict) for s in payload['steps']), 'Provide 1–300 step objects.')
    if 'allowed_tools' in payload:
        require(all(isinstance(t, str) for t in payload['allowed_tools']), 'allowed_tools must contain strings.')
    return {**deepcopy(payload), 'reviewer': reviewer.strip(), 'candidate_id': payload.get('candidate_id')}


class JobManager:
    def __init__(self, data_dir, workers=2):
        self.root = (Path(data_dir).expanduser().resolve() / 'jobs')
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = open(self.root.parent / '.platform.lock', 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.owner.seek(0)
                self.owner.write(b'0')
                self.owner.flush()
                self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise JobConflict('Another platform server owns this data directory.') from None
        self.lock = threading.RLock()
        self.jobs = {}
        self.futures = {}
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='switchit-platform')
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text())
                if job.get('id') != path.parent.name or not re.fullmatch(r'[a-f0-9]{32}', job['id']):
                    continue
                if job['status'] in ACTIVE:
                    job.update(status='interrupted', message='Server stopped during this job. Resume to continue.', updated_at=now())
                    atomic_write_json(path, job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue

    def _directory(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise KeyError('Unknown job.')
        path = self.root / job_id
        if not path.resolve().is_relative_to(self.root):
            raise KeyError('Unknown job.')
        return path

    def _save(self, job):
        job['updated_at'] = now()
        atomic_write_json(self._directory(job['id']) / 'job.json', job)

    def _snapshot(self, job):
        result = deepcopy(job)
        result.pop('review', None)
        result['artifacts'] = sorted(name for name in ARTIFACTS if (self._directory(job['id']) / name).is_file())
        return result

    def list(self):
        with self.lock:
            return [self._snapshot(job) for job in sorted(self.jobs.values(), key=lambda item: item['created_at'], reverse=True)]

    def get(self, job_id):
        with self.lock:
            self._directory(job_id)
            if job_id not in self.jobs:
                raise KeyError('Unknown job.')
            return self._snapshot(self.jobs[job_id])

    def artifact(self, job_id, name):
        self.get(job_id)
        if name not in ARTIFACTS:
            raise KeyError('Unknown artifact.')
        path = self._directory(job_id) / name
        if not path.resolve().is_relative_to(self._directory(job_id).resolve()) or not path.is_file():
            raise KeyError('Artifact is not available.')
        return path.read_bytes()

    def submit(self, payload):
        name, events, synthetic = validate_submission(payload)
        with self.lock:
            if self.closed or sum(job['status'] in ACTIVE for job in self.jobs.values()) >= MAX_ACTIVE:
                raise JobConflict('The local queue is full. Wait for a job to finish.')
            job_id = uuid.uuid4().hex
            job = {'id': job_id, 'name': name, 'status': 'queued', 'phase': 'import',
                   'message': 'Queued for evidence analysis.', 'created_at': now(), 'updated_at': now(),
                   'synthetic': synthetic, 'cancel_requested': False, 'task': 'analyze', 'progress': 0,
                   'counts': {'events': len(events), 'traces': len({e['trace_id'] for e in events})}, 'error': None}
            self.jobs[job_id] = job
            atomic_write_json(self._directory(job_id) / 'input.json', {'events': events})
            self._save(job)
            self._enqueue(job_id)
            return self._snapshot(job)

    def _enqueue(self, job_id):
        self.futures[job_id] = self.pool.submit(self._work, job_id)

    def _checkpoint(self, job_id, phase, progress, message):
        with self.lock:
            job = self.jobs[job_id]
            if job['cancel_requested'] or self.closed:
                raise JobCancelled
            job.update(status='running', phase=phase, progress=progress, message=message)
            self._save(job)

    def _read(self, job_id, name):
        return json.loads(self.artifact(job_id, name))

    def _write(self, job_id, name, document):
        require(name in ARTIFACTS, 'Invalid artifact name.')
        atomic_write_json(self._directory(job_id) / name, document)

    def _work(self, job_id):
        try:
            with self.lock:
                task = self.jobs[job_id]['task']
            if task == 'analyze':
                self._analyze(job_id)
            else:
                self._evaluate(job_id)
        except JobCancelled:
            with self.lock:
                self.jobs[job_id].update(status='cancelled', message='Cancelled. Saved evidence can be resumed.')
                self._save(self.jobs[job_id])
        except Exception as error:
            from switchit.redaction import redact_text
            with self.lock:
                self.jobs[job_id].update(status='failed', error=redact_text(str(error))[:1000],
                                         message='This job needs attention. Inspect the error, then resume or review again.')
                self._save(self.jobs[job_id])

    def _analyze(self, job_id):
        self._checkpoint(job_id, 'import', 10, 'Validating the imported evidence.')
        events = self._read(job_id, 'input.json')['events']
        self._checkpoint(job_id, 'graph', 30, 'Reconstructing recorded relationships and inferred ordering.')
        graph = build_execution_graph(events)
        self._write(job_id, 'graph.json', graph)
        self._write(job_id, 'dependencies.json', profile_dependencies(events))
        self._checkpoint(job_id, 'workflows', 55, 'Finding task segments and workflow candidates.')
        candidates = discover_workflows(graph)
        self._write(job_id, 'candidates.json', candidates)
        require(candidates['candidates'], 'No workflow candidates were found in these events.')
        drafts = []
        for candidate in candidates['candidates']:
            try:
                drafts.append(build_workflow(candidate, graph))
            except ContractError:
                continue
        if not drafts and all(event['evidence'] == 'metadata_only' for event in events):
            with self.lock:
                job = self.jobs[job_id]
                if job['cancel_requested']:
                    raise JobCancelled
                job.update(status='evidence_only', phase='complete', progress=100,
                    message='Activity evidence imported. Model/tool execution traces are needed to build a replayable workflow.',
                    counts={**job['counts'], 'candidates': 0, 'graph_issues': len(graph['issues'])})
                self._save(job)
            return
        require(drafts, 'This evidence does not contain an executable draft. Metadata-only imports cannot be replayed.')
        self._write(job_id, 'draft.json', {'drafts': drafts})
        self._checkpoint(job_id, 'review', 65, 'Review the workflow and define what a successful outcome means.')
        with self.lock:
            job = self.jobs[job_id]
            if job['cancel_requested']:
                raise JobCancelled
            job.update(status='needs_review', counts={**job['counts'], 'candidates': len(drafts),
                                                      'graph_issues': len(graph['issues'])})
            self._save(job)

    def review(self, job_id, payload):
        review = validate_review(payload)
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            if job['status'] in ACTIVE or self.closed:
                raise JobConflict('Wait for the current job to finish before reviewing.')
            if sum(j['status'] in ACTIVE for j in self.jobs.values()) >= MAX_ACTIVE:
                raise JobConflict('The local queue is full. Wait for a job to finish.')
            drafts = self._read(job_id, 'draft.json')['drafts']
            candidate_id = review['candidate_id'] or drafts[0]['workflow_id']
            require(any(draft['workflow_id'] == candidate_id for draft in drafts), 'Unknown workflow candidate.')
            draft = next(d for d in drafts if d['workflow_id'] == candidate_id)
            review_workflow(draft, reviewer=review['reviewer'], success_checks=review['checks'],
                            **{k: review[k] for k in ('inputs', 'outputs', 'steps', 'allowed_tools') if k in review})
            review['candidate_id'] = candidate_id
            job.pop('verdicts', None)
            for name in ('workflow.json', 'cases.json', 'evaluation.json'):
                (self._directory(job_id) / name).unlink(missing_ok=True)
            job.update(review=review, task='evaluate', status='queued', phase='review', progress=65,
                       message='Queued for reviewed fixture verification.', cancel_requested=False, error=None)
            self._save(job)
            self._enqueue(job_id)
            return self._snapshot(job)

    def _evaluate(self, job_id):
        self._checkpoint(job_id, 'review', 68, 'Applying your explicit outcome checks.')
        with self.lock:
            review = deepcopy(self.jobs[job_id]['review'])
        draft = next(d for d in self._read(job_id, 'draft.json')['drafts'] if d['workflow_id'] == review['candidate_id'])
        spec = review_workflow(draft, reviewer=review['reviewer'], success_checks=review['checks'],
                               **{k: review[k] for k in ('inputs', 'outputs', 'steps', 'allowed_tools') if k in review})
        self._write(job_id, 'workflow.json', spec)
        candidate = next(c for c in self._read(job_id, 'candidates.json')['candidates'] if c['candidate_id'] == spec['workflow_id'])
        self._checkpoint(job_id, 'cases', 76, 'Building whole-segment cases and exact tool fixtures.')
        suite = build_workflow_cases(spec, candidate, self._read(job_id, 'graph.json'))
        require(len(suite['cases']) <= 100, 'This workflow contains more than 100 cases. Import a smaller evidence set.')
        self._write(job_id, 'cases.json', suite)
        trials = []
        for index, case in enumerate(suite['cases']):
            self._checkpoint(job_id, 'evaluation', 80 + int(index/max(1, len(suite['cases']))*18),
                             f'Verifying recorded case {index+1} of {len(suite["cases"])}.')
            run = run_workflow(spec, case, FixtureRuntime(case), max_steps=300, max_seconds=15)
            trials.append({'case_id': case['case_id'], 'split': case['split'],
                           'run': run, 'evaluation': evaluate_workflow(spec['success_checks'], run)})
        verdicts = {name: sum(trial['evaluation']['verdict'] == name for trial in trials)
                    for name in ('pass', 'fail', 'error', 'uncertain')}
        result = {'platform_evaluation_version': 1, 'mode': 'recorded_fixture_verification',
                  'workflow_id': spec['workflow_id'], 'workflow_revision': spec['revision'],
                  'verdicts': verdicts, 'trials': trials, 'uncertainty': suite['uncertainty'],
                  'limitations': ['Recorded outputs and fixtures are being checked against your criteria.',
                                  'No target vendor, live model, or real business tool was executed.']}
        self._write(job_id, 'evaluation.json', result)
        self._checkpoint(job_id, 'complete', 100, 'Recorded fixture verification finished.')
        with self.lock:
            if self.jobs[job_id]['cancel_requested']:
                raise JobCancelled
            self.jobs[job_id].update(status='completed', verdicts=verdicts,
                                    counts={**self.jobs[job_id]['counts'], 'cases': len(trials)})
            self._save(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            if job['status'] in ('completed', 'cancelled'):
                raise JobConflict('This job is already finished.')
            job['cancel_requested'] = True
            job.update(status='cancelling' if job['status'] in ACTIVE else 'cancelled',
                       message='Cancellation requested; the current phase will finish first.')
            self._save(job)
            return self._snapshot(job)

    def resume(self, job_id):
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            future = self.futures.get(job_id)
            if job['status'] not in ('failed', 'interrupted', 'cancelled') or (future and not future.done()):
                raise JobConflict('Only stopped jobs can be resumed.')
            if self.closed:
                raise JobConflict('Server is shutting down.')
            if sum(j['status'] in ACTIVE for j in self.jobs.values()) >= MAX_ACTIVE:
                raise JobConflict('The local queue is full. Wait for a job to finish.')
            job.pop('verdicts', None)
            if job['task'] == 'evaluate':
                (self._directory(job_id) / 'evaluation.json').unlink(missing_ok=True)
            job.update(status='queued', cancel_requested=False, error=None, message='Queued to resume from saved input.')
            self._save(job)
            self._enqueue(job_id)
            return self._snapshot(job)

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.owner.close()
