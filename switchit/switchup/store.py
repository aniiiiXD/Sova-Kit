"""Transactional projects, immutable captures and durable improvement rounds."""
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from switchit.contracts import fingerprint, require
from switchit.events import normalize_event

from .contracts import project_name, record_id, replay_suite, scrub_evidence


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY, document TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY, project TEXT NOT NULL REFERENCES projects(name),
    event_id TEXT NOT NULL, trace_id TEXT NOT NULL, session_id TEXT NOT NULL,
    priority INTEGER NOT NULL, learnable INTEGER NOT NULL, reviewed INTEGER NOT NULL DEFAULT 0, document TEXT NOT NULL,
    UNIQUE (project, event_id)
);
CREATE INDEX IF NOT EXISTS event_traces ON events(project, trace_id, sequence);
CREATE INDEX IF NOT EXISTS evidence_queue ON events(project, reviewed, priority DESC, sequence DESC);
CREATE TABLE IF NOT EXISTS partitions (
    project TEXT NOT NULL REFERENCES projects(name), identity TEXT NOT NULL, split TEXT NOT NULL,
    PRIMARY KEY (project, identity)
);
CREATE TABLE IF NOT EXISTS rounds (
    project TEXT NOT NULL REFERENCES projects(name), id TEXT NOT NULL, document TEXT NOT NULL, summary TEXT NOT NULL,
    PRIMARY KEY (project, id)
);
CREATE TABLE IF NOT EXISTS suites (
    project TEXT NOT NULL REFERENCES projects(name), digest TEXT NOT NULL, document TEXT NOT NULL,
    PRIMARY KEY (project, digest)
);
CREATE TABLE IF NOT EXISTS history (
    project TEXT NOT NULL REFERENCES projects(name), generation INTEGER NOT NULL, document TEXT NOT NULL,
    PRIMARY KEY (project, generation)
);
CREATE TABLE IF NOT EXISTS invocations (
    project TEXT NOT NULL REFERENCES projects(name), id TEXT NOT NULL, status TEXT NOT NULL,
    PRIMARY KEY (project, id)
);
"""


class SwitchUpConflict(ValueError):
    """A generation, operation or evidence identity conflicts with saved state."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def capture_event(raw):
    require(isinstance(raw, dict), 'capture must be an evidence event object')
    event = normalize_event({**raw, 'payload': scrub_evidence(raw.get('payload', {})),
                             'source': scrub_evidence(raw.get('source', {}))})
    for field in ('event_id', 'trace_id'):
        record_id(event[field], field)
    if event['session_id'] is not None:
        record_id(event['session_id'], 'session_id')
    return event


def _identities(trace, session):
    return (f'trace:{trace}', f'session:{session or trace}')


def _read(db, sql, parameters):
    row = db.execute(sql, parameters).fetchone()
    if row is None:
        raise KeyError('SwitchUP project or round does not exist')
    return json.loads(row['document'])


def _project(db, name):
    return _read(db, 'SELECT document FROM projects WHERE name = ?', (project_name(name),))


def _round(db, name, round_id):
    record_id(round_id, 'round_id')
    return _read(db, 'SELECT document FROM rounds WHERE project = ? AND id = ?', (project_name(name), round_id))


def _generation(project, expected):
    if project['generation'] != expected:
        raise SwitchUpConflict('project generation changed; reload the active version')


def _label(db, name, trace, session, split):
    for identity in _identities(trace, session):
        row = db.execute('SELECT split FROM partitions WHERE project = ? AND identity = ?', (name, identity)).fetchone()
        if row and row['split'] != split:
            raise SwitchUpConflict(f'{identity} was already assigned to {row["split"]}; use an independent session')
        db.execute('INSERT OR IGNORE INTO partitions VALUES (?, ?, ?)', (name, identity, split))


def _pin_suite(db, project, suite, digest):
    name = project['name']
    for case in suite['cases']:
        trace_id, split = case['trace_id'], case['split']
        _label(db, name, trace_id, case.get('split_group'), split)
        prior = db.execute('SELECT DISTINCT session_id FROM events WHERE project = ? AND trace_id = ?', (name, trace_id))
        for row in prior:
            _label(db, name, trace_id, row['session_id'], split)
        for event in case.get('events', []):
            _label(db, name, event['trace_id'], event.get('session_id'), split)
    db.execute('INSERT OR IGNORE INTO suites VALUES (?, ?, ?)', (name, digest, _json(suite)))


def _insert_events(db, name, events):
    added = 0
    for event in events:
        trace_id, session_id = event['trace_id'], event['session_id'] or event['trace_id']
        partition = db.execute('SELECT split FROM partitions WHERE project = ? AND identity IN (?, ?)',
                               (name, *_identities(trace_id, session_id))).fetchone()
        if partition:
            _label(db, name, trace_id, session_id, partition['split'])
        payload = event['payload']
        response = payload.get('response')
        priority = 2 if payload.get('correction') else int(bool(
            payload.get('error') or (isinstance(response, dict) and response.get('error'))))
        learnable = event['evidence'] == 'recorded' and event['kind'] in (
            'model_call', 'user_input', 'tool_call', 'tool_result', 'state')
        document = _json(event)
        changed = db.execute('''INSERT OR IGNORE INTO events
            (project, event_id, trace_id, session_id, priority, learnable, document) VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (name, event['event_id'], trace_id, session_id, priority, learnable, document))
        if not changed.rowcount:
            original = _read(db, 'SELECT document FROM events WHERE project = ? AND event_id = ?', (name, event['event_id']))
            if original != event:
                raise SwitchUpConflict(f'conflicting capture ID {event["event_id"]}')
        added += changed.rowcount
    return added


def _save_project(db, project, change):
    project['generation'] += 1
    name = project['name']
    db.execute('UPDATE projects SET document = ? WHERE name = ?', (_json(project), name))
    db.execute('INSERT INTO history VALUES (?, ?, ?)',
               (name, project['generation'], _json({'generation': project['generation'], 'at': _now(), **change})))
    return project


def _save_round(db, record):
    db.execute('UPDATE rounds SET document = ?, summary = ? WHERE project = ? AND id = ?',
               (_json(record), _json(round_summary(record)), record['project'], record['id']))


class Workspace:
    def __init__(self, directory):
        directory = Path(directory).expanduser()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory / 'switchup.sqlite3'
        self.path.touch(mode=0o600, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute('PRAGMA journal_mode = WAL')
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self, *, write=False):
        with closing(sqlite3.connect(self.path, timeout=10, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys = ON')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def create(self, name, workflow, connection, recipe, suite=None):
        name = project_name(name)
        suite_digest = replay_suite(workflow, suite) if suite is not None else None
        project = {'switchup_project_version': 1, 'name': name, 'generation': 0, 'created_at': _now(),
                   'workflow': workflow, 'connection': connection, 'initial_recipe': recipe,
                   'active_recipe': recipe, 'suite': suite, 'suite_digest': suite_digest}
        with self._db(write=True) as db:
            inserted = db.execute('INSERT OR IGNORE INTO projects VALUES (?, ?)', (name, _json(project)))
            if not inserted.rowcount:
                raise SwitchUpConflict(f'project {name} already exists')
            if suite is not None:
                _pin_suite(db, project, suite, suite_digest)
                _insert_events(db, name, [capture_event(event) for case in suite['cases'] for event in case.get('events', [])])
        return project

    def project(self, name):
        with self._db() as db:
            return _project(db, name)

    def install_suite(self, name, suite, *, expected_generation):
        with self._db(write=True) as db:
            project = _project(db, name)
            _generation(project, expected_generation)
            digest = replay_suite(project['workflow'], suite)
            if digest == project['suite_digest']:
                return project
            _pin_suite(db, project, suite, digest)
            _insert_events(db, name, [capture_event(event) for case in suite['cases'] for event in case.get('events', [])])
            project.update(suite=suite, suite_digest=digest)
            return _save_project(db, project, {'action': 'snapshot', 'suite_digest': digest})

    def ingest(self, name, rows):
        require(isinstance(rows, list) and 0 < len(rows) <= 3000, 'ingest accepts 1–3000 events per batch')
        events = [capture_event(event) for event in rows]
        with self._db(write=True) as db:
            _project(db, name)
            added = _insert_events(db, name, events)
        return {'added': added, 'duplicates': len(events) - added}

    def feedback(self, name, trace_id, correction, event_id=None):
        record_id(trace_id, 'trace_id')
        require(isinstance(correction, str) and 0 < len(correction.strip()) <= 4000,
                'correction must contain 1–4000 characters')
        event_id = record_id(event_id, 'event_id') if event_id is not None else uuid4().hex
        with self._db(write=True) as db:
            _project(db, name)
            row = db.execute('SELECT session_id FROM events WHERE project = ? AND trace_id = ? ORDER BY sequence DESC LIMIT 1',
                             (name, trace_id)).fetchone()
            require(row is not None, 'feedback must refer to an imported trace')
            event = capture_event({'event_id': event_id, 'trace_id': trace_id, 'session_id': row['session_id'],
                                   'kind': 'state', 'source': {'importer': 'switchup-feedback'},
                                   'payload': {'correction': correction}})
            _insert_events(db, name, [event])
        return {'event_id': event_id, 'trace_id': trace_id}

    def events(self, name, *, exclude_holdout=False):
        with self._db() as db:
            _project(db, name)
            cursor = db.execute('''SELECT document FROM events AS e WHERE project = ? AND (? = 0 OR NOT EXISTS (
                SELECT 1 FROM partitions AS p WHERE p.project = e.project AND p.split = 'holdout'
                    AND p.identity IN ('trace:' || e.trace_id, 'session:' || e.session_id)
                )) ORDER BY sequence''', (name, exclude_holdout))
            for row in cursor:
                yield json.loads(row['document'])

    def prepare_round(self, name, limit):
        require(type(limit) is int and 1 <= limit <= 40, 'evidence limit must be between 1 and 40')
        with self._db(write=True) as db:
            project = _project(db, name)
            require(project['suite'] is not None, 'pin a replay suite before preparing an improvement')
            rows = db.execute('''SELECT sequence, document FROM events AS e
                WHERE project = ? AND learnable = 1 AND NOT EXISTS (
                    SELECT 1 FROM partitions AS p WHERE p.project = e.project AND p.split = 'holdout'
                        AND p.identity IN ('trace:' || e.trace_id, 'session:' || e.session_id)
                ) ORDER BY reviewed, priority DESC, sequence DESC LIMIT ?''', (name, limit)).fetchall()
            evidence = []
            for row in rows:
                event = json.loads(row['document'])
                evidence.append(event)
                _label(db, name, event['trace_id'], event['session_id'], 'train')
                db.execute('UPDATE events SET reviewed = reviewed + 1 WHERE sequence = ?', (row['sequence'],))
            require(evidence, 'import recorded training traces before preparing an improvement')
            round_doc = {'switchup_round_version': 1, 'id': uuid4().hex, 'project': name, 'status': 'prepared',
                         'created_at': _now(), 'updated_at': _now(), 'generation': project['generation'],
                         'workflow_revision': project['workflow']['revision'], 'suite_digest': project['suite_digest'],
                         'connection_digest': fingerprint(project['connection']), 'baseline': project['active_recipe'],
                         'evidence_digest': fingerprint(evidence), 'evidence': evidence}
            db.execute('INSERT INTO rounds VALUES (?, ?, ?, ?)',
                       (name, round_doc['id'], _json(round_doc), _json(round_summary(round_doc))))
        return round_doc

    def round(self, name, round_id):
        with self._db() as db:
            return _round(db, name, round_id)

    def suite(self, name, digest):
        with self._db() as db:
            return _read(db, 'SELECT document FROM suites WHERE project = ? AND digest = ?', (project_name(name), digest))

    def transition(self, name, round_id, expected, status, *, fresh=False, **updates):
        with self._db(write=True) as db:
            project, record = _project(db, name), _round(db, name, round_id)
            if record['status'] != expected:
                raise SwitchUpConflict(f'round is {record["status"]}; {expected} is required')
            if fresh:
                _generation(project, record['generation'])
            record.update(status=status, updated_at=_now(), **updates)
            _save_round(db, record)
        return project, record

    def activate(self, name, round_id, expected_generation, validate):
        with self._db(write=True) as db:
            project, record = _project(db, name), _round(db, name, round_id)
            _generation(project, expected_generation)
            _generation(project, record['generation'])
            require(record['status'] == 'verified', 'activate a verified improvement round')
            recipe = validate(project, record)
            before = project['active_recipe']
            project['active_recipe'] = recipe
            _save_project(db, project, {'action': 'activate', 'round_id': round_id,
                                        'previous_recipe': before, 'recipe_id': recipe['recipe_id']})
            record.update(status='activated', updated_at=_now(), activated_generation=project['generation'])
            _save_round(db, record)
        return project

    def rollback(self, name, expected_generation):
        with self._db(write=True) as db:
            project = _project(db, name)
            _generation(project, expected_generation)
            active = project['active_recipe']['recipe_id']
            require(active != project['initial_recipe']['recipe_id'], 'project is already on its initial recipe')
            rows = db.execute('SELECT document FROM history WHERE project = ? ORDER BY generation DESC', (name,))
            for row in rows:
                change = json.loads(row['document'])
                if change['action'] == 'activate' and change['recipe_id'] == active:
                    project['active_recipe'] = change['previous_recipe']
                    return _save_project(db, project, {'action': 'rollback', 'previous_recipe_id': active,
                                                        'recipe_id': project['active_recipe']['recipe_id']})
            raise SwitchUpConflict('active recipe has no recorded predecessor')

    def begin_invocation(self, name, request_id):
        with self._db(write=True) as db:
            project = _project(db, name)
            row = db.execute('INSERT OR IGNORE INTO invocations VALUES (?, ?, ?)', (name, request_id, 'running'))
            if not row.rowcount:
                status = db.execute('SELECT status FROM invocations WHERE project = ? AND id = ?', (name, request_id)).fetchone()[0]
                raise SwitchUpConflict(f'request_id was already submitted ({status})')
        return project

    def finish_invocation(self, name, request_id, events, status):
        with self._db(write=True) as db:
            _insert_events(db, name, events)
            changed = db.execute("UPDATE invocations SET status = ? WHERE project = ? AND id = ? AND status = 'running'",
                                 (status, name, request_id))
            require(changed.rowcount == 1, 'invocation receipt must belong to a running request')

    def status(self, name=None):
        with self._db() as db:
            if name is None:
                projects = [json.loads(row['document']) for row in db.execute('SELECT document FROM projects ORDER BY name')]
            else:
                projects = [_project(db, name)]
            for project in projects:
                key = project['name']
                project['event_count'] = db.execute('SELECT count(*) FROM events WHERE project = ?', (key,)).fetchone()[0]
                rounds = db.execute('SELECT summary FROM rounds WHERE project = ? ORDER BY rowid DESC LIMIT 20', (key,))
                project['rounds'] = [json.loads(row['summary']) for row in rounds]
                history = db.execute('SELECT document FROM history WHERE project = ? ORDER BY generation DESC LIMIT 20', (key,))
                project['history'] = [json.loads(row['document']) for row in history]
        return [project_summary(project) for project in projects]


def round_summary(record):
    keys = ('id', 'project', 'status', 'generation', 'created_at', 'updated_at', 'evidence_digest',
            'suite_digest', 'decision', 'error', 'activated_generation')
    return {key: record[key] for key in keys if key in record}


def project_summary(project):
    keys = ('name', 'generation', 'created_at', 'suite_digest', 'event_count', 'rounds', 'history')
    return {**{key: project[key] for key in keys if key in project},
            'workflow_id': project['workflow']['workflow_id'], 'workflow_revision': project['workflow']['revision'],
            'model': project['connection']['model'], 'provider': project['connection']['kind'],
            'active_recipe_id': project['active_recipe']['recipe_id'],
            'can_rollback': project['active_recipe']['recipe_id'] != project['initial_recipe']['recipe_id'],
            'case_count': len(project['suite']['cases']) if project['suite'] is not None else 0}
