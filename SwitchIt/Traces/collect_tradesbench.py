"""Join one published TradesBench run to its final simulator outcomes."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import json
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from switchit.contracts import require
from switchit.data_paths import kit_data_dir
from switchit.events import normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.storage import atomic_write_json, digest_file


REVISION = '0c3983bb519c9885a63f3085818ca516e3ec2067'
RUN = 'gpt-6.1-sol-xhigh-a2'
RUN_URL = f'https://huggingface.co/datasets/adityabressl/trades-bench/resolve/{REVISION}/runs/seed5/{RUN}'
DATABASE_SHA256 = 'c3999089b2e7a17eb80e711677e8f295dd52c9630e5aab183c0d03a0555a751d'
SAMPLE_CALLS = (1, 7, 8, 105)
RECEIPT_OUTCOMES = {'booked': 'accepted', 'rejected': 'rejected'}
OUTCOME_SQL = '''
WITH daily_turns AS (
    SELECT i.*, sequence - MAX(CASE WHEN type = 'opening' THEN sequence END)
           OVER (ORDER BY sequence) + 1 AS daily_turn
    FROM _invocations i
)
SELECT c.id AS call_id, c.call_kind, c.status AS call_status, c.terminal_reason,
       c.service_date, c.target_appointment_id, h.arrival_event_id,
       i.invocation_id, i.sequence, i.daily_turn, j.id AS job_id, j.outcome AS job_outcome,
       a.id AS appointment_id, a.status AS appointment_status
FROM _calls c
JOIN _call_hidden h ON h.call_id = c.id
JOIN daily_turns i ON i.origin_event_id = h.arrival_event_id AND i.type = 'call'
LEFT JOIN _jobs j ON j.id = c.created_job_id
LEFT JOIN _appointments a ON a.job_id = j.id
ORDER BY i.sequence
'''


def fetch(url):
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    return raw, {'url': url, 'revision': REVISION, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def read_final_state(path):
    require(digest_file(path) == DATABASE_SHA256, 'simulator database digest differs from the published run')
    source = {'url': f'{RUN_URL}/run.db', 'revision': REVISION,
              'sha256': DATABASE_SHA256, 'bytes': path.stat().st_size}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA trusted_schema = OFF')
        db.row_factory = sqlite3.Row
        run = dict(db.execute('''
            SELECT run_id, master_seed, model_id, harness_name, harness_version, status, spec_version,
                   config_version, config_hash, first_service_date, last_service_date
            FROM _run
        ''').fetchone())
        calls = [dict(row) for row in db.execute(OUTCOME_SQL)]
        invocation_count = db.execute("SELECT count(*) FROM _invocations WHERE type = 'call'").fetchone()[0]
        call_count = db.execute('SELECT count(*) FROM _calls').fetchone()[0]
        require(len(calls) == len({row['call_id'] for row in calls}) == invocation_count == call_count,
                'simulator call and invocation join is incomplete or ambiguous')
        appointments = dict(db.execute('SELECT status, count(*) FROM _appointments GROUP BY status'))
        histories = {}
        for row in calls:
            if row['call_id'] in SAMPLE_CALLS and row['appointment_id'] is not None:
                histories[row['call_id']] = [dict(event) for event in db.execute('''
                    SELECT id, event_type, occurred_at, technician_id
                    FROM _appointment_events WHERE appointment_id = ?
                    ORDER BY occurred_at, id
                ''', (row['appointment_id'],))]
    return run, source, calls, appointments, histories


def terminal_receipts(result, case):
    output = (result.get('structuredContent') or {}).get('output', '')
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or not isinstance(data := item.get('data'), dict):
            continue
        if (item.get('meta', {}).get('invocation_id') == case['invocation_id']
                and data.get('call_id') == case['call_id']
                and data.get('outcome') == RECEIPT_OUTCOMES[case['call_status']]):
            require(data.get('appointment_id') == case['appointment_id'] and data.get('job_id') == case['job_id'],
                    'terminal receipt entity IDs differ from final database')
            yield {key: data[key] for key in ('call_id', 'job_id', 'appointment_id', 'outcome') if key in data}


def collect_case(case, run, database_source, history):
    # Keep the source URL valid when canonical events mask phone-like strings.
    date_slug = case['service_date'].replace('-', '%2D')
    stem = f"{RUN_URL}/pi-logs/{date_slug}-turn-{case['daily_turn']:04d}-call"
    metadata_raw, metadata_source = fetch(f'{stem}.turn.json')
    raw, source = fetch(f'{stem}.stdout.jsonl')
    metadata = json.loads(metadata_raw)
    require(metadata['service_date'] == case['service_date'] and metadata['turn_index'] == case['daily_turn']
            and metadata['kind'] == 'call' and metadata['exit_code'] == 0, 'log metadata differs from simulator join')

    trace_id = f"tradesbench/{run['run_id']}/call-{case['call_id']}"
    events, proposals, starts, receipts = [], {}, {}, []

    def emit(locator, kind, payload, *, origin=source, **relations):
        event = normalize_event({'event_id': f'{trace_id}/{locator}', 'kind': kind, 'trace_id': trace_id,
                                 'session_id': metadata['session_id'], 'index': len(events),
                                 'source': {'importer': 'tradesbench', **origin, 'locator': locator},
                                 'payload': payload, **relations})
        events.append(event)
        return event['event_id']

    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        locator = f'line-{line_number}'
        if record['type'] == 'message_end':
            message = record['message']
            if message['role'] not in ('assistant', 'user'):
                continue
            blocks = message['content']
            if isinstance(blocks, str):
                blocks = [{'type': 'text', 'text': blocks}]
            text = '\n'.join(block['text'] for block in blocks if block['type'] == 'text')
            if message['role'] == 'user':
                emit(locator, 'user_input', {'role': 'user', 'content': text})
            else:
                calls = [{key: block[key] for key in ('id', 'name', 'arguments')}
                         for block in blocks if block['type'] == 'toolCall']
                model_id = emit(locator, 'model_call', {'model': message.get('model', run['model_id']),
                                                       'usage': message.get('usage'),
                                                       'response': {'text': text, 'tool_calls': calls}},
                                uncertainty=['provider_request_missing'])
                proposals.update({call['id']: model_id for call in calls})
        elif record['type'] == 'tool_execution_start':
            tool_id = record['toolCallId']
            require(tool_id not in starts, 'duplicated tool execution ID')
            starts[tool_id] = (record['toolName'], emit(locator, 'tool_call',
                                                       {'name': record['toolName'], 'arguments': record['args']},
                                                       tool_call_id=tool_id, parent_id=proposals.get(tool_id)))
        elif record['type'] == 'tool_execution_end':
            tool_id = record['toolCallId']
            require(tool_id in starts and starts[tool_id][0] == record['toolName'], 'tool receipt lacks a matching start')
            result = record['result']
            result_id = emit(locator, 'tool_result', {'name': record['toolName'], 'content': result['content'],
                                                       'error': record['isError'], 'duration_ms': record.get('durationMs')},
                             tool_call_id=tool_id)
            if not record['isError']:
                receipts.extend({'tool_call_id': tool_id, 'call_event_id': starts[tool_id][1],
                                 'receipt_event_id': result_id, 'data': data}
                                for data in terminal_receipts(result, case))
    require(len(receipts) == 1, 'expected exactly one terminal call receipt in the selected turn')
    state_id = emit('final-state', 'state', {'outcome': case, 'appointment_events': history}, origin=database_source)
    audit = {'call_id': case['call_id'], 'trace_id': trace_id, 'metadata_source': metadata_source,
             'stdout_source': source, 'terminal_receipt': receipts[0], 'state_event_id': state_id,
             'appointment_events': history, 'event_count': len(events)}
    return audit, events


def report(run, calls, appointments, samples):
    dispositions = Counter(row['call_status'] for row in calls)
    lines = [
        '# TradesBench outcome collection', '',
        f"Run `{RUN}` · seed {run['master_seed']} · model `{run['model_id']}`.", '',
        f"The final SQLite database joins **{len(calls)} call turns** to simulator records. "
        f"The call dispositions are {dispositions['booked']} booked, {dispositions['declined']} customer declined, "
        f"{dispositions['rejected']} agent rejected, and {dispositions['acknowledged']} cancellation acknowledged. "
        'These counts match the published run report.', '',
        f"Of the {dispositions['booked']} bookings, {appointments['completed']} completed, "
        f"{appointments['cancelled']} were canceled, {appointments['no_show']} were company no-shows, "
        f"and {appointments['scheduled']} were still scheduled when the simulation ended.", '',
        '| Call | Published turn | Decision receipt | Final outcome |', '| --- | --- | --- | --- |',
    ]
    index = {row['call_id']: row for row in calls}
    for sample in samples:
        row = index[sample['call_id']]
        receipt = sample['terminal_receipt']['data']
        outcome = row['job_outcome'] or row['terminal_reason']
        lines.append(f"| {row['call_id']} | [{row['service_date']} turn {row['daily_turn']}]"
                     f"({sample['stdout_source']['url']}) | `{receipt['outcome']}` | `{outcome}` |")
    lines += ['', 'Each selected tool receipt matches the simulator invocation and call IDs. '
              'Accepted bookings also match the job and appointment IDs. The two no-shows are verified downstream failures. '
              'Evidence that a specific model decision caused either no-show, or that a correction would prevent it, is pending.', '',
              'The join SQL, source hashes, all indexed outcomes, and selected receipt IDs are in [outcomes.json](outcomes.json). '
              'Four normalized traces are in [events.json](events.json).', '',
              f'[Published final database]({RUN_URL}/run.db) · [Published run report]({RUN_URL}/policy-report.json)', '']
    return '\n'.join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True, help='Local copy of the published run.db')
    parser.add_argument('--output', type=Path, default=kit_data_dir(ROOT.parent) / 'traces' / 'tradesbench' / 'seed5' / RUN)
    args = parser.parse_args()
    run, database, calls, appointments, histories = read_final_state(args.database)
    policy_raw, policy_source = fetch(f'{RUN_URL}/policy-report.json')
    policy = json.loads(policy_raw)
    require(run['master_seed'] == policy['master_seed'] == 5 and run['config_hash'] == policy['config_hash']
            and run['status'] == policy['run_status'] == 'completed', 'run report does not match the database')
    observed = Counter((row['call_kind'], row['call_status']) for row in calls)
    reported = {(kind, status): count for kind, states in policy['calls'].items() for status, count in states.items() if count}
    require(observed == reported and all(appointments.get(key, 0) == value for key, value in policy['appointments'].items()),
            'simulator outcome counts differ from the run report')

    selected = [row for row in calls if row['call_id'] in SAMPLE_CALLS]
    require(len(selected) == len(SAMPLE_CALLS), 'selected calls missing from simulator database')
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        samples = list(pool.map(lambda case: collect_case(case, run, database, histories.get(case['call_id'], [])), selected))
    audits = [audit for audit, _ in samples]
    events = [event for _, trace in samples for event in trace]
    graph = build_execution_graph(events)
    require(not graph['issues'], 'collected trace graph has unresolved references')
    atomic_write_json(args.output / 'events.json', {'events': events})
    atomic_write_json(args.output / 'outcomes.json', {'run': run, 'database_source': database, 'policy_source': policy_source,
                                                      'outcome_sql': OUTCOME_SQL, 'calls': calls, 'appointments': appointments,
                                                      'sampled_traces': audits})
    (args.output / 'audit.md').write_text(report(run, calls, appointments, audits), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'indexed_calls': len(calls), 'traces': len(audits),
                      'events': len(events), 'no_shows': appointments['no_show']}, indent=2))


if __name__ == '__main__':
    main()
