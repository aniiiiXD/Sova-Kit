"""Collect and audit the published EnterpriseOps-Gym Teams sample."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from pathlib import Path
from urllib.request import urlopen
import json
import re
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from switchit.contracts import require
from switchit.data_paths import kit_data_dir
from switchit.events import normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.fixtures import build_fixtures
from switchit.redaction import redact_literal, redact_literal_value
from switchit.storage import atomic_write_json


SAMPLE_URL = 'https://benchmarkrouter.newmeasure.ai/benchmarks/enterpriseops_gym/sample'
TASK_URL = ('https://datasets-server.huggingface.co/rows?dataset=ServiceNow-AI%2FEnterpriseOps-Gym'
            '&config=oracle&split=teams&offset=0&length=1')
TASK_ID = 'task_20251121_102744_757_7ebc1127_dadb0c94'
TASK_REVISION = 'c8e538eae8a6205294f0a86675fefdc1fac408f6'
TURN = re.compile(r'<span class="font-medium text-muted-foreground">Turn (\d+)</span>')
ROLE = re.compile(r'<p[^>]*>(Model|Tool)</p>')
DETAIL = re.compile(r'<details\b[^>]*>.*?<summary\b[^>]*>(.*?)</summary>.*?'
                    r'<pre\b[^>]*>(.*?)</pre>.*?</details>', re.S)
GAPS = {
    'verify_james_has_zero_townhalls': 'The cancellation receipt gives no final town hall state.',
    'verify_james_zero_active_webinars': 'The cancellation receipt gives no final webinar state.',
    'verify_bob_chat_members': 'The chat receipt gives no final member rows.',
    'verify_carol_chat_members': 'The chat receipt gives no final member rows.',
}


def fetch(url):
    with urlopen(url, timeout=30) as response:
        raw = response.read()
        source = {'url': response.geturl(), 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}
        if revision := response.headers.get('X-Revision'):
            source['revision'] = revision
    return raw.decode('utf-8'), source


def visible_text(markup):
    markup = re.sub(r'<li\b[^>]*>', '- ', markup)
    markup = re.sub(r'</(?:p|li|h[1-6])>|<br\s*/?>', '\n', markup)
    return unescape(re.sub(r'<[^>]*>', '', markup)).strip()


def read_sample(page, task):
    request, rest = re.split(r'<h2\b[^>]*>Transcript</h2>', page, maxsplit=1)
    transcript, grading = re.split(r'<h2\b[^>]*>Grading</h2>', rest, maxsplit=1)
    published_prompt = visible_text(request.split('>User</p>', 1)[1])
    require(published_prompt.split() == task['user_prompt'].split(), 'published user prompt mismatch')
    heading, grade_pre = grading.split('<pre', 1)
    require(TASK_ID in heading, 'published task ID mismatch')
    grade = json.loads(unescape(grade_pre.split('>', 1)[1].split('</pre>', 1)[0]))
    require(len(grade) == 1, 'expected a single published grade')
    grade = grade[0]

    turns = []
    parts = TURN.split(transcript)
    for number, fragment in zip(parts[1::2], parts[2::2], strict=True):
        expected_roles = ['Model'] if not turns else ['Tool', 'Model']
        sections = ROLE.split(fragment)
        require(int(number) == len(turns) + 1 and sections[1::2] == expected_roles,
                'published turn structure changed')
        header = visible_text(sections[0])
        model = re.match(r'(.+?)([\d,]+) in · ([\d,]+) out$', header)
        require(model is not None, 'published model usage missing')
        model_name, input_tokens, output_tokens = model.groups()
        calls, receipts = [], []
        for role, block in zip(sections[1::2], sections[2::2], strict=True):
            details = DETAIL.findall(block)
            require(len(details) == len(re.findall(r'<details\b', block)), 'unparsed published details')
            for label, pre in details:
                payload = json.loads(unescape(pre))
                if role == 'Tool':
                    require(isinstance(payload.get('isError'), bool) and isinstance(payload.get('content'), list),
                            'invalid published receipt')
                    receipts.append({'response': payload, 'source_sha256': sha256(unescape(pre).encode()).hexdigest()})
                else:
                    name = re.search(r'<span class="font-mono">([^<]+)</span>', label)
                    require(name is not None, 'published tool name missing')
                    calls.append({'name': unescape(name[1]), 'arguments': payload})
        model_text = visible_text(DETAIL.sub('', sections[-1]))
        turns.append({'turn': int(number), 'model': model_name,
                      'usage': {'input_tokens': int(input_tokens.replace(',', '')),
                                'output_tokens': int(output_tokens.replace(',', ''))},
                      'text': model_text, 'calls': calls, 'receipts': receipts})
    calls = sum(len(turn['calls']) for turn in turns)
    receipts = sum(len(turn['receipts']) for turn in turns)
    require(len(turns) == grade['llm_turns'] and calls == receipts == grade['tool_calls']
            and turns[-1]['text'], 'published transcript count mismatch')
    return turns, grade


def make_events(turns, task, sources, grade):
    trace_id = f"enterpriseops-gym/{TASK_ID}/run-{grade['run']}/{sources['transcript']['sha256'][:12]}"
    source = {'importer': 'benchmark-router', **sources['transcript'], 'task_id': TASK_ID}
    events, links = [], []
    rows = {'chats': [], 'chat_messages': []}

    def emit(locator, kind, payload, *, origin=source, parent=None, uncertainty=()):
        event = normalize_event({'event_id': f'{trace_id}/{locator}', 'kind': kind,
                                 'trace_id': trace_id, 'index': len(events),
                                 'source': {**origin, 'locator': locator}, 'payload': payload,
                                 'parent_id': parent, 'uncertainty': list(uncertainty)})
        events.append(event)
        return event['event_id']

    emit('user', 'user_input', {'role': 'user', 'content': task['user_prompt']},
         origin={'importer': 'enterpriseops-gym', **sources['task'], 'task_id': TASK_ID})
    previous_calls = []
    for turn in turns:
        number = turn['turn']
        require(len(turn['receipts']) == len(previous_calls), 'receipt batch differs from preceding call batch')
        for position, ((call, call_id), item) in enumerate(zip(previous_calls, turn['receipts'], strict=True), 1):
            response = item['response']
            blocks = response['content']
            # Decode MCP's nested JSON so canonical redaction can preserve field structure.
            content = []
            for block in blocks:
                value = block.get('text')
                if isinstance(value, str) and value.lstrip().startswith(('{', '[')):
                    content.append({**block, 'type': 'json', 'data': json.loads(value), 'text': None})
                else:
                    content.append(block)
            receipt_id = emit(f'turn-{number}/result-{position}', 'tool_result',
                              {'content': content, 'error': response['isError'], 'turn': number,
                               'source_sha256': item['source_sha256']},
                              uncertainty=('tool_call_identity_missing', 'call_result_link_inferred'))
            links.append({'call': call_id, 'receipt': receipt_id, 'tool': call['name'],
                          'evidence': 'inferred', 'method': 'position in the next published turn'})
            if response['isError'] or len(content) != 1 or content[0].get('type') != 'json':
                continue
            data = content[0]['data']
            if call['name'] == 'create_chat':
                rows['chats'].append({'chat_id': data['id'], 'chat_type': data['chatType'], 'receipt': receipt_id})
            if call['name'] == 'send_chat_message':
                require(data['chatId'] == call['arguments']['chatId'] and data['body'] == call['arguments']['body'],
                        'message receipt does not match the requested chat and body')
                # The saved projection must retain distinct keys after phone-like IDs are masked.
                message_id = sha256(data['id'].encode()).hexdigest()
                rows['chat_messages'].append({'message_id': message_id, 'chat_id': data['chatId'],
                                              'body_json': json.dumps(data['body'], ensure_ascii=False),
                                              'receipt': receipt_id})
        model_id = emit(f'turn-{number}/model', 'model_call',
                        {'turn': number, 'model': turn['model'], 'usage': turn['usage'],
                         'response': {'text': turn['text'], 'tool_calls': turn['calls']}},
                        uncertainty=('provider_request_missing',))
        previous_calls = []
        for position, call in enumerate(turn['calls'], 1):
            call_id = emit(f'turn-{number}/call-{position}', 'tool_call', {**call, 'turn': number},
                           parent=model_id, uncertainty=('tool_call_identity_missing',))
            previous_calls.append((call, call_id))
    require(not previous_calls, 'sample ends before receipts for final tool calls')
    return events, links, rows


def audit_checks(verifiers, rows):
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('CREATE TABLE chats (chat_id TEXT PRIMARY KEY, chat_type TEXT)')
        db.execute('CREATE TABLE chat_messages (message_id TEXT PRIMARY KEY, chat_id TEXT, body_json TEXT)')
        db.executemany('INSERT INTO chats VALUES (:chat_id, :chat_type)', rows['chats'])
        db.executemany('INSERT INTO chat_messages VALUES (:message_id, :chat_id, :body_json)', rows['chat_messages'])
        db.execute('PRAGMA query_only = ON')

        def authorize(action, first, second, _database, _trigger):
            allowed = (action == sqlite3.SQLITE_SELECT or
                       (action == sqlite3.SQLITE_READ and first in rows) or
                       (action == sqlite3.SQLITE_FUNCTION and second in {'count', 'lower', 'like'}))
            return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        results = []
        for verifier in verifiers:
            name = verifier['name']
            config = verifier['validation_config']
            require(verifier['verifier_type'] == 'database_state' and config['comparison_type'] == 'equals',
                    'unexpected benchmark verifier contract')
            result = {'name': name, 'expected': config['expected_value']}
            if name in GAPS:
                result.update(status='insufficient_evidence', reason=GAPS[name])
            else:
                query = config['query']
                table = re.search(r'\bFROM\s+(chats|chat_messages)\b', query, re.I)
                chat = re.search(r"\bchat_id\s*=\s*'([^']+)'", query)
                require(table is not None and chat is not None, 'unrecognized receipt verifier')
                values = db.execute(query).fetchall()
                require(len(values) == 1 and len(values[0]) == 1, 'expected scalar verifier result')
                observed = values[0][0]
                result.update(observed=observed, status='receipt_pass' if observed == result['expected'] else 'receipt_fail',
                              receipts=[row['receipt'] for row in rows[table[1].lower()] if row['chat_id'] == chat[1]])
            results.append(result)
    return results


def report(evidence):
    grade, checks = evidence['published_grade'], evidence['receipt_checks']
    coverage = Counter(check['status'] for check in checks)
    links = evidence['call_receipt_links']
    lines = [
        '# EnterpriseOps-Gym Teams trace audit', '',
        f"Task `{TASK_ID}` · published run {grade['run']} · model `{evidence['transcript'][0]['model']}`.", '',
        f"The transcript records {grade['llm_turns']} model turns, {len(links)} tool calls and {len(links)} receipts. "
        f"The source reports **{grade['verifiers_passed']}/{grade['verifiers_total']}** checks passed. "
        f"The original SQL returned the expected value for **{coverage['receipt_pass']}/{len(checks)}** checks "
        'on the rows contained in the receipts.', '',
        '| Check | Expected | Receipt check |', '| --- | --- | --- |',
    ]
    for check in checks:
        outcome = check.get('reason') or f"{check['observed']} ({check['status'].replace('_', ' ')})"
        lines.append(f"| `{check['name']}` | {check['expected']} | {outcome} |")
    lines += ['', 'The SQLite rows come from the two chat creation receipts and two message receipts. '
              'The projection uses hashes for message IDs and preserves the body fields used by the SQL. '
              'Global counts require the final database.', '',
              'Call-to-receipt links follow published order and are marked inferred. The source provides no original tool call IDs. '
              f"SwitchIt produced **{evidence['exact_fixture_count']} exact replay fixtures**. "
              'Use this run for importer development. A final database or a fully captured run is needed for training labels.', '',
              '| Tool call | Locally paired receipt |', '| --- | --- |']
    for link in links:
        call = link['call'].split('/turn-', 1)[1]
        receipt = link['receipt'].split('/turn-', 1)[1]
        lines.append(f"| `{link['tool']}` · `turn-{call}` | `turn-{receipt}` |")
    lines += ['', 'The task and original SQL, sanitized transcript, receipt projection, and source hashes are in [evidence.json](evidence.json). '
              'The canonical SwitchIt events are in [events.json](events.json).', '',
              f'[Published run]({SAMPLE_URL}) · [Versioned task dataset]'
              f'(https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym/tree/{TASK_REVISION})', '']
    return '\n'.join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=kit_data_dir(ROOT.parent) / 'traces' / 'enterpriseops-gym' / TASK_ID)
    output = parser.parse_args().output
    page, transcript_source = fetch(SAMPLE_URL)
    dataset, task_source = fetch(TASK_URL)
    require(task_source.get('revision') == TASK_REVISION, 'upstream dataset revision differs from the pinned revision')
    record = json.loads(dataset)['rows'][0]
    task = record['row']
    require(record['row_idx'] == 0 and not record['truncated_cells'] and task['task_id'] == TASK_ID,
            'upstream task row missing or truncated')
    verifiers = json.loads(task['verifiers'])
    turns, grade = read_sample(page, task)
    require(len(verifiers) == grade['verifiers_total'], 'upstream verifier count mismatch')
    sources = {'transcript': transcript_source, 'task': {**task_source, 'config': 'oracle', 'split': 'teams', 'row_idx': 0}}
    events, links, rows = make_events(turns, task, sources, grade)
    checks = audit_checks(verifiers, rows)
    graph = build_execution_graph(events)
    fixtures = build_fixtures(graph)
    evidence = redact_literal_value({
        'collected_at': datetime.now(timezone.utc).isoformat(),
        'sources': sources,
        'task': {key: task[key] for key in ('task_id', 'domain', 'user_prompt', 'selected_tools',
                                           'restricted_tools', 'reset_database_between_runs')},
        'seed_databases': [{key: db[key] for key in ('mcp_server_name', 'seed_database_file')}
                           for db in json.loads(task['gym_servers_config'])],
        'transcript': turns, 'published_grade': grade, 'verifiers': verifiers,
        'call_receipt_links': links, 'receipt_projection': rows, 'receipt_checks': checks,
        'graph_issues': graph['issues'], 'exact_fixture_count': len(fixtures['fixtures']),
    })
    evidence['task']['system_prompt'] = redact_literal(task['system_prompt'], limit=len(task['system_prompt']))
    require(audit_checks(evidence['verifiers'], evidence['receipt_projection']) == checks,
            'saved receipt projection differs from the source check results')
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / 'events.json', {'events': events})
    atomic_write_json(output / 'evidence.json', evidence)
    (output / 'audit.md').write_text(report(evidence), encoding='utf-8')
    print(json.dumps({'output': str(output), 'events': len(events),
                      'receipt_checks': Counter(check['status'] for check in checks),
                      'exact_fixtures': len(fixtures['fixtures'])}, indent=2))


if __name__ == '__main__':
    main()
