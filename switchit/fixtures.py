"""Exact argument/occurrence fixtures; never reuse a result by tool name alone."""
from copy import deepcopy
from collections import defaultdict
from switchit.contracts import fingerprint, require, version


def build_fixtures(graph, trace_ids=None):
    version(graph, 'graph_version')
    nodes = {e['event_id']: e for e in graph['nodes']}
    links = defaultdict(list)
    for edge in graph['edges']:
        if edge['relation'] == 'tool_result' and edge['evidence'] == 'recorded':
            links[edge['source']].append(edge['target'])
    fixtures, issues, counts = [], [], defaultdict(int)
    for event in graph['nodes']:
        if event['kind'] != 'tool_call' or (trace_ids is not None and event['trace_id'] not in trace_ids):
            continue
        payload = event['payload']
        key = (event['trace_id'], payload.get('name'), fingerprint(payload.get('arguments', {})))
        occurrence = counts[key]
        counts[key] += 1
        results = links[event['event_id']]
        if len(results) != 1:
            issues.append({'event_id': event['event_id'], 'reason': 'missing_or_ambiguous_result'})
            continue
        result = nodes[results[0]]
        if result['payload'].get('error') or 'payload_truncated' in result['uncertainty']:
            issues.append({'event_id': event['event_id'], 'reason': 'failed_or_truncated_result'})
            continue
        fixtures.append({'fixture_id': fingerprint([event['event_id'], result['event_id']])[:24],
                         'trace_id': event['trace_id'], 'tool_call_id': event.get('tool_call_id'),
                         'name': payload.get('name'), 'arguments': payload.get('arguments', {}),
                         'occurrence': occurrence, 'result': result['payload'].get('content'),
                         'evidence': [event['event_id'], result['event_id']]})
    return {'fixtures_version': 1, 'fixtures': fixtures, 'issues': issues}


class FixtureSession:
    def __init__(self, document, trace_id=None):
        version(document, 'fixtures_version')
        self.fixtures = deepcopy([f for f in document['fixtures'] if trace_id is None or f['trace_id'] == trace_id])
        require(len({f['fixture_id'] for f in self.fixtures}) == len(self.fixtures), 'duplicate fixture IDs')
        self.counts = defaultdict(int)
        self.used = []

    def resolve(self, name, arguments):
        key = (name, fingerprint(arguments))
        occurrence = self.counts[key]
        self.counts[key] += 1
        matches = [f for f in self.fixtures if f['name'] == name and f['arguments'] == arguments
                   and f['occurrence'] == occurrence and f['fixture_id'] not in self.used]
        if len(matches) != 1:
            return {'status': 'missing' if not matches else 'ambiguous', 'result': None}
        fixture = matches[0]
        self.used.append(fixture['fixture_id'])
        return {'status': 'matched', 'result': deepcopy(fixture['result']), 'fixture_id': fixture['fixture_id']}
