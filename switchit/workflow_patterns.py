"""Task segmentation and evidence-backed recurring workflow candidates."""
import re
from switchit.contracts import fingerprint, require, version


def _task(event):
    payload = event['payload']
    if payload.get('text'):
        return payload['text']
    return '\n'.join(m.get('content', '') for m in (payload.get('request') or {}).get('messages', [])
                     if m.get('role') == 'user')


def discover_workflows(graph, similarity=0.65):
    version(graph, 'graph_version')
    require(0 <= similarity <= 1, 'similarity must be between zero and one')
    nodes = {e['event_id']: e for e in graph['nodes']}
    segments = []
    for trace, ids in sorted(graph['traces'].items()):
        current = []
        for eid in ids:
            event = nodes[eid]
            if event['kind'] == 'user_input' and current and any(nodes[k]['kind'] != 'user_input' for k in current):
                segments.append(_segment(trace, current, nodes))
                current = []
            current.append(eid)
        if current:
            segments.append(_segment(trace, current, nodes))
    groups = []
    for segment in segments:
        tokens = set(re.findall(r'[a-z0-9_]+', segment['task'].lower()))
        selected = None
        for group in groups:
            reference = set(re.findall(r'[a-z0-9_]+', group['task'].lower()))
            overlap = len(tokens & reference) / len(tokens | reference) if tokens and reference else 0
            if overlap >= similarity and tokens and reference:
                selected = group
                break
        if selected is None:
            selected = {'candidate_id': 'workflow_' + fingerprint([segment['task'], segment['segment_id']])[:16],
                        'task': segment['task'], 'segments': [], 'variants': [], 'uncertainty': []}
            groups.append(selected)
        selected['segments'].append(segment)
        if segment['tool_sequence'] not in selected['variants']:
            selected['variants'].append(segment['tool_sequence'])
        selected['uncertainty'] = sorted(set(selected['uncertainty'] + segment['uncertainty']))
    return {'workflow_candidates_version': 1, 'candidates': groups,
            'notes': ['Task boundaries and similarity grouping are inferred and require review.']}


def _segment(trace, ids, nodes):
    events = [nodes[eid] for eid in ids]
    tasks = [_task(e) for e in events if e['kind'] == 'user_input']
    if not any(tasks):
        tasks = [_task(e) for e in events if _task(e)]
    task = next((t for t in tasks if t), '')
    return {'segment_id': fingerprint([trace, ids])[:24], 'trace_id': trace, 'event_ids': list(ids),
            'task': task, 'tool_sequence': [e['payload'].get('name') for e in events if e['kind'] == 'tool_call'],
            'failed': any(e['payload'].get('error') or (e['payload'].get('response') or {}).get('error') for e in events),
            'uncertainty': ([] if task else ['task_boundary_unknown']) +
                           (['metadata_only'] if all(e['evidence'] == 'metadata_only' for e in events) else [])}
