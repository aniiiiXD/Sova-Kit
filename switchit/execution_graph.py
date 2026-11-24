"""Reconstruct evidence-backed relationships; chronology alone is not causality."""
from collections import defaultdict

from switchit.contracts import require, topological_order
from switchit.events import validate_event


def build_execution_graph(events):
    nodes, issues, edges = {}, [], []
    successors = defaultdict(set)
    records, spans, tools, agents = (defaultdict(list) for _ in range(4))
    for event in events:
        validate_event(event)
        eid = event['event_id']
        require(eid not in nodes, f'duplicate event ID: {eid}')
        nodes[eid] = event
        if event.get('record_id') and event['kind'] not in ('tool_call', 'tool_result'):
            records[(event['trace_id'], event['record_id'])].append(eid)
        if event.get('span_id') and event['kind'] not in ('tool_call', 'tool_result'):
            spans[(event['trace_id'], event['span_id'])].append(eid)
        if event['kind'] == 'tool_call' and event.get('tool_call_id'):
            tools[(event['trace_id'], event['tool_call_id'])].append(eid)
        if event['kind'] == 'agent_start' and event.get('agent_id'):
            agents[(event.get('session_id'), event['agent_id'])].append(eid)

    def link(candidates, target, relation, evidence='recorded', causal=None):
        if len(candidates) != 1:
            issues.append({'event_id': target, 'relation': relation,
                           'reason': 'missing_reference' if not candidates else 'ambiguous_reference'})
            return
        source = candidates[0]
        edge = {'source': source, 'target': target, 'relation': relation, 'evidence': evidence}
        if causal is not None:
            edge['causal'] = causal
        if edge in edges:
            return
        pending, visited = [target], set()
        while pending:
            current = pending.pop()
            if current == source:
                issues.append({'event_id': target, 'relation': relation, 'reason': 'cycle'})
                return
            if current not in visited:
                visited.add(current)
                pending.extend(successors[current])
        edges.append(edge)
        successors[source].add(target)

    for event in events:
        eid, trace = event['event_id'], event['trace_id']
        for field, relation in (('parent_id', 'parent'), ('retry_of', 'retry'), ('handoff_from', 'handoff')):
            value = event.get(field)
            if value:
                link([value] if value in nodes else records[(trace, value)], eid, relation)
        if event['source'].get('importer') == 'github' and event['evidence'] == 'metadata_only':
            for reference in event['payload'].get('github_references', []):
                require(isinstance(reference, dict) and reference.get('relation') in ('tested_commit', 'references_pull_request'),
                        'invalid GitHub reference relation')
                value = reference.get('event_id')
                link([value] if value in nodes else [], eid, reference['relation'], causal=False)
        if event.get('parent_span_id'):
            link(spans[(trace, event['parent_span_id'])], eid, 'parent_span')
        if event.get('parent_agent_id'):
            link(agents[(event.get('session_id'), event['parent_agent_id'])], eid, 'subagent')
        if event['kind'] == 'tool_result':
            link(tools[(trace, event.get('tool_call_id'))], eid, 'tool_result')
    from .evidence_correlation import correlate_github
    correlation = correlate_github(events)
    for reference in correlation['links']:
        # Point from the referenced artifact to the mentioning event for display.
        # The direction is not causal and never becomes a workflow dependency.
        link([reference['target_event_id']], reference['source_event_id'], 'references_github', causal=False)
    traces = defaultdict(list)
    for event in events:
        traces[event['trace_id']].append(event['event_id'])
    # Sequence is an observed/inferred ordering edge, never labeled a dependency.
    positions = {event["event_id"]: index for index, event in enumerate(events)}
    for ids in traces.values():
        ids.sort(key=lambda key: (nodes[key]['index'], positions[key]))
        for left, right in zip(ids, ids[1:]):
            if all(nodes[key]['source'].get('importer') == 'github' for key in (left, right)):
                continue  # API collection order is not observed execution order.
            link([left], right, 'sequence', 'inferred')
    return {'graph_version': 1, 'correlation': correlation, 'nodes': list(nodes.values()), 'edges': edges,
            'traces': dict(traces), 'issues': issues,
            'order': topological_order(nodes, [(e['source'], e['target']) for e in edges]),
            'coverage': {'events': len(nodes), 'explicit_edges': sum(e['evidence'] == 'recorded' for e in edges),
                         'unresolved': len(issues)}}
