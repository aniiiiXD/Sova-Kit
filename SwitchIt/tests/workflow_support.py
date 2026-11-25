"""Synthetic, credential-free evidence shared by module contract tests."""
from switchit.events import normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.workflow_patterns import discover_workflows
from switchit.workflow_builder import build_workflow, review_workflow
from switchit.workflow_cases import build_workflow_cases


def event(eid, kind='model_call', trace='trace-a', index=0, **fields):
    return normalize_event({'event_id': eid, 'trace_id': trace, 'session_id': trace, 'index': index,
                            'kind': kind, 'source': {'importer': 'synthetic', 'location': 'fixture'}, **fields})


def evidence():
    events = []
    for trace in ('trace-a', 'trace-b'):
        events.extend([
            event(trace+'-user', 'user_input', trace, 0, payload={'text': 'Look up account status'}),
            event(trace+'-model', trace=trace, index=1, payload={'response': {'text': 'Looking up account'},
                   'usage': {'input_tokens': 10, 'output_tokens': 2, 'basis': 'recorded'}}),
            event(trace+'-call', 'tool_call', trace, 2, tool_call_id='lookup-1', parent_id=trace+'-model',
                  payload={'name': 'lookup', 'arguments': {'account': 'A'}}),
            event(trace+'-result', 'tool_result', trace, 3, tool_call_id='lookup-1', payload={'content': 'active'}),
            event(trace+'-final', trace=trace, index=4, payload={'response': {'text': 'Account is active'},
                   'usage': {'input_tokens': 14, 'output_tokens': 4, 'basis': 'recorded'}}),
        ])
    return events


def pipeline():
    events = evidence()
    graph = build_execution_graph(events)
    candidate = discover_workflows(graph)['candidates'][0]
    draft = build_workflow(candidate, graph)
    spec = review_workflow(draft, reviewer='fixture-owner', success_checks=[
        {'type': 'no_error'}, {'type': 'output_contains', 'text': 'active'},
        {'type': 'tool_sequence', 'calls': [{'name': 'lookup', 'arguments': {'account': 'A'}}]},
    ])
    suite = build_workflow_cases(spec, candidate, graph)
    return events, graph, candidate, spec, suite
