"""Live model routing and promotion-independent replay case capture."""
from collections import OrderedDict
from datetime import datetime, timezone

from switchit.contracts import ContractError, fingerprint, require
from switchit.runtime_adapters import model_request

from .contracts import MAX_CASES, live_timeout, model_adapter, model_input, record_id, scrub_evidence
from .store import capture_event


def invoke(workspace, project, payload, *, allow_live, timeout=45):
    require(allow_live, 'explicit live model authorization is required')
    live_timeout(timeout)
    require(isinstance(payload, dict) and set(payload) <= {'request_id', 'trace_id', 'session_id', 'request'},
            'invocation accepts request_id, trace_id, session_id and request')
    request_id = record_id(payload.get('request_id'), 'request_id')
    trace_id = record_id(payload.get('trace_id'), 'trace_id')
    session_id = record_id(payload.get('session_id', trace_id), 'session_id')
    request = model_input(payload.get('request'))
    current = workspace.begin_invocation(project, request_id)
    recipe, connection = current['active_recipe'], current['connection']
    model_event = fingerprint([project, request_id, 'model'])[:32]
    at = datetime.now(timezone.utc).isoformat()
    status, events = 'failed', []
    try:
        adapter = model_adapter(connection, allow_live=allow_live, timeout=timeout)
        effective = model_request(request, recipe)
        result = adapter.replay(effective)
        require(isinstance(result, dict), 'provider returned an invalid result')
        calls = result.get('tool_calls', [])
        require(isinstance(calls, list) and all(isinstance(call, dict) and isinstance(call.get('name'), str)
                and isinstance(call.get('arguments', {}), dict) for call in calls), 'provider returned invalid tool proposals')
        calls = [{**call, 'id': call.get('id') or f'call_{fingerprint([model_event, index])[:24]}'}
                 for index, call in enumerate(calls)]
        allowed = {tool['name'] for tool in effective.get('tools', [])}
        if any(call['name'] not in allowed for call in calls):
            result['error'] = 'provider proposed a tool outside the request and recipe allowlist'
        result['tool_calls'] = calls
        status = 'failed' if result.get('error') else 'completed'
    except Exception as error:
        result = {'text': '', 'tool_calls': [], 'error': f'{type(error).__name__}: {error}', 'model': connection['model']}
    try:
        base = {'trace_id': trace_id, 'session_id': session_id, 'record_id': request_id, 'timestamp': at,
                'source': {'importer': 'switchup', 'project': project}}
        events.append(capture_event({**base, 'event_id': model_event, 'kind': 'model_call',
            'payload': {'request': request, 'response': result, 'model': connection['model'],
                        'provider': connection['kind'], 'usage': result.get('usage'),
                        'latency_ms': result.get('latency_ms'), 'recipe_id': recipe['recipe_id'],
                        'generation': current['generation']}}))
        for index, call in enumerate(result['tool_calls'], 1):
            events.append(capture_event({**base, 'event_id': f'{model_event}:tool:{index}', 'index': index,
                'kind': 'tool_call', 'parent_id': model_event, 'tool_call_id': call['id'],
                'payload': {'name': call['name'], 'arguments': call.get('arguments', {})}}))
    except Exception:
        workspace.finish_invocation(project, request_id, [], 'capture_failed')
        raise
    workspace.finish_invocation(project, request_id, events, status)
    if status == 'failed':
        result = {**scrub_evidence(result), 'tool_calls': []}
    return {'project': project, 'request_id': request_id, 'event_id': model_event,
            'trace_id': trace_id, 'session_id': session_id, 'generation': current['generation'],
            'recipe_id': recipe['recipe_id'], 'status': status, 'result': result}


def _replayable(event):
    if (event['kind'] != 'model_call' or event['evidence'] != 'recorded'
            or 'payload_truncated' in event['uncertainty']):
        return False
    try:
        model_input(event['payload'].get('request'))
    except ContractError:
        return False
    return True


def chat_suite(workspace, project, holdout_traces, *, max_cases=30):
    """Seal the latest usable request in each selected session as an independent case."""
    spec = project['workflow']
    require(spec.get('tool_proposals_as_output') and spec['steps'][0].get('request') == {'$ref': 'inputs.request'},
            'use a reviewed suite file for a multi-step workflow')
    require(type(max_cases) is int and 2 <= max_cases <= MAX_CASES, f'max_cases must be between 2 and {MAX_CASES}')
    require(isinstance(holdout_traces, list) and 0 < len(holdout_traces) < max_cases,
            'supply holdout trace IDs and leave room for training cases')
    holdout_ids = {record_id(trace, 'holdout trace ID') for trace in holdout_traces}
    heldout = {}
    selected_traces = set()
    for event in workspace.events(project['name']):
        if event['trace_id'] in holdout_ids and _replayable(event):
            heldout[event['session_id'] or event['trace_id']] = event
            selected_traces.add(event['trace_id'])
    require(selected_traces == holdout_ids, 'each holdout trace needs a complete captured model request')
    train = OrderedDict()
    train_limit = max_cases - len(heldout)
    for event in workspace.events(project['name'], exclude_holdout=True):
        group = event['session_id'] or event['trace_id']
        if group in heldout or event['trace_id'] in holdout_ids or not _replayable(event):
            continue
        train[group] = event
        train.move_to_end(group)
        if len(train) > train_limit:
            train.popitem(last=False)
    require(train, 'capture at least one training session separate from the holdout sessions')
    cases = []
    for split, events in (('train', train), ('holdout', heldout)):
        for group, event in events.items():
            cases.append({'case_id': fingerprint([spec['revision'], event['event_id']])[:24],
                          'workflow_id': spec['workflow_id'], 'workflow_revision': spec['revision'],
                          'trace_id': event['trace_id'], 'split_group': group, 'split': split,
                          'inputs': {'request': event['payload']['request']}, 'checks': spec['success_checks'],
                          'source_event_id': event['event_id']})
    return {'workflow_cases_version': 1, 'workflow_revision': spec['revision'], 'cases': cases,
            'uncertainty': ['single_request_replay']}
