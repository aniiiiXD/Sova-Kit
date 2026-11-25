"""Canonical evidence events, preserving explicit identities and missing information."""
from __future__ import annotations

from copy import deepcopy
from switchit.contracts import fingerprint, require, version
from switchit.redaction import redact_value, redact_literal_value

KINDS = {'user_input', 'model_call', 'tool_call', 'tool_result', 'artifact',
         'agent_start', 'agent_end', 'approval', 'state', 'handoff'}
RELATION_FIELDS = ('parent_id', 'parent_span_id', 'retry_of', 'handoff_from',
                   'agent_id', 'parent_agent_id', 'span_id', 'tool_call_id')


def normalize_event(raw: dict, source: dict | None = None) -> dict:
    require(isinstance(raw, dict), 'event must be an object')
    require(raw.get('event_version', 1) == 1, 'unsupported event version')
    require(raw.get('kind') in KINDS, f'unsupported event kind: {raw.get("kind")}')
    require(isinstance(raw.get('trace_id'), str) and bool(raw['trace_id']), 'trace_id is required')
    index = raw.get('index', 0)
    require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, 'index must be a nonnegative integer')
    provenance = deepcopy(source if source is not None else raw.get('source', {}))
    require(isinstance(provenance, dict) and provenance, 'source provenance is required')
    payload = raw.get('payload', {})
    require(isinstance(payload, dict), 'payload must be an object')
    event_id = raw.get('event_id') or fingerprint([provenance, raw['trace_id'], raw.get('record_id'), index, raw['kind']])[:32]
    require(isinstance(event_id, str) and bool(event_id), 'event_id must be a nonempty string')
    response_payload = payload.get('response')
    response_calls = response_payload.get('tool_calls', []) if isinstance(response_payload, dict) else []
    custom_payload = payload.get('tool_format') == 'custom' or any(
        isinstance(call, dict) and call.get('type') == 'custom' for call in response_calls)
    literal_payload = custom_payload or provenance.get('importer') == 'switchup'
    cleaned_payload = redact_literal_value(payload) if literal_payload else redact_value(payload)
    result = {'event_version': 1, 'event_id': event_id, 'trace_id': raw['trace_id'],
              'session_id': raw.get('session_id'), 'record_id': raw.get('record_id'),
              'kind': raw['kind'], 'index': index, 'timestamp': raw.get('timestamp'),
              'source': redact_value(provenance), 'payload': cleaned_payload,
              'evidence': raw.get('evidence', 'recorded'), 'uncertainty': list(raw.get('uncertainty', []))}
    require(result['evidence'] in ('recorded', 'inferred', 'metadata_only'), 'invalid evidence level')
    for field in RELATION_FIELDS:
        result[field] = str(raw[field]) if raw.get(field) is not None else None
    if raw.get('timestamp') is None:
        result['uncertainty'].append('timestamp_missing')
    # Existing redaction bounds long strings. Never present an elided payload as exact.
    def elided(value):
        if isinstance(value, str):
            return ' […] ' in value
        if isinstance(value, dict):
            return any(elided(v) for v in value.values())
        if isinstance(value, list):
            return any(elided(v) for v in value)
        return False
    if elided(result['payload']):
        result['uncertainty'].append('payload_truncated')
    result['uncertainty'] = sorted(set(result['uncertainty']))
    return result


def validate_event(event):
    version(event, 'event_version')
    normalized = normalize_event(event)
    require(normalized == event, 'event is not canonical; normalize it before storing')
    return event


def events_from_calls(calls: list) -> list:
    """Bridge legacy calls to evidence events without fabricating causal edges."""
    events, seen_results = [], set()
    for call in calls:
        trace = call['trace_id']
        response, request = call.get('response') or {}, call.get('request') or {}
        base = {'trace_id': trace, 'session_id': call.get('session_id'),
                'record_id': call.get('source_record_id') or call['call_id'], 'index': call.get('call_index', 0),
                'timestamp': call.get('timestamp'), 'source': call.get('source') or {'importer': 'legacy'},
                **{key: call.get(key) for key in RELATION_FIELDS}}
        identity = fingerprint([trace, call['call_id']])[:24]
        messages = request.get('messages') or []
        user_only = call.get('model') == 'user-turn' or (not response.get('text') and not response.get('tool_calls')
                    and any(m.get('role') == 'user' for m in messages))
        tool_only = call.get('model') == 'tool-result'
        if not tool_only:
            events.append(normalize_event({**base, 'event_id': identity, 'kind': 'user_input' if user_only else 'model_call',
                'payload': {'request': request, 'response': response, 'model': call.get('model'),
                            'provider': call.get('provider'), 'usage': call.get('usage'),
                            'latency_ms': call.get('latency_ms'), **call.get('context', {})},
                'evidence': 'metadata_only' if call.get('model') in ('cloud-thread', 'devin-session', 'cloud-agent') else 'recorded'}))
        for position, tool in enumerate(response.get('tool_calls') or []):
            events.append(normalize_event({**base, 'event_id': f'{identity}:tool:{position}', 'kind': 'tool_call',
                'parent_id': identity, 'tool_call_id': tool.get('id') or tool.get('call_id'),
                'payload': {'name': tool.get('name'), 'arguments': tool.get('arguments') or {},
                            **({'tool_format': 'custom', 'input_format': tool.get('input_format'),
                                'input': tool.get('input')} if tool.get('type') == 'custom' else {})},
                'uncertainty': (['custom_tool_requires_adapter'] +
                                (['custom_tool_input_missing'] if tool.get('input_format') != 'freeform' else []) +
                                (['tool_call_identity_missing'] if not tool.get('id') else []))
                                if tool.get('type') == 'custom' else []}))
        for position, message in enumerate(messages):
            if message.get('role') != 'tool':
                continue
            tool_id = message.get('tool_call_id')
            key = (trace, tool_id, fingerprint(message.get('content')))
            if tool_id and key in seen_results:
                continue  # Repeated conversation history is not a repeated execution.
            seen_results.add(key)
            events.append(normalize_event({**base, 'event_id': f'{identity}:result:{position}', 'kind': 'tool_result',
                'tool_call_id': tool_id, 'payload': {'name': message.get('name'), 'content': message.get('content'),
                                                  'error': message.get('error'),
                                                  **({key: message[key] for key in ('tool_format', 'output_format', 'content_blocks')
                                                      if key in message} if message.get('tool_format') == 'custom' else {})},
                'uncertainty': [] if tool_id else ['tool_call_identity_missing']}))
    return events
