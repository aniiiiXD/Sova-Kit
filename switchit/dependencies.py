"""Dependency manifests with evidence; observed tools are not inferred permissions."""
from switchit.contracts import fingerprint


def profile_dependencies(events):
    entries = {}
    unknown = set()
    def add(kind, value, event, confidence='recorded'):
        key = (kind, fingerprint(value))
        item = entries.setdefault(key, {'kind': kind, 'value': value, 'evidence': [], 'confidence': confidence})
        if event['event_id'] not in item['evidence']:
            item['evidence'].append(event['event_id'])
    for event in events:
        payload = event['payload']
        if event['kind'] == 'tool_call':
            add('tool', payload.get('name'), event)
        for tool in (payload.get('request') or {}).get('tools', []):
            add('tool_schema', tool, event)
        for message in (payload.get('request') or {}).get('messages', []):
            if message.get('role') == 'system':
                add('system_prompt', message.get('content'), event)
        for key, kind in (('skills', 'skill'), ('services', 'service'), ('permissions', 'permission'),
                          ('artifacts', 'artifact'), ('environment', 'environment')):
            value = payload.get(key)
            if value is None:
                continue
            for item in value if isinstance(value, list) else [value]:
                add(kind, item, event)
        if payload.get('model'):
            add('model', {'provider': payload.get('provider'), 'model': payload['model']}, event)
        if event['kind'] == 'artifact':
            add('artifact', payload, event)
    kinds = {entry['kind'] for entry in entries.values()}
    for kind in ('permission', 'environment', 'skill', 'service'):
        if kind not in kinds:
            unknown.add(f'{kind}_requirements_unobserved')
    return {'manifest_version': 1, 'dependencies': sorted(entries.values(), key=lambda x: (x['kind'], fingerprint(x['value']))),
            'unknown': sorted(unknown)}
