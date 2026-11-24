"""Versioned inputs for connected models and SwitchUP replay projects."""
from copy import deepcopy
import math
import os
import re
from urllib.parse import urlsplit

from switchit import providers
from switchit.composer import validate_suite
from switchit.contracts import fingerprint, require
from switchit.redaction import redact_literal_value
from switchit.workflow_builder import validate_workflow, workflow_revision

DEFAULT_STATE = '.switchit/platform/switchup'
MAX_CASES = 100
MAX_PROMPT_CHARS = 12_000
LOOPBACK = frozenset({'localhost', '127.0.0.1', '::1'})
SECRET_FIELDS = frozenset({'authorization', 'proxy_authorization', 'api_key', 'apikey', 'password',
                           'access_token', 'refresh_token', 'token', 'secret', 'client_secret',
                           'cookie', 'set_cookie', 'private_key'})


def project_name(value):
    require(isinstance(value, str) and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', value),
            'project must contain 1–63 lowercase letters, digits or hyphens')
    return value


def record_id(value, field):
    require(isinstance(value, str) and 0 < len(value) <= 200 and value == value.strip()
            and not any(ord(char) < 32 for char in value), f'{field} must be a nonempty identifier of at most 200 characters')
    return value


def live_timeout(timeout):
    require(type(timeout) in (float, int) and math.isfinite(timeout) and 0 < timeout <= 120,
            'model timeout must be greater than zero and at most 120 seconds')
    return timeout


def scrub_evidence(value):
    """Redact credential fields as well as secret patterns in text."""
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if key.lower().replace('-', '_') in SECRET_FIELDS else scrub_evidence(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_evidence(item) for item in value]
    return redact_literal_value(value)


def connection_settings(settings):
    require(isinstance(settings, dict), 'model connection must be an object')
    require(set(settings) <= {'kind', 'model', 'base_url', 'api_key_env', 'region', 'profile'},
            'unknown connection field; use a credential environment variable name')
    kind = settings.get('kind')
    require(isinstance(kind, str) and (kind in providers.LIVE_ADAPTERS or kind in providers.PROVIDER_PRESETS),
            'model connection requires a live adapter or a provider preset')
    model = record_id(settings.get('model'), 'model')
    key_env = settings.get('api_key_env')
    require(key_env is None or (isinstance(key_env, str) and re.fullmatch(r'[A-Z_][A-Z0-9_]*', key_env)),
            'api_key_env must name an environment variable')
    base_url = settings.get('base_url')
    if kind == 'bedrock':
        require(base_url is None and key_env is None, 'Bedrock uses AWS SDK credentials and endpoints')
        for field in ('region', 'profile'):
            if field in settings:
                record_id(settings[field], field)
    else:
        require(not {'region', 'profile'} & set(settings), 'region and profile apply to Bedrock')
    if base_url is not None:
        require(isinstance(base_url, str) and 0 < len(base_url) <= 600 and base_url == base_url.strip(),
                'base_url must be a URL')
        url = urlsplit(base_url)
        require(url.hostname and url.username is None and url.password is None and not url.query and not url.fragment
                and (url.scheme == 'https' or (url.scheme == 'http' and url.hostname in LOOPBACK)),
                'use an HTTPS endpoint or a loopback HTTP endpoint without embedded credentials')
        require(url.hostname in LOOPBACK or key_env, 'custom remote endpoints require an explicit api_key_env')
        require(kind != 'anthropic' or key_env, 'custom Anthropic endpoints require an explicit api_key_env')
    return {**deepcopy(settings), 'kind': kind, 'model': model}


def model_adapter(settings, *, allow_live, timeout):
    require(allow_live, 'explicit live model authorization is required')
    settings = connection_settings(settings)
    kind = settings['kind']
    config = {key: settings[key] for key in ('model', 'base_url', 'region', 'profile') if key in settings}
    config['timeout'] = live_timeout(timeout)
    if kind != 'bedrock':
        base_url = config.get('base_url')
        default_base = providers.PROVIDER_PRESETS.get(kind, {}).get('base_url')
        endpoint = base_url or default_base or ''
        key_env = settings.get('api_key_env')
        if key_env:
            config['api_key'] = os.environ.get(key_env)
            require(config['api_key'], f'set the {key_env} environment variable for this connection')
        elif not base_url and kind in providers.PROVIDER_PRESETS:
            config['api_key'] = providers.read_preset_key(kind)
        elif not base_url:
            from switchit.cli import _live_key
            config['api_key'] = _live_key(kind)
        if kind != 'anthropic' and urlsplit(endpoint).hostname in LOOPBACK and not key_env:
            config['api_key_required'] = False
        require(config.get('api_key') or config.get('api_key_required') is False,
                'set the provider credential environment variable before connecting')
    return providers.build_adapter(kind, config, remote=True)


def model_input(request):
    """Validate and copy a canonical text chat input without rewriting its content."""
    require(isinstance(request, dict) and set(request) <= {'messages', 'tools'},
            'request must contain messages and optional tools')
    messages = request.get('messages')
    require(isinstance(messages, list) and 0 < len(messages) <= 256, 'request needs 1–256 messages')
    for message in messages:
        require(isinstance(message, dict) and message.get('role') in ('system', 'user', 'assistant', 'tool'),
                'messages need a supported text-chat role')
        role = message['role']
        require(isinstance(message.get('content'), str) or (role == 'assistant' and message.get('content') is None),
                'message content must be text')
        calls = message.get('tool_calls', [])
        require(isinstance(calls, list) and all(isinstance(call, dict) and isinstance(call.get('name'), str)
                and isinstance(call.get('arguments', {}), dict) for call in calls),
                'assistant tool calls must use canonical names and argument objects')
        require(not calls or role == 'assistant', 'only assistant messages contain tool calls')
        if role == 'tool':
            record_id(message.get('tool_call_id'), 'tool_call_id')
    tools = request.get('tools', [])
    require(isinstance(tools, list) and len(tools) <= 128 and all(
        isinstance(tool, dict) and isinstance(tool.get('name'), str) and bool(tool['name'])
        and isinstance(tool.get('parameters', {}), dict) for tool in tools), 'invalid canonical tool definitions')
    return deepcopy(request)


def chat_workflow(project, checks, reviewer, allowed_tools=()):
    """Reviewed next-turn contract used for a project created from captures."""
    require(isinstance(allowed_tools, (list, tuple)) and all(isinstance(tool, str) and tool for tool in allowed_tools),
            'allowed_tools must be a list of tool names')
    spec = {'workflow_version': 1, 'workflow_id': project_name(project), 'name': project,
            'status': 'reviewed', 'reviewed_by': record_id(reviewer, 'reviewer'),
            'inputs': {'request': {'type': 'object'}}, 'outputs': [],
            'steps': [{'id': 'reply', 'kind': 'model', 'depends_on': [], 'request': {'$ref': 'inputs.request'}}],
            'allowed_tools': list(dict.fromkeys(allowed_tools)), 'success_checks': deepcopy(checks),
            'tool_proposals_as_output': True, 'evidence': [], 'uncertainty': []}
    spec['revision'] = workflow_revision(spec)
    return validate_workflow(spec, require_reviewed=True)


def replay_suite(spec, suite):
    validate_suite(spec, suite)
    require(2 <= len(suite['cases']) <= MAX_CASES, f'provide 2–{MAX_CASES} reviewed replay cases')
    for case in suite['cases']:
        require(case.get('workflow_id') == spec['workflow_id'], 'case belongs to another workflow')
        record_id(case['trace_id'], 'case trace_id')
        record_id(case.get('split_group', case['trace_id']), 'case split_group')
        if spec.get('tool_simulation'):
            require(isinstance(case.get('initial_state'), dict), 'stateful cases need an initial_state object')
        if spec.get('tool_proposals_as_output') and spec['steps'][0].get('request') == {'$ref': 'inputs.request'}:
            model_input(case.get('inputs', {}).get('request'))
    return fingerprint(suite)
