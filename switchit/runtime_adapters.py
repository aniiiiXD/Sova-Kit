"""Explicit runtime contracts. Agent log import support never implies execution support."""
from copy import deepcopy
import json
import os
import subprocess
import tempfile
import time

from switchit.contracts import require
from switchit.fixtures import FixtureSession


PROVIDER_RECIPE_SETTINGS = {'max_attempts', 'context_policy', 'include_step_outputs'}


def model_request(request, recipe, previous_steps=None):
    """Apply the same recipe to live requests and reviewed replay steps."""
    request = deepcopy(request)
    require(request.get('messages'), 'model step needs an explicit request')
    config = recipe.get('config', {})
    require(set(config) <= PROVIDER_RECIPE_SETTINGS, 'unsupported provider recipe setting')
    policy = config.get('context_policy', 'full')
    require(policy in ('full', 'last_user'), 'unsupported context policy')
    request['messages'] = list(request['messages'])
    if policy == 'last_user':
        systems = [m for m in request['messages'] if m.get('role') == 'system']
        users = [m for m in request['messages'] if m.get('role') == 'user']
        request['messages'] = systems + users[-1:]
    if recipe.get('system_prompt'):
        request['messages'].insert(0, {'role': 'system', 'content': recipe['system_prompt']})
    if config.get('include_step_outputs', True) and previous_steps:
        request['messages'].append({'role': 'user', 'content': 'Previous workflow step outputs: ' + json.dumps(previous_steps)})
    if 'allowed_tools' in recipe:
        allowed = set(recipe['allowed_tools'])
        request['tools'] = [tool for tool in request.get('tools', []) if tool.get('name') in allowed]
    return request


class RuntimeAdapter:
    runtime_id = 'abstract'
    version = '1'
    remote = False
    executes_tools = False
    step_kinds = frozenset()
    models = frozenset()

    def capabilities(self):
        return {'runtime_id': self.runtime_id, 'version': self.version, 'remote': self.remote,
                'executes_tools': self.executes_tools, 'step_kinds': sorted(self.step_kinds),
                'models': sorted(self.models)}

    def execute(self, step, inputs, context):
        raise NotImplementedError

    def snapshot_state(self, context):
        """Return observed JSON state or None when this runtime cannot observe it."""
        return None


class FixtureRuntime(RuntimeAdapter):
    """Offline baseline; reads explicitly supplied model outputs and exact tool fixtures."""
    runtime_id = 'fixtures'
    step_kinds = frozenset({'model', 'tool', 'artifact', 'approval', 'handoff'})

    def __init__(self, case):
        self.case = deepcopy(case)
        self.fixtures = FixtureSession(case.get('fixtures', {'fixtures_version': 1, 'fixtures': []}), case.get('trace_id'))

    def snapshot_state(self, context):
        return deepcopy(self.case.get('state_after'))

    def execute(self, step, inputs, context):
        if step['kind'] == 'tool':
            fixture = self.fixtures.resolve(step['tool'], step.get('arguments', {}))
            require(fixture['status'] == 'matched', f'tool fixture {fixture["status"]}: {step["tool"]}')
            return {'value': fixture['result'], 'fixture_id': fixture['fixture_id'], 'usage': None, 'cost': 0}
        outputs = self.case.get('step_outputs', {})
        model_fixture = context.get('recipe', {}).get('config', {}).get('model_fixture')
        if step['kind'] == 'model' and model_fixture is not None:
            require(isinstance(model_fixture, str) and model_fixture, 'model_fixture must be a nonempty name')
            outputs = self.case.get('model_fixtures', {}).get(model_fixture)
            require(isinstance(outputs, dict), f'missing model fixture: {model_fixture}')
        output = outputs.get(step['id'])
        require(output is not None, f'no recorded output for step {step["id"]}')
        return deepcopy(output)


class ProviderRuntime(RuntimeAdapter):
    """Use an existing model adapter for explicit model steps; tools remain fixture-backed."""
    runtime_id = 'provider'
    step_kinds = frozenset({'model', 'tool'})

    def __init__(self, provider, case):
        self.provider = provider
        self.fixture_runtime = FixtureRuntime(case)
        self.case = deepcopy(case)
        self.remote = bool(getattr(provider, 'remote', False))
        self.models = frozenset({provider.model}) if getattr(provider, 'model', None) else frozenset()

    def execute(self, step, inputs, context):
        if step['kind'] == 'tool':
            return self.fixture_runtime.execute(step, inputs, context)
        recipe = context.get('recipe', {})
        config = recipe.get('config', {})
        attempts = config.get('max_attempts', 1)
        require(isinstance(attempts, int) and 1 <= attempts <= 3, 'max_attempts must be between 1 and 3')
        request = model_request(step.get('request') or self.case.get('request') or {}, recipe, context.get('steps'))
        results = []
        started = time.monotonic()
        for _ in range(attempts):
            if results and time.monotonic() - started >= context.get('remaining_seconds', float('inf')):
                break
            if results and context.get('remaining_tokens') is not None:
                tokens = [(r.get('usage') or {}).get(key) for r in results for key in ('input_tokens', 'output_tokens')]
                if any(value is None for value in tokens) or sum(tokens) >= context['remaining_tokens']:
                    break
            result = self.provider.replay(request, self.case)
            results.append(result)
            if not result.get('error'):
                break
        final = deepcopy(results[-1])
        bases = {(r.get('usage') or {}).get('basis', 'unknown') for r in results}
        final['usage'] = {'basis': next(iter(bases)) if len(bases) == 1 else 'unknown', 'estimator': None}
        for key in ('input_tokens', 'output_tokens'):
            values = [(r.get('usage') or {}).get(key) for r in results]
            final['usage'][key] = sum(values) if all(v is not None for v in values) else None
        for key in ('latency_ms', 'cost'):
            values = [r.get(key) for r in results]
            final[key] = sum(values) if all(v is not None for v in values) else None
        final['attempts'] = len(results)
        return final


class CommandRuntime(RuntimeAdapter):
    """Opt-in JSON subprocess bridge for a caller-supplied agent wrapper.

    The subprocess receives {step, inputs, context} on stdin and returns one JSON
    object on stdout. A temporary cwd is isolation for artifacts, NOT an OS sandbox.
    No vendor CLI flags or undocumented APIs are guessed here.
    """
    step_kinds = frozenset({'model', 'tool', 'artifact', 'approval', 'handoff'})
    executes_tools = True

    def __init__(self, runtime_id, argv, *, version, allow_execution=False, remote=False,
                 allow_remote=False, timeout=60, env=None, max_output_bytes=1_000_000, models=()):
        require(allow_execution, 'command execution requires explicit opt-in')
        require(not remote or allow_remote, 'remote runtime requires explicit opt-in')
        require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), 'argv must be a nonempty string list')
        require(timeout > 0 and max_output_bytes > 0, 'timeout/output limit must be positive')
        self.runtime_id, self.argv, self.version = runtime_id, list(argv), version
        self.remote, self.timeout, self.env, self.max_output_bytes = remote, timeout, dict(env or {}), max_output_bytes
        self.models = frozenset(models)

    def execute(self, step, inputs, context):
        payload = json.dumps({'step': step, 'inputs': inputs, 'context': context}).encode()
        environment = {'PATH': os.defpath, **self.env}
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            completed = subprocess.run(self.argv, input=payload, stdout=out, stderr=err,
                                       cwd=context['workspace'], env=environment,
                                       timeout=min(self.timeout, context.get('remaining_seconds', self.timeout)), check=False)
            require(completed.returncode == 0, f'agent wrapper exited with status {completed.returncode}')
            size = out.tell()
            require(size <= self.max_output_bytes, 'agent wrapper exceeded output limit')
            out.seek(0)
            result = json.loads(out.read())
            require(isinstance(result, dict), 'agent wrapper output must be a JSON object')
            return result


class RuntimeRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, runtime_id, factory):
        require(runtime_id not in self._factories, f'runtime already registered: {runtime_id}')
        require(callable(factory), 'runtime factory must be callable')
        self._factories[runtime_id] = factory

    def create(self, recipe, case):
        name = recipe['runtime']
        require(name in self._factories, f'unsupported execution runtime: {name}; importing its traces does not enable execution')
        runtime = self._factories[name](deepcopy(recipe), deepcopy(case))
        require(isinstance(runtime, RuntimeAdapter), 'factory must return a RuntimeAdapter')
        require(runtime.runtime_id == name, 'runtime identity does not match registration')
        require(runtime.version == recipe['runtime_version'], 'runtime version mismatch')
        return runtime
