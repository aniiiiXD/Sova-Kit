"""Bounded step execution in a fresh workspace, separate from workflow evaluation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time

from switchit.contracts import fingerprint, nonnegative, require, topological_order
from switchit.redaction import redact_value
from switchit.runtime_adapters import FixtureRuntime
from switchit.tool_simulation import apply_tool_operation
from switchit.workflow_builder import validate_workflow
from switchit.workflow_evaluation import case_checks, lookup, lookup_observation


def _path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'artifact path must be relative')
    target = (root / relative).resolve()
    require(target.is_relative_to(root.resolve()) and target != root.resolve(), 'artifact path escapes workspace')
    return target


def _resolve(value, scope):
    if isinstance(value, dict):
        if set(value) == {'$ref'}:
            exists, resolved = lookup(scope, value['$ref'])
            require(exists, f'missing input binding: {value["$ref"]}')
            return deepcopy(resolved)
        return {key: _resolve(item, scope) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, scope) for item in value]
    return value


def run_workflow(spec, case, runtime, recipe=None, *, max_steps=100, max_seconds=60,
                 max_tokens=None, allow_remote=False, allow_tool_execution=False):
    validate_workflow(spec, require_reviewed=True)
    require(case.get('workflow_revision') == spec['revision'], 'case belongs to another workflow revision')
    require(max_steps > 0 and max_seconds > 0, 'execution budgets must be positive')
    require(not runtime.remote or allow_remote, 'remote execution not enabled')
    require(not runtime.executes_tools or allow_tool_execution, 'business-tool execution not enabled')
    simulation = spec.get('tool_simulation', {})
    require(not simulation or not runtime.executes_tools, 'stateful replay requires a runtime without external tool execution')
    if max_tokens is not None:
        nonnegative(max_tokens, 'max_tokens')
    capabilities = runtime.capabilities()
    required_kinds = {s['kind'] for s in spec['steps']}
    if simulation:
        required_kinds.discard('tool')
    require(required_kinds <= set(capabilities['step_kinds']), 'runtime lacks required step kinds')
    recipe = deepcopy(recipe or {})
    allowed = set(recipe.get('allowed_tools', spec['allowed_tools']))
    require(allowed <= set(spec['allowed_tools']), 'recipe expands workflow tool permissions')
    if recipe.get('model'):
        require(recipe['model'] in capabilities['models'], 'runtime cannot execute the recipe model')
    order = topological_order([s['id'] for s in spec['steps']], [(dep, s['id']) for s in spec['steps'] for dep in s['depends_on']])
    steps = {s['id']: s for s in spec['steps']}
    outputs, records, calls, proposals_observed, pending_proposals = {}, [], [], [], []
    token_totals = {'input_tokens': 0, 'output_tokens': 0}
    unknown_usage = False
    usage_bases = set()
    costs, final_text, status, error = [], '', 'completed', None
    started = time.monotonic()
    artifacts = {}
    state = None
    if simulation:
        require(isinstance(case.get('initial_state'), dict), 'stateful replay requires a case initial_state object')
        state = deepcopy(case['initial_state'])
        initial_state_digest = fingerprint(state)
        executed_tools = []
        if any(s['kind'] == 'tool' and s['tool'] not in simulation for s in spec['steps']):
            recorded_tools = FixtureRuntime(case)
    with tempfile.TemporaryDirectory(prefix='switchit-run-') as directory:
        root = Path(directory)
        for output_path in spec["outputs"]:
            _path(root, output_path)
        for name, content in case.get('workspace_files', {}).items():
            target = _path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for sid in order:
            step = deepcopy(steps[sid])
            scope = {'inputs': case.get('inputs', {}), 'steps': outputs}
            if step.get('when'):
                when = step['when']
                exists, value = lookup(scope, when['path'])
                if not exists and when.get('on_missing', 'uncertain') != 'skip':
                    status, error = 'uncertain', f'branch input not observed for {sid}'
                    break
                if not exists or value != when['equals']:
                    records.append({'step_id': sid, 'kind': step['kind'], 'status': 'skipped'})
                    continue
            if any(dep not in outputs for dep in step['depends_on']):
                status, error = 'uncertain', f'dependency unavailable for {sid}'
                break
            elapsed = time.monotonic() - started
            if len(records) >= max_steps or elapsed >= max_seconds:
                status, error = 'budget_exceeded', 'step/time budget exhausted'
                break
            if max_tokens is not None and (unknown_usage or sum(token_totals.values()) >= max_tokens):
                status, error = 'budget_exceeded', 'token budget exhausted or usage unknown'
                break
            step_started = time.monotonic()
            try:
                step = _resolve(step, scope)
                if step['kind'] == 'tool':
                    require(step['tool'] in allowed, f'tool outside approved scope: {step["tool"]}')
                    require(not step.get('requires_proposal') or pending_proposals,
                            f'tool step requires a model proposal: {step["tool"]}')
                    call = {'name': step['tool'], 'arguments': step.get('arguments', {})}
                    call_digest = fingerprint(call) if simulation else None
                    if pending_proposals:
                        proposed = pending_proposals[0]
                        proposed_call = {'name': proposed.get('name'), 'arguments': proposed.get('arguments', {})}
                        matches = fingerprint(proposed_call) == call_digest if simulation else proposed_call == call
                        require(matches,
                                'model tool proposal does not match the approved workflow step')
                        pending_proposals.pop(0)
                if simulation and step['kind'] == 'tool':
                    rule = simulation.get(step['tool'])
                    if rule:
                        action = _resolve(rule, {'arguments': call['arguments']})
                        output = apply_tool_operation(state, **action)
                    else:
                        output = recorded_tools.execute(step, {}, {})
                    executed_tools.append({**redact_value(call), 'step_id': redact_value(sid),
                                           'call_digest': call_digest,
                                           'source': 'simulated' if rule else 'recorded_fixture'})
                else:
                    output = runtime.execute(step, deepcopy(case.get('inputs', {})),
                        {'workspace': str(root), 'steps': deepcopy(outputs), 'recipe': deepcopy(recipe),
                         'remaining_seconds': max_seconds - elapsed,
                         'remaining_tokens': max_tokens - sum(token_totals.values()) if max_tokens is not None else None})
                require(isinstance(output, dict), 'runtime output must be an object')
                require(not output.get('error'), str(output.get('error')))
                proposals = output.get('tool_calls', [])
                require(all(call.get('name') in allowed for call in proposals), 'runtime proposed an unapproved tool')
                proposals_observed.extend(deepcopy(proposals))
                if step["kind"] == "model":
                    pending_proposals.extend(deepcopy(proposals))
                if step['kind'] == 'tool':
                    calls.append(call)
                usage = output.get('usage') or {}
                if step['kind'] == 'model':
                    for key in token_totals:
                        value = usage.get(key)
                        if value is None:
                            unknown_usage = True
                        else:
                            nonnegative(value, key)
                            token_totals[key] += value
                    usage_bases.add(usage.get('basis', 'unknown'))
                cost = output.get('cost')
                if cost is not None:
                    nonnegative(cost, 'cost')
                costs.append(cost)
                outputs[sid] = deepcopy(output)
                final_text = output.get('text', final_text)
                records.append({'step_id': sid, 'kind': step['kind'], 'tool': step.get('tool'), 'status': 'completed',
                                'output': output, 'latency_ms': round((time.monotonic()-step_started)*1000)})
                if time.monotonic() - started > max_seconds or (max_tokens is not None and sum(token_totals.values()) > max_tokens):
                    status, error = 'budget_exceeded', 'runtime call exceeded budget; no further steps executed'
                    break
            except Exception as failure:
                status, error = 'error', str(failure)
                records.append({'step_id': sid, 'kind': step['kind'], 'status': 'error', 'error': str(failure)})
                break
        if pending_proposals and status == 'completed' and not spec.get('tool_proposals_as_output'):
            status, error = 'uncertain', 'model proposed tools that the workflow did not execute'
        if not simulation:
            try:
                state = runtime.snapshot_state({'workspace': str(root), 'steps': deepcopy(outputs)})
                require(state is None or isinstance(state, dict), 'state snapshot must be an object or None')
            except Exception as failure:
                status, error = 'error', f'state capture failed: {failure}'
        for path in spec['outputs']:
            target = _path(root, path)
            if target.is_file():
                require(target.stat().st_size <= 1_000_000, 'artifact exceeds capture limit')
                artifacts[path] = target.read_text()
    latency = round((time.monotonic()-started)*1000)
    basis = next(iter(usage_bases)) if len(usage_bases) == 1 and not unknown_usage else 'unknown'
    usage = {**token_totals, 'basis': basis, 'estimator': None}
    if unknown_usage:
        usage.update(input_tokens=None, output_tokens=None)
    result = {'text': final_text, 'tool_calls': calls or proposals_observed, 'error': error if status == 'error' else None,
              'latency_ms': latency, 'usage': usage}
    run = redact_value({'status': status, 'error': error, 'steps': records, 'result': result,
            'artifacts': artifacts, 'state': state, 'proposals': proposals_observed,
            'metrics': {'latency_ms': latency, 'cost': sum(costs) if costs and all(c is not None for c in costs) else None,
                        'tokens_known': token_totals, 'usage_incomplete': unknown_usage},
            'limitations': ['temporary workspace is not an OS security sandbox',
                            'token/time limits stop subsequent calls; an in-flight provider call can exceed a budget']})
    run.update(workflow_run_version=1, workflow_id=spec['workflow_id'], workflow_revision=spec['revision'],
               case_id=case['case_id'], runtime=capabilities, recipe_digest=fingerprint(recipe))
    if simulation:
        observations = {}
        for check in case_checks(spec['success_checks'], case):
            if check['type'] == 'state_equals':
                path = check.get('path', '')
                exists, value = lookup_observation(state, path)
                observations[path] = fingerprint(value) if exists else None
        run['stateful_replay'] = {'initial_state_digest': initial_state_digest, 'final_state_digest': fingerprint(state),
                                  'state_observations': observations, 'executed_tools': executed_tools}
        run['limitations'].append('Tool effects are limited to the reviewed simulation and the supplied initial state.')
    return run
