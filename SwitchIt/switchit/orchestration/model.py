"""Validated, immutable command DAGs for autonomous development workers."""
from copy import deepcopy
import math
from pathlib import Path
import re

from switchit.contracts import require, topological_order

_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def _integer(value, label, minimum, maximum):
    require(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum,
            f'{label} must be an integer in [{minimum}, {maximum}]')
    return value


def _seconds(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, f'{label} must be finite and positive')
    return value


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get('plan_version') == 1, 'unsupported orchestration plan version')
    require(set(plan) <= {'plan_version', 'name', 'tasks', 'max_parallel', 'max_run_seconds'}, 'unknown orchestration plan field')
    require(isinstance(plan.get('tasks'), list) and plan['tasks'], 'plan must contain tasks')
    result = {'plan_version': 1, 'name': plan.get('name', 'SwitchIt development graph'),
              'max_parallel': _integer(plan.get('max_parallel', 3), 'max_parallel', 1, 32),
              'max_run_seconds': _seconds(plan.get('max_run_seconds', 3600), 'max_run_seconds'), 'tasks': []}
    require(isinstance(result['name'], str), 'plan name must be a string')
    for raw in plan['tasks']:
        require(isinstance(raw, dict), 'task must be an object')
        require(set(raw) <= {'id', 'command', 'cwd', 'depends_on', 'max_attempts', 'timeout_seconds',
                            'env', 'pass_env', 'outputs', 'input', 'output_limit_bytes'}, 'unknown task field')
        require(isinstance(raw.get('id'), str) and _ID.fullmatch(raw['id']), 'task id must be a safe short identifier')
        command = raw.get('command')
        require(isinstance(command, list) and command and all(isinstance(arg, str) and '\0' not in arg for arg in command), 'command must be a nonempty argv string array')
        cwd = raw.get('cwd')
        require(isinstance(cwd, str) and Path(cwd).is_absolute(), 'each task needs an absolute cwd')
        require(Path(cwd).is_dir(), f'task cwd does not exist: {cwd}')
        dependencies = raw.get('depends_on', [])
        require(isinstance(dependencies, list) and all(isinstance(dep, str) for dep in dependencies), 'depends_on must be a string array')
        require(len(set(dependencies)) == len(dependencies), 'duplicate dependency')
        env = raw.get('env', {})
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) and k and '=' not in k and '\0' not in k + v for k, v in env.items()), 'env must map valid string names to string values')
        pass_env = raw.get('pass_env', [])
        require(isinstance(pass_env, list) and all(isinstance(key, str) and key and '=' not in key and '\0' not in key for key in pass_env), 'pass_env must list environment variable names')
        outputs = raw.get('outputs', [])
        require(isinstance(outputs, list) and all(isinstance(path, str) and path and not Path(path).is_absolute()
                and '..' not in Path(path).parts for path in outputs), 'outputs must be relative paths within cwd')
        require(len(set(outputs)) == len(outputs), 'duplicate output path')
        result['tasks'].append({'id': raw['id'], 'command': list(command), 'cwd': str(Path(cwd).resolve()),
            'depends_on': list(dependencies), 'max_attempts': _integer(raw.get('max_attempts', 1), 'max_attempts', 1, 10),
            'timeout_seconds': _seconds(raw.get('timeout_seconds', 300), 'timeout_seconds'),
            'output_limit_bytes': _integer(raw.get('output_limit_bytes', 4_000_000), 'output_limit_bytes', 1, 100_000_000),
            'env': deepcopy(env), 'pass_env': list(pass_env), 'outputs': list(outputs), 'input': deepcopy(raw.get('input'))})
    topological_order([task['id'] for task in result['tasks']],
                      [(dep, task['id']) for task in result['tasks'] for dep in task['depends_on']])
    return result
