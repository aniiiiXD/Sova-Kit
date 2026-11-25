"""Configurable development DAG templates and a credential-free smoke graph."""
from copy import deepcopy
from pathlib import Path
import sys

from switchit.contracts import require
from .model import validate_plan


def generate_development_plan(workers, *, sources, include_publish=False, max_parallel=3, repairs=None):
    """Create investigator → coder → verifier → package tasks from real wrappers.

    workers maps role names to task command/cwd/settings dictionaries. Required
    roles are listed in docs/orchestration.md. No vendor command is invented, and
    publication only exists when include_publish=True and a publisher is supplied.
    """
    require(isinstance(workers, dict) and isinstance(sources, dict), 'workers and sources must be mappings')
    require(set(sources) == {'downloads', 'claude', 'codex'}, 'sources must declare downloads, claude, and codex paths')
    repairs = repairs or {}
    require(isinstance(repairs, dict) and set(repairs) <= {'source', 'workflow', 'target'}, 'repairs keys must be source, workflow, or target')
    tasks = []

    def node(task_id, role, dependencies, instruction, **details):
        require(role in workers, f'worker wrapper not configured: {role}')
        config = deepcopy(workers[role])
        require(isinstance(config, dict) and 'command' in config and 'cwd' in config, f'{role} requires command and cwd')
        require(not {'id', 'depends_on', 'input'} & set(config), f'{role} worker configuration cannot override graph routing')
        config.setdefault('outputs', ['result.json'])
        tasks.append({**config, 'id': task_id, 'depends_on': dependencies,
                      'input': {'role': role, 'instruction': instruction, **details}})

    for name in ('downloads', 'claude', 'codex'):
        node('investigate-'+name, 'investigate_'+name, [],
             'Investigate the configured source only; produce a source/schema inventory and evidence gaps. Do not modify the original source.', source=str(sources[name]))
    investigations = ['investigate-downloads', 'investigate-claude', 'investigate-codex']
    node('code-source-adapters', 'code_source', investigations,
         'Implement source adapters against the recorded schema contracts. Preserve identities, provenance, and uncertainty.')
    node('code-neutral-workflows', 'code_workflow', investigations,
         'Implement the vendor-neutral event, hierarchy, dependency, and workflow contracts. Preserve reviewed success criteria.')
    node('code-target-adapters', 'code_target', investigations,
         'Implement configured target runtime adapters with explicit capabilities; unsupported behavior must remain explicit.')
    # Optional explicit repair lanes execute the actual configured coder and fixed
    # verifier before the separate downstream test gate. No cyclic DAG mutation.
    from .repair import _configuration
    for lane, coder_id, code_role, test_role in [
            ('source', 'code-source-adapters', 'code_source', 'test_source'),
            ('workflow', 'code-neutral-workflows', 'code_workflow', 'test_workflow'),
            ('target', 'code-target-adapters', 'code_target', 'test_target')]:
        if lane not in repairs:
            continue
        settings = repairs[lane]
        require(isinstance(settings, dict) and set(settings) <= {'criteria_paths', 'max_rounds', 'max_run_seconds'},
                'repair lane accepts criteria_paths, max_rounds, and max_run_seconds')
        require(test_role in workers, f'worker wrapper not configured: {test_role}')
        current = next(task for task in tasks if task['id'] == coder_id)
        coder = {key: deepcopy(value) for key, value in current.items() if key not in ('id', 'depends_on', 'max_attempts')}
        verifier = deepcopy(workers[test_role])
        require(not set(verifier) & {'id', 'depends_on', 'input'}, f'{test_role} cannot override graph routing')
        verifier.pop('max_attempts', None)
        verifier.setdefault('outputs', ['result.json'])
        repair_config = {'repair_version': 1, 'coder': coder, 'verifier': verifier, **deepcopy(settings)}
        validated = _configuration(repair_config)
        current.update(command=[sys.executable, '-m', 'switchit.orchestration.repair', '--task-input'],
                       timeout_seconds=validated['max_run_seconds']+5, max_attempts=1,
                       outputs=['repair-result.json'], input={'repair_config': repair_config},
                       pass_env=sorted(set(coder.get('pass_env', []) + verifier.get('pass_env', []))),
                       env={'PYTHONPATH': str(Path(__file__).resolve().parents[2])})
    for suffix, coder, role in [('source', 'code-source-adapters', 'test_source'),
                                ('workflow', 'code-neutral-workflows', 'test_workflow'),
                                ('target', 'code-target-adapters', 'test_target')]:
        node('test-'+suffix, role, [coder],
             'Independently test the supplied implementation and fixed contracts. Return failure evidence; do not weaken checks to obtain a pass.')
    node('verify-integration', 'verify_integration', ['test-source', 'test-workflow', 'test-target'],
         'Integrate tested revisions in the designated integration checkout and verify vendor-1 to neutral workflow to vendor-2 behavior against fixed outcomes.')
    for product in ('skill', 'harness', 'platform'):
        node('package-'+product, 'package_'+product, ['verify-integration'],
             f'Build the {product} artifact from the verified revision and report artifact digests. Do not publish.')
    node('verify-release', 'verify_release', ['package-skill', 'package-harness', 'package-platform'],
         'Run installation and release smoke checks against the exact produced artifacts; return a release evidence manifest.')
    if include_publish:
        node('publish', 'publish', ['verify-release'],
             'Publish only the exact artifacts covered by the release evidence and configured release destinations.')
    return validate_plan({'plan_version': 1, 'name': 'SwitchIt autonomous development',
                          'max_parallel': max_parallel, 'tasks': tasks})


def smoke_plan(workspace):
    """Create three isolated local work dirs and a small executable fanout/join plan."""
    workspace = Path(workspace).expanduser().resolve()
    tasks = []
    for name, dependencies in [('left', []), ('right', []), ('join', ['left', 'right'])]:
        cwd = workspace/name
        cwd.mkdir(parents=True, exist_ok=True)
        script = ('import json,sys; from pathlib import Path; request=json.load(sys.stdin); '
                  'Path("result.json").write_text(json.dumps({"task":request["task_id"],'
                  '"dependencies":sorted(request["dependencies"])}))')
        tasks.append({'id': name, 'depends_on': dependencies, 'cwd': str(cwd),
                      'command': [sys.executable, '-c', script], 'outputs': ['result.json']})
    return validate_plan({'plan_version': 1, 'name': 'Offline orchestration smoke', 'max_parallel': 2, 'tasks': tasks})
