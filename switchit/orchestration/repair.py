"""Bounded coder → fixed independent verifier → feedback repair rounds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import signal
import sys
from pathlib import Path
import time

from switchit.contracts import fingerprint, require
from .model import _integer, _seconds, validate_plan
from .scheduler import run_plan
from .state import coordinator_lock, digest_file, utc_now, write_json


def _configuration(config):
    require(isinstance(config, dict) and config.get('repair_version') == 1, 'unsupported repair version')
    require(set(config) <= {'repair_version', 'coder', 'verifier', 'criteria_paths', 'max_rounds', 'max_run_seconds'}, 'unknown repair config field')
    result = {'repair_version': 1, 'max_rounds': _integer(config.get('max_rounds', 3), 'max_rounds', 1, 10),
              'max_run_seconds': _seconds(config.get('max_run_seconds', 900), 'max_run_seconds')}
    for role in ('coder', 'verifier'):
        raw = config.get(role)
        require(isinstance(raw, dict), f'{role} must be a configured command object')
        require(not set(raw) & {'id', 'depends_on', 'max_attempts'}, f'{role} cannot override id, dependencies, or attempt count')
        result[role] = validate_plan({'plan_version': 1, 'tasks': [{'id': role, **raw}]})['tasks'][0]
    criteria = config.get('criteria_paths')
    require(isinstance(criteria, list) and criteria and all(isinstance(p, str) and Path(p).is_absolute() for p in criteria),
            'criteria_paths must list absolute verifier criteria files')
    result['criteria_paths'] = sorted({str(Path(p).resolve()) for p in criteria})
    require(all(Path(p).is_file() for p in result['criteria_paths']), 'each criteria path must be an existing file')
    return result


def _frozen_files(config):
    paths = set(config['criteria_paths'])
    # Explicit script/interpreter paths are pinned as well as the argv itself.
    # Imported modules and PATH-resolved executables require explicit criteria_paths.
    cwd = Path(config['verifier']['cwd'])
    for argument in config['verifier']['command']:
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = cwd/candidate
        try:
            if candidate.is_file():
                paths.add(str(candidate.resolve()))
        except OSError:
            continue
    return {path: digest_file(path) for path in sorted(paths)}


def _check_frozen(frozen):
    for path, digest in frozen.items():
        require(Path(path).is_file() and digest_file(path) == digest,
                f'fixed verifier or acceptance criteria changed: {path}')


def _check_receipts(state, root):
    for receipt in state.get('receipts', []):
        path = (root/receipt['path']).resolve()
        require(path.is_relative_to(root) and path.is_file() and digest_file(path) == receipt['sha256'],
                'accepted repair artifact changed or disappeared; use a new run root')


def run_repair_loop(config, root, *, poll_interval=0.05):
    """Run/resume bounded rounds using actual configured executables.

    Verifier failures are passed to the next coder in input.previous_verification.
    Criteria and verifier script digests are immutable for this run. Both phases
    run through the command DAG scheduler; cancellation uses the common cancel API.
    """
    config = _configuration(config)
    root = Path(root).expanduser().resolve()
    with coordinator_lock(root):
        config_path, state_path = root/'repair-plan.json', root/'state.json'
        if config_path.exists():
            require(json.loads(config_path.read_text()) == config, 'repair plan is immutable; use a new run directory')
        else:
            write_json(config_path, config)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            require(state.get('repair_state_version') == 1 and state.get('config_digest') == fingerprint(config), 'persisted repair config/state mismatch')
            _check_receipts(state, root)
        else:
            state = {'repair_state_version': 1, 'config_digest': fingerprint(config),
                     'verifier_digest': fingerprint(config['verifier']), 'criteria_digests': _frozen_files(config),
                     'status': 'pending', 'created_at': utc_now(), 'elapsed_seconds': 0,
                     'rounds': [], 'receipts': [], 'tasks': {}}
        prior_elapsed, started = state['elapsed_seconds'], time.monotonic()
        deadline = started + max(0, config['max_run_seconds']-prior_elapsed)

        def save():
            state['elapsed_seconds'] = prior_elapsed + time.monotonic()-started
            state['updated_at'] = utc_now()
            write_json(state_path, state)

        try:
            _check_frozen(state['criteria_digests'])
        except ValueError as error:
            state.update(status='failed', error=str(error))
            save()
            return deepcopy(state)
        if state['status'] in ('succeeded', 'failed'):
            return deepcopy(state)
        (root/'cancel.json').unlink(missing_ok=True)
        state['status'] = 'running'
        save()

        def stopped():
            if (root/'cancel.json').exists():
                state.update(status='cancelled', error='repair cancellation requested')
                return True
            if time.monotonic() >= deadline:
                state.update(status='failed', error='repair time budget exhausted')
                return True
            return False

        def phase(round_record, role):
            previous = state['rounds'][-2].get('feedback') if len(state['rounds']) > 1 else None
            spec = deepcopy(config[role])
            spec['input'] = {'round': round_record['number'], 'request': spec['input'],
                             'previous_verification': previous,
                             'criteria_digests': state['criteria_digests'],
                             'coder_receipt': round_record.get('coder') if role == 'verifier' else None}
            plan = {'plan_version': 1, 'name': f"Repair round {round_record['number']} {role}",
                    'max_parallel': 1, 'max_run_seconds': config['max_run_seconds'], 'tasks': [spec]}
            directory = root/'rounds'/f"round-{round_record['number']:04d}"/role
            round_record['phase'] = role
            save()
            result = run_plan(plan, directory, poll_interval=poll_interval,
                              cancel_path=root/'cancel.json', deadline=deadline)
            receipt = {'status': result['status'], 'state_path': str(directory/'state.json'),
                       'task': result['tasks'][role]}
            round_record[role] = receipt
            for artifact in result['tasks'][role].get('artifacts', []):
                item = {**artifact, 'path': str((directory/artifact['path']).relative_to(root))}
                if item not in state['receipts']:
                    state['receipts'].append(item)
            save()
            return result['status']

        try:
            while True:
                if stopped():
                    break
                if not state['rounds'] or state['rounds'][-1].get('complete'):
                    if len(state['rounds']) >= config['max_rounds']:
                        state.update(status='failed', error='repair round budget exhausted')
                        break
                    state['rounds'].append({'number': len(state['rounds'])+1, 'complete': False})
                    save()
                current = state['rounds'][-1]
                _check_frozen(state['criteria_digests'])
                coder_status = phase(current, 'coder')
                _check_frozen(state['criteria_digests'])
                if coder_status in ('cancelled', 'interrupted'):
                    state['status'] = coder_status
                    break
                if coder_status != 'succeeded':
                    current.update(complete=True, feedback={'phase': 'coder', **current['coder']})
                    save()
                    continue
                if stopped():
                    break
                verifier_status = phase(current, 'verifier')
                _check_frozen(state['criteria_digests'])
                if verifier_status in ('cancelled', 'interrupted'):
                    state['status'] = verifier_status
                    break
                current.update(complete=True, feedback={'phase': 'verifier', **current['verifier']})
                save()
                if verifier_status == 'succeeded':
                    state.update(status='succeeded', error=None)
                    break
        except ValueError as error:
            state.update(status='failed', error=str(error))
        except BaseException:
            state['status'] = 'interrupted'
            save()
            raise
        save()
        return deepcopy(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--root')
    parser.add_argument('--task-input', action='store_true')
    args = parser.parse_args(argv)
    if args.task_input:
        require(not args.config and not args.root, '--task-input cannot be combined with --config/--root')
        request = json.load(sys.stdin)
        config = deepcopy(request['input']['repair_config'])
        config['coder']['input'] = {'objective': config['coder'].get('input'),
                                     'dependencies': request['dependencies'], 'run_root': request['run_root']}
        root = Path(request['run_root'])/'repairs'/request['task_id']
    else:
        require(args.config and args.root, '--config and --root are required')
        config, root = json.loads(Path(args.config).read_text()), args.root
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_repair_loop(config, root)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if args.task_input:
        write_json(Path.cwd()/'repair-result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
