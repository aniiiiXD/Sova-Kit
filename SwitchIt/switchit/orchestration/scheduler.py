"""A durable, bounded command DAG coordinator with independent task artifacts."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from switchit.contracts import fingerprint, require
from .model import validate_plan
from .state import coordinator_lock, digest_file, utc_now, write_json

_TERMINAL_TASKS = {'succeeded', 'failed', 'blocked'}


def _alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop(process):
    """Stop the owned process tree on POSIX; never signal a PID read from disk."""
    if process.poll() is not None:
        return
    try:
        if os.name != 'nt':
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=2)
    except ProcessLookupError:
        process.wait()


def _artifact(path, root, kind):
    return {'path': str(path.relative_to(root)), 'sha256': digest_file(path),
            'bytes': path.stat().st_size, 'kind': kind}


def _verify_success(state, root):
    """A resume cannot trust a success whose accepted output was changed/deleted."""
    for task in state['tasks'].values():
        if task['status'] != 'succeeded':
            continue
        for artifact in task.get('artifacts', []):
            path = (root/artifact['path']).resolve()
            require(path.is_relative_to(root) and path.is_file(), 'accepted task artifact is missing')
            require(digest_file(path) == artifact['sha256'], 'accepted task artifact changed; use a new run root')


def _initialize(plan, root):
    digest = fingerprint(plan)
    plan_path, state_path = root/'plan.json', root/'state.json'
    if plan_path.exists():
        require(json.loads(plan_path.read_text()) == plan, 'run plan is immutable; use a new run directory')
    else:
        write_json(plan_path, plan)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        require(state.get('orchestration_state_version') == 1, 'unsupported orchestration state version')
        require(state.get('plan_digest') == digest, 'persisted plan/state mismatch')
        require(set(state['tasks']) == {task['id'] for task in plan['tasks']}, 'persisted task set changed')
        _verify_success(state, root)
        for task_id, task in state['tasks'].items():
            if task['status'] == 'running':
                attempt = task['attempts'][-1]
                result_path = root/'tasks'/task_id/f"attempt-{attempt['number']:04d}"/'result.json'
                if result_path.exists():
                    recovered = json.loads(result_path.read_text())
                    require(recovered.get('number') == attempt['number'], 'attempt result identity mismatch')
                    task['attempts'][-1] = recovered
                    if recovered['status'] == 'succeeded':
                        task.update(status='succeeded', artifacts=recovered['artifacts'])
                        continue
                    if recovered['status'] in ('failed', 'timed_out'):
                        spec = next(spec for spec in plan['tasks'] if spec['id'] == task_id)
                        task['status'] = 'pending' if _failures(task) < spec['max_attempts'] else 'failed'
                        continue
                    if recovered['status'] in ('cancelled', 'interrupted'):
                        task['status'] = 'pending'
                        continue
                require(attempt.get('pid') is not None,
                        'prior launch outcome is unknown; reconcile its side effects before starting a new run')
                require(not _alive(attempt.get('pid')),
                        'a worker from the previous coordinator may still be running; wait for or stop that worker before resume')
                attempt.update(status='interrupted', finished_at=utc_now(), error='previous coordinator stopped before recording completion')
                task['status'] = 'pending'
            elif task['status'] in ('cancelled', 'interrupted'):
                task['status'] = 'pending'
        _verify_success(state, root)
        return state
    return {'orchestration_state_version': 1, 'plan_digest': digest,
            'name': plan['name'], 'status': 'pending', 'created_at': utc_now(),
            'updated_at': utc_now(), 'generation': 0, 'elapsed_seconds': 0.0,
            'tasks': {task['id']: {'status': 'pending', 'attempts': [], 'artifacts': []}
                      for task in plan['tasks']}}


def run_plan(plan, root, *, max_parallel=None, poll_interval=0.05, cancel_path=None, deadline=None):
    """Execute/resume an immutable command DAG and return its persisted state.

    Successful tasks are not repeated. Failure retries are bounded by max_attempts.
    Cancelled/interrupted attempts preserve their evidence but do not consume the
    failure retry allowance. Elapsed run budget is cumulative across normal resumes.
    Commands are executed directly, never through a shell. Their explicit cwd is
    not an OS sandbox, and arbitrary wrappers need idempotent external effects.
    """
    plan = validate_plan(plan)
    require(isinstance(poll_interval, (int, float)) and 0 < poll_interval <= 1, 'poll_interval must be in (0, 1]')
    parallel = max_parallel if max_parallel is not None else plan['max_parallel']
    require(isinstance(parallel, int) and not isinstance(parallel, bool) and 1 <= parallel <= plan['max_parallel'],
            'max_parallel override must be positive and cannot exceed the plan ceiling')
    root = Path(root).expanduser().resolve()
    with coordinator_lock(root):
        state = _initialize(plan, root)
        if state['status'] == 'succeeded':
            return deepcopy(state)
        # Resume is an explicit new invocation; clear the prior cancellation request.
        # A request issued after this point is observed by the active coordinator.
        (root/'cancel.json').unlink(missing_ok=True)
        tasks = {task['id']: task for task in plan['tasks']}
        active = {}
        started = time.monotonic()
        prior_elapsed = state.get('elapsed_seconds', 0)
        state['status'] = 'running'
        state['coordinator_pid'] = os.getpid()

        def save():
            state['elapsed_seconds'] = prior_elapsed + time.monotonic() - started
            state['updated_at'] = utc_now()
            state['generation'] += 1
            write_json(root/'state.json', state)

        def launch(task_id):
            task, record = tasks[task_id], state['tasks'][task_id]
            number = len(record['attempts']) + 1
            directory = root/'tasks'/task_id/f'attempt-{number:04d}'
            directory.mkdir(parents=True, exist_ok=False)
            request = {'task_id': task_id, 'attempt': number, 'run_root': str(root),
                       'input': task['input'], 'previous_attempt': deepcopy(record['attempts'][-1]) if record['attempts'] else None, 'dependencies': {
                           dep: deepcopy(state['tasks'][dep]['artifacts']) for dep in task['depends_on']}}
            write_json(directory/'input.json', request)
            attempt = {'number': number, 'status': 'running', 'started_at': utc_now(),
                       'pid': None, 'exit_code': None, 'error': None, 'artifacts': []}
            record['status'] = 'running'
            record['attempts'].append(attempt)
            save()  # Persist launch intent before starting a process.
            environment = {'PATH': os.defpath}
            environment.update({key: os.environ[key] for key in task['pass_env'] if key in os.environ})
            environment.update(task['env'])
            environment.update(SWITCHIT_TASK_INPUT=str(directory/'input.json'), SWITCHIT_RUN_ROOT=str(root), SWITCHIT_TASK_ID=task_id)
            stdin = (directory/'input.json').open('rb')
            stdout = (directory/'stdout.log').open('wb')
            stderr = (directory/'stderr.log').open('wb')
            try:
                process = subprocess.Popen(task['command'], cwd=task['cwd'], env=environment,
                                           stdin=stdin, stdout=stdout, stderr=stderr,
                                           start_new_session=os.name != 'nt')
            except Exception as error:
                for stream in (stdin, stdout, stderr):
                    stream.close()
                attempt.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=utc_now())
                write_json(directory/'result.json', attempt)
                record['status'] = 'pending' if _failures(record) < task['max_attempts'] else 'failed'
                save()
                return
            attempt['pid'] = process.pid
            active[task_id] = {'process': process, 'streams': (stdin, stdout, stderr),
                               'directory': directory, 'started': time.monotonic()}
            save()

        def finish(task_id, disposition=None, message=None):
            running = active.pop(task_id)
            task, record = tasks[task_id], state['tasks'][task_id]
            attempt = record['attempts'][-1]
            process = running['process']
            if disposition:
                _stop(process)
            for stream in running['streams']:
                stream.close()
            attempt.update(exit_code=process.returncode, finished_at=utc_now(),
                           elapsed_seconds=time.monotonic()-running['started'])
            attempt['status'] = disposition or ('succeeded' if process.returncode == 0 else 'failed')
            attempt['error'] = message or (f'command exited with status {process.returncode}' if process.returncode else None)
            directory = running['directory']
            artifacts = [_artifact(directory/name, root, 'log') for name in ('stdout.log', 'stderr.log')]
            if any(a['bytes'] > task['output_limit_bytes'] for a in artifacts):
                attempt.update(status='failed', error='command output exceeded the configured byte limit')
            if attempt['status'] == 'succeeded':
                try:
                    cwd = Path(task['cwd']).resolve()
                    for relative in task['outputs']:
                        source = (cwd/relative).resolve()
                        require(source.is_relative_to(cwd) and source.is_file(), f'declared output missing or outside cwd: {relative}')
                        require(source.stat().st_size <= task['output_limit_bytes'], f'declared output too large: {relative}')
                        destination = directory/'artifacts'/relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(source.read_bytes())
                        artifacts.append(_artifact(destination, root, 'output'))
                except (OSError, ValueError) as error:
                    attempt.update(status='failed', error=str(error))
            attempt['artifacts'] = artifacts
            write_json(directory/'result.json', attempt)
            if attempt['status'] == 'succeeded':
                record.update(status='succeeded', artifacts=artifacts)
            elif attempt['status'] in ('cancelled', 'interrupted'):
                record['status'] = attempt['status']
            else:
                record['status'] = 'pending' if _failures(record) < task['max_attempts'] else 'failed'
            save()

        save()
        try:
            while True:
                if (root/'cancel.json').exists() or (cancel_path is not None and Path(cancel_path).exists()):
                    for task_id in list(active):
                        finish(task_id, 'cancelled', 'run cancellation requested')
                    for record in state['tasks'].values():
                        if record['status'] == 'pending':
                            record['status'] = 'cancelled'
                    state['status'] = 'cancelled'
                    break
                if prior_elapsed + time.monotonic() - started >= plan['max_run_seconds'] or (deadline is not None and time.monotonic() >= deadline):
                    for task_id in list(active):
                        finish(task_id, 'timed_out', 'run time budget exhausted')
                    for record in state['tasks'].values():
                        if record['status'] == 'pending':
                            record.update(status='failed', error='run time budget exhausted')
                    state['status'] = 'failed'
                    break
                for task_id, running in list(active.items()):
                    task = tasks[task_id]
                    if running['process'].poll() is not None:
                        finish(task_id)
                    elif time.monotonic() - running['started'] >= task['timeout_seconds']:
                        finish(task_id, 'timed_out', 'task timeout exceeded')
                    elif any((running['directory']/name).stat().st_size > task['output_limit_bytes'] for name in ('stdout.log', 'stderr.log')):
                        finish(task_id, 'failed', 'command output exceeded the configured byte limit')
                for task_id, task in tasks.items():
                    record = state['tasks'][task_id]
                    if record['status'] != 'pending':
                        continue
                    dependencies = [state['tasks'][dep]['status'] for dep in task['depends_on']]
                    if any(value in ('failed', 'blocked') for value in dependencies):
                        record.update(status='blocked', error='required dependency failed')
                        save()
                occupied = {tasks[task_id]['cwd'] for task_id in active}
                for task_id, task in tasks.items():
                    if len(active) >= parallel:
                        break
                    record = state['tasks'][task_id]
                    if record['status'] == 'pending' and task['cwd'] not in occupied and all(
                            state['tasks'][dep]['status'] == 'succeeded' for dep in task['depends_on']):
                        launch(task_id)
                        if task_id in active:
                            occupied.add(task['cwd'])
                if not active and all(record['status'] in _TERMINAL_TASKS for record in state['tasks'].values()):
                    state['status'] = 'succeeded' if all(r['status'] == 'succeeded' for r in state['tasks'].values()) else 'failed'
                    break
                time.sleep(poll_interval)
        except BaseException:
            for task_id in list(active):
                finish(task_id, 'interrupted', 'coordinator interrupted')
            state['status'] = 'interrupted'
            save()
            raise
        save()
        return deepcopy(state)


def _failures(record):
    return sum(attempt['status'] in ('failed', 'timed_out') for attempt in record['attempts'])
