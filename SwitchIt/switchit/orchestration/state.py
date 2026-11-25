"""Single-coordinator locking, atomic state snapshots, and artifact fingerprints."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


class CoordinatorBusy(RuntimeError):
    """Another coordinator owns this run directory."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    """Atomic and fsynced per-file writes; callers serialize related mutations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def coordinator_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root/'coordinator.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            stream.write(b'0')
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise CoordinatorBusy(f'another coordinator owns {root}') from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CoordinatorBusy(f'another coordinator owns {root}') from error
        yield
    finally:
        stream.close()  # Kernel releases the lock even after coordinator death.


def status(root):
    path = Path(root)/'state.json'
    if not path.exists():
        return {'orchestration_state_version': 1, 'status': 'not_started', 'tasks': {}}
    state = json.loads(path.read_text())
    state['cancel_requested'] = (Path(root)/'cancel.json').exists()
    return state


def cancel(root):
    root = Path(root)
    if not (root/'state.json').exists():
        raise ValueError('cannot cancel a run that has not started')
    write_json(root/'cancel.json', {'requested_at': utc_now()})
    return status(root)
