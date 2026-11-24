"""Coverage-aware discovery with content revisions and duplicate-source detection."""
from pathlib import Path
import hashlib

from switchit.contracts import fingerprint, require
from switchit.source_discovery import AGENT_DEFAULTS, discover_agent_sessions


def source_digest(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*')
            if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.jsonl', '.txt', '.zst', '.db'))
    require(files, f'no readable source files: {path}')
    digest = hashlib.sha256()
    for file in files:
        digest.update(("" if path.is_file() else str(file.relative_to(path))).encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(65536), b''):
                digest.update(chunk)
    return digest.hexdigest()


def inventory_sources(defaults=None, limit=200, previous=None):
    require(isinstance(limit, int) and limit > 0, 'limit must be positive')
    defaults = AGENT_DEFAULTS if defaults is None else defaults
    previous = {item['source_id']: item for item in (previous or {}).get('sources', [])}
    sources, coverage, seen = [], [], {}
    for agent, roots in defaults.items():
        for root in roots:
            root = Path(root).expanduser()
            status = {'agent': agent, 'root': str(root), 'status': 'complete', 'discovered': 0}
            if not root.exists():
                coverage.append({**status, 'status': 'missing'})
                continue
            try:
                entries = discover_agent_sessions([agent], limit + 1, defaults={agent: [root]})
            except OSError as error:
                coverage.append({**status, 'status': 'unreadable', 'error': type(error).__name__})
                continue
            status['discovered'] = len(entries[:limit])
            if len(entries) > limit:
                status['status'] = 'limited'
            for entry in entries[:limit]:
                path = Path(entry['path']).resolve()
                source_id = fingerprint([agent, str(path)])[:24]
                if any(item['source_id'] == source_id for item in sources):
                    continue
                item = {**entry, 'path': str(path), 'source_id': source_id, 'status': 'ready'}
                try:
                    item['digest'] = source_digest(path)
                    duplicate = seen.get((agent, item['digest']))
                    if duplicate:
                        item.update(status='duplicate', duplicate_of=duplicate)
                    else:
                        seen[(agent, item['digest'])] = source_id
                        if previous.get(source_id, {}).get('digest') == item['digest']:
                            item['status'] = 'unchanged'
                except (OSError, ValueError) as error:
                    item.update(status='unreadable', error=type(error).__name__)
                    status['status'] = 'partial'
                sources.append(item)
            coverage.append(status)
    removed = sorted(set(previous) - {item['source_id'] for item in sources})
    return {'inventory_version': 1, 'sources': sources, 'coverage': coverage,
            'missing_since_previous': removed, 'complete': all(item['status'] == 'complete' for item in coverage)}
