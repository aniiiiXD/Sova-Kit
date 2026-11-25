"""Versioned JSON contracts shared by the independently usable workflow modules."""
from __future__ import annotations

import hashlib
import json
import math


class ContractError(ValueError):
    """An artifact is invalid or incompatible; do not silently reinterpret it."""


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ContractError(message)


def version(document, key, expected=1):
    require(isinstance(document, dict), 'artifact must be an object')
    require(document.get(key) == expected, f'unsupported {key}: {document.get(key)!r}')


def nonnegative(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f'{name} must be finite and nonnegative')


def topological_order(nodes, edges):
    """Stable ordering; reject duplicate IDs, dangling links, and cycles."""
    ids = list(nodes)
    require(len(ids) == len(set(ids)), 'duplicate node IDs')
    import heapq
    incoming = {key: set() for key in ids}
    outgoing = {key: set() for key in ids}
    for source, target in edges:
        require(source in incoming and target in incoming, f'dangling edge: {source} -> {target}')
        incoming[target].add(source)
        outgoing[source].add(target)
    ready = [key for key in ids if not incoming[key]]
    heapq.heapify(ready)
    result = []
    while ready:
        key = heapq.heappop(ready)
        result.append(key)
        for target in outgoing[key]:
            incoming[target].remove(key)
            if not incoming[target]:
                heapq.heappush(ready, target)
    require(len(result) == len(ids), 'dependency cycle')
    return result
