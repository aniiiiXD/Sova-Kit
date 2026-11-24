"""Apply reviewed tool operations to an in-memory JSON state snapshot."""
from copy import deepcopy

from switchit.contracts import fingerprint, require


def _valid_path_part(part):
    if isinstance(part, str):
        return bool(part)
    return (isinstance(part, dict) and set(part) == {'$ref'}
            and isinstance(part['$ref'], str) and part['$ref'].startswith('arguments.'))


def validate_tool_simulation(tools, allowed_tools):
    require(isinstance(tools, dict), 'tool_simulation must be an object')
    require(set(tools) <= set(allowed_tools), 'tool simulation exceeds workflow tool permissions')
    for name, rule in tools.items():
        require(isinstance(rule, dict), f'simulation rule for {name} must be an object')
        operation = rule.get('operation')
        require(operation in ('read', 'set', 'delete'), f'unsupported simulation operation for {name}')
        fields = {'operation', 'path', 'value'} if operation == 'set' else {'operation', 'path'}
        require(set(rule) == fields, f'invalid simulation fields for {name}')
        path = rule['path']
        require(isinstance(path, list) and path and all(_valid_path_part(part) for part in path),
                f'simulation path for {name} must contain keys or argument bindings')


def apply_tool_operation(state, *, operation, path, value=None):
    require(all(isinstance(part, str) and part for part in path),
            'resolved simulation path keys must be nonempty strings')
    if operation == 'set':
        fingerprint(value)  # Reject non-JSON values before modifying state.
        value = deepcopy(value)
    parent = state
    for key in path[:-1]:
        require(isinstance(parent, dict), 'simulation path crosses a non-object value')
        parent = parent.setdefault(key, {}) if operation == 'set' else parent.get(key, {})
    require(isinstance(parent, dict), 'simulation path crosses a non-object value')
    key = path[-1]
    if operation == 'set':
        parent[key] = value
    found = key in parent
    observed = parent.pop(key, None) if operation == 'delete' else parent.get(key)
    return {'value': deepcopy(observed), 'found': found, 'cost': 0}
