"""Independent outcome checks over run evidence; absent observations stay unknown."""
from switchit.contracts import fingerprint, require
from switchit.evaluation import evaluate_check


def lookup(document, path):
    current = document
    for part in path.split('.') if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def lookup_observation(document, path):
    if isinstance(document, dict) and path in document:
        return True, document[path]
    return lookup(document, path)


def case_checks(checks, case):
    """Append reviewed case-specific expectations to the workflow's fixed checks."""
    expected = case.get('expected_checks', [])
    require(isinstance(expected, list) and all(isinstance(check, dict) and isinstance(check.get('type'), str)
            for check in expected), 'expected_checks must be an array of outcome checks')
    return [*checks, *expected]


def evaluate_workflow(checks, run):
    findings = []
    result = run.get('result', {})
    records = run.get('steps', [])
    for check in checks:
        kind = check.get('type')
        finding = {'type': kind, 'required': check.get('required', True), 'status': 'unknown', 'detail': ''}
        if kind == 'state_equals' and 'stateful_replay' in run:
            path = check.get('path', '')
            observations = run['stateful_replay']['state_observations']
            if path in observations:
                matches = observations[path] == fingerprint(check.get('value'))
                finding.update(status='pass' if matches else 'fail', detail='simulated state comparison')
            else:
                finding['detail'] = 'no observation captured for the required simulated state path'
        elif kind in ('state_equals', 'artifact_equals', 'artifact_exists'):
            bucket = run.get('state') if kind == 'state_equals' else run.get('artifacts')
            if bucket is not None:
                path = check.get('path', '')
                exists, value = lookup_observation(bucket, path)
                passed = exists if kind == 'artifact_exists' else exists and value == check.get('value')
                finding.update(status='pass' if passed else 'fail', detail='observed state/artifact comparison')
            else:
                finding['detail'] = 'runtime did not capture the required state/artifact evidence'
        elif kind == 'step_before':
            order = [s['step_id'] for s in records if s.get('status') == 'completed']
            left, right = check.get('before'), check.get('after')
            finding.update(status='pass' if left in order and right in order and order.index(left) < order.index(right) else 'fail',
                           detail='completed step ordering')
        elif kind == 'allowed_tools':
            allowed = set(check.get('names', []))
            proposed = [c.get('name') for c in result.get('tool_calls', []) + run.get('proposals', [])]
            executed = [s.get('tool') for s in records if s.get('kind') == 'tool' and s.get('status') == 'completed']
            finding.update(status='pass' if all(name in allowed for name in proposed + executed) else 'fail',
                           detail='tool scope checked against proposals and executed steps')
        elif kind == 'tool_sequence':
            actual = [{'name': t.get('name'), 'arguments': t.get('arguments', {})} for t in result.get('tool_calls', [])]
            expected = [{'name': t.get('name'), 'arguments': t.get('arguments', {})} for t in check.get('calls', [])]
            finding.update(status='pass' if actual == expected else 'fail', detail='tool call order, multiplicity, and arguments')
        elif kind == 'cost_max':
            cost = run.get('metrics', {}).get('cost')
            if cost is not None:
                finding.update(status='pass' if cost <= check.get('value', 0) else 'fail', detail='reported whole-run cost')
            else:
                finding['detail'] = 'cost not reported; tokens are not an invoice'
        else:
            finding.update(evaluate_check(check, result))
        findings.append(finding)
    required = [f for f in findings if f['required']]
    if run.get('status') == 'error':
        verdict = 'error'
    elif any(f['status'] == 'fail' for f in required):
        verdict = 'fail'
    elif not required or any(f['status'] == 'unknown' for f in required) or run.get('status') != 'completed':
        verdict = 'uncertain'
    else:
        verdict = 'pass'
    return {'evaluation_version': 1, 'checks_digest': fingerprint(checks), 'run_digest': fingerprint(run),
            'verdict': verdict, 'checks': findings}
