"""Build reviewable, versioned workflow specifications from recorded executions."""
from copy import deepcopy
from switchit.contracts import fingerprint, require, topological_order, version
from switchit.dependencies import profile_dependencies
from switchit.tool_simulation import validate_tool_simulation


def validate_workflow(spec, require_reviewed=False):
    version(spec, 'workflow_version')
    require(isinstance(spec.get('workflow_id'), str) and bool(spec['workflow_id']), 'workflow_id is required')
    require(isinstance(spec.get('inputs'), dict) and isinstance(spec.get('outputs'), list), 'inputs/outputs must be declared')
    require(spec.get('status') in ('draft', 'reviewed'), 'invalid workflow status')
    steps = spec.get('steps')
    require(isinstance(steps, list) and steps, 'workflow must contain steps')
    if 'tool_proposals_as_output' in spec:
        require(spec['tool_proposals_as_output'] is True and len(steps) == 1 and steps[0].get('kind') == 'model',
                'tool_proposals_as_output requires a single model step')
    for step in steps:
        require(isinstance(step.get('id'), str) and bool(step['id']), 'step ID is required')
        require(step.get('kind') in ('model', 'tool', 'artifact', 'approval', 'handoff'), 'invalid step kind')
        require(isinstance(step.get('depends_on'), list), 'depends_on must be a list')
        if step['kind'] == 'tool':
            require(isinstance(step.get('tool'), str) and bool(step['tool']), 'tool step requires a name')
            if 'requires_proposal' in step:
                require(isinstance(step['requires_proposal'], bool), 'requires_proposal must be a boolean')
        if step.get('when') is not None:
            when = step['when']
            require(isinstance(when, dict) and {'path', 'equals'} <= set(when) <= {'path', 'equals', 'on_missing'},
                    'branch condition needs path and equals, with optional on_missing')
            require(isinstance(when['path'], str), 'branch path must be a string')
            require(when.get('on_missing', 'uncertain') in ('uncertain', 'skip'), 'branch on_missing must be uncertain or skip')
    topological_order([s['id'] for s in steps], [(dep, s['id']) for s in steps for dep in s['depends_on']])
    require(isinstance(spec.get('allowed_tools'), list), 'allowed_tools must be explicit')
    require(all(s['tool'] in spec['allowed_tools'] for s in steps if s['kind'] == 'tool'), 'tool steps exceed workflow scope')
    if 'tool_simulation' in spec:
        validate_tool_simulation(spec['tool_simulation'], spec['allowed_tools'])
    require(isinstance(spec.get('success_checks'), list), 'success_checks must be a list')
    for check in spec['success_checks']:
        require(isinstance(check, dict) and isinstance(check.get('type'), str), 'invalid success check')
    if require_reviewed or spec['status'] == 'reviewed':
        require(spec['status'] == 'reviewed', 'workflow needs explicit review')
        require(any(c.get('required', True) and c.get('type') != 'no_error' for c in spec['success_checks']),
                'at least one required outcome check is necessary')
        require(bool(spec.get('reviewed_by')), 'review attribution is required')
    require(spec.get('revision') == workflow_revision(spec), 'workflow revision does not match content')
    return spec


def workflow_revision(spec):
    return fingerprint({key: value for key, value in spec.items() if key != 'revision'})


def build_workflow(candidate, graph, segment_id=None):
    version(graph, 'graph_version')
    segments = candidate.get('segments', [])
    require(segments, 'candidate contains no segments')
    nodes = {e['event_id']: e for e in graph['nodes']}
    if segment_id:
        segment = next((s for s in segments if s['segment_id'] == segment_id), None)
    else:
        # Metadata may describe the same task but cannot supply executable steps.
        segment = next((s for s in segments if any(
            nodes.get(eid, {}).get('evidence') not in (None, 'metadata_only') and
            nodes.get(eid, {}).get('kind') in ('model_call', 'tool_call', 'artifact', 'approval', 'handoff')
            for eid in s['event_ids'])), segments[0])
    require(segment is not None, 'unknown segment')
    require(all(eid in nodes for eid in segment['event_ids']), 'candidate references missing graph evidence')
    events = [nodes[eid] for eid in segment['event_ids']]
    steps, mapped = [], {}
    kinds = {'model_call': 'model', 'tool_call': 'tool', 'artifact': 'artifact', 'approval': 'approval', 'handoff': 'handoff'}
    for event in events:
        if event['kind'] not in kinds or event['evidence'] == 'metadata_only':
            continue
        sid = 'step_' + fingerprint(event['event_id'])[:12]
        step = {'id': sid, 'kind': kinds[event['kind']], 'depends_on': [], 'evidence': [event['event_id']], 'when': None}
        if step['kind'] == 'model':
            step['request'] = deepcopy(event['payload'].get('request') or {})
            if not step['request'].get('messages'):
                step['request']['messages'] = [{'role': 'user', 'content': {'$ref': 'inputs.task'}}]
        if step['kind'] == 'tool':
            step.update(tool=event['payload'].get('name'), arguments=event['payload'].get('arguments', {}))
            if event['payload'].get('tool_format') == 'custom':
                step.update(tool_format='custom', input_format=event['payload'].get('input_format'),
                            input=deepcopy(event['payload'].get('input')))
        mapped[event['event_id']] = sid
        steps.append(step)
    require(steps, 'metadata-only evidence cannot become an executable workflow')
    by_id = {step['id']: step for step in steps}
    dependency_evidence = []
    incoming = {}
    for edge in graph['edges']:
        if edge.get('causal') is False:
            continue  # Repository references are context, not execution dependencies.
        incoming.setdefault(edge['target'], []).append(edge)
    for event_id, target in mapped.items():
        pending = [(edge, []) for edge in incoming.get(event_id, [])]
        visited = set()
        while pending:
            edge, path = pending.pop()
            source_event = edge['source']
            if source_event in visited:
                continue
            visited.add(source_event)
            path = path + [edge]
            if source_event in mapped:
                source = mapped[source_event]
                if source not in by_id[target]['depends_on']:
                    by_id[target]['depends_on'].append(source)
                dependency_evidence.append({'source': source, 'target': target,
                    'relation': edge['relation'] if len(path) == 1 else 'via_events',
                    'evidence': 'inferred' if any(e['evidence'] == 'inferred' for e in path) else 'recorded',
                    'event_path': [e['source'] for e in reversed(path)] + [event_id]})
            elif source_event in segment['event_ids']:
                pending.extend((previous, path) for previous in incoming.get(source_event, []))
    spec = {'workflow_version': 1, 'workflow_id': candidate['candidate_id'], 'name': candidate.get('task') or 'Untitled workflow',
            'status': 'draft', 'reviewed_by': None,
            'inputs': {'task': {'type': 'string', 'description': 'Task for this recorded workflow; review tool bindings before reuse.'}},
            'outputs': [], 'steps': steps,
            'allowed_tools': sorted({s['tool'] for s in steps if s['kind'] == 'tool'}), 'success_checks': [],
            'evidence': segment['event_ids'], 'source_segment': segment['segment_id'],
            'observed_variants': candidate.get('variants', []),
            'graph_issues': [i for i in graph.get('issues', []) if i['event_id'] in segment['event_ids']], 'dependency_evidence': dependency_evidence,
            'dependencies': profile_dependencies(events),
            'uncertainty': sorted(set(candidate.get('uncertainty', []) + ['success_criteria_require_review', 'ordering_requires_review'] +
                                     (['custom_tools_require_runtime_adapter'] if any(s.get('tool_format') == 'custom' for s in steps) else [])))}
    spec['revision'] = workflow_revision(spec)
    return validate_workflow(spec)


def review_workflow(spec, *, reviewer, success_checks, inputs=None, outputs=None, steps=None,
                    allowed_tools=None, tool_simulation=None):
    validate_workflow(spec)
    result = deepcopy(spec)
    result.update(status='reviewed', reviewed_by=reviewer, success_checks=deepcopy(success_checks))
    for key, value in (('inputs', inputs), ('outputs', outputs), ('steps', steps),
                       ('allowed_tools', allowed_tools), ('tool_simulation', tool_simulation)):
        if value is not None:
            result[key] = deepcopy(value)
    result['uncertainty'] = [u for u in result['uncertainty'] if u not in ('success_criteria_require_review', 'ordering_requires_review')]
    result['revision'] = workflow_revision(result)
    return validate_workflow(result, require_reviewed=True)
