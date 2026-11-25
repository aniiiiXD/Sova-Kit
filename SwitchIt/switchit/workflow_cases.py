"""Whole-segment cases and deterministic train/holdout splits without trace leakage."""
from copy import deepcopy
from switchit.contracts import fingerprint, require
from switchit.fixtures import build_fixtures
from switchit.workflow_builder import validate_workflow


def build_workflow_cases(spec, candidate, graph, holdout_fraction=0.25):
    validate_workflow(spec, require_reviewed=True)
    require(0 < holdout_fraction < 1, 'holdout_fraction must be between zero and one')
    nodes = {e['event_id']: e for e in graph['nodes']}
    candidate = deepcopy(candidate)
    eligible, excluded = [], []
    for segment in candidate['segments']:
        require(all(eid in nodes for eid in segment['event_ids']), 'candidate references missing evidence')
        if any(nodes[eid]['evidence'] != 'metadata_only' for eid in segment['event_ids']):
            eligible.append(segment)
        else:
            excluded.append(segment['segment_id'])
    require(eligible, 'metadata-only segments cannot supply replay cases')
    candidate['segments'] = eligible
    group_by_trace = {}
    for segment in candidate['segments']:
        sessions = {nodes[eid].get('session_id') for eid in segment['event_ids']} - {None}
        require(len(sessions) <= 1, 'trace spans multiple sessions; resolve its identity before splitting')
        group_by_trace[segment['trace_id']] = next(iter(sessions)) if sessions else segment['trace_id']
    traces = sorted(set(group_by_trace.values()), key=fingerprint)
    count = max(1, min(len(traces)-1, round(len(traces)*holdout_fraction))) if len(traces) > 1 else 0
    # Evidence used to construct the reviewed steps cannot be held out.
    construction_traces = {nodes[eid]['trace_id'] for eid in spec.get('evidence', []) if eid in nodes}
    construction_groups = {group_by_trace[trace] for trace in construction_traces if trace in group_by_trace}
    eligible = [group for group in traces if group not in construction_groups]
    holdout = set(eligible[:count])
    cases = []
    for segment in candidate['segments']:
        events = [nodes[eid] for eid in segment['event_ids']]
        require(any(e['evidence'] != 'metadata_only' for e in events), 'metadata-only segment is not replayable')
        subgraph = {**graph, 'nodes': events, 'edges': [edge for edge in graph['edges']
                    if edge['source'] in segment['event_ids'] and edge['target'] in segment['event_ids']]}
        model_steps = [step for step in spec['steps'] if step['kind'] == 'model']
        models = [event for event in events if event['kind'] == 'model_call']
        step_outputs = {}
        if len(model_steps) == len(models):
            for step, event in zip(model_steps, models):
                step_outputs[step['id']] = {**deepcopy(event['payload'].get('response') or {}),
                                            'usage': deepcopy(event['payload'].get('usage')), 'cost': 0}
        cases.append({'step_outputs': step_outputs, 'case_id': fingerprint([spec['workflow_id'], segment['segment_id']])[:24],
                      'workflow_id': spec['workflow_id'], 'workflow_revision': spec['revision'],
                      'trace_id': segment['trace_id'], 'split_group': group_by_trace[segment['trace_id']],
                      'split': 'holdout' if group_by_trace[segment['trace_id']] in holdout else 'train',
                      'inputs': {'task': segment['task']}, 'events': deepcopy(events),
                      'fixtures': build_fixtures(subgraph), 'checks': deepcopy(spec['success_checks']),
                      'tags': ['failed'] if segment['failed'] else ['observed']})
    return {'workflow_cases_version': 1, 'workflow_revision': spec['revision'], 'cases': cases,
            'excluded_metadata_segments': excluded,
            'uncertainty': ['discovery_used_source_corpus'] + (['metadata_only_segments_excluded'] if excluded else [])
                           + ([] if holdout else ['independent_holdout_unavailable'])}
