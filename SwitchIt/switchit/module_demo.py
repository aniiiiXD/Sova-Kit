"""Run the independent workflow modules on synthetic evidence, entirely offline.

Usage: python -m switchit.module_demo --output /tmp/switchit-modules
"""
from __future__ import annotations

import argparse
from pathlib import Path

from switchit.composer import evaluate_candidates, propose_recipes
from switchit.dependencies import profile_dependencies
from switchit.event_store import empty_event_store, merge_events
from switchit.events import normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.promotions import PromotionStore
from switchit.recipes import RecipeRegistry, make_recipe
from switchit.runtime_adapters import FixtureRuntime, RuntimeRegistry
from switchit.storage import atomic_write_json
from switchit.workflow_builder import build_workflow, review_workflow
from switchit.workflow_cases import build_workflow_cases
from switchit.workflow_patterns import discover_workflows


class DemoRuntime(FixtureRuntime):
    """A deterministic positive/negative control, not a benchmark of real agents."""
    runtime_id = 'demo'

    def __init__(self, recipe, case):
        super().__init__(case)
        self.fail = recipe.get('config', {}).get('negative_control', False)

    def execute(self, step, inputs, context):
        result = super().execute(step, inputs, context)
        if self.fail and step['kind'] == 'model':
            result['text'] = 'No verified account status.'
        return result


def synthetic_events():
    events = []
    for trace in ('example-one', 'example-two'):
        rows = [
            ('user_input', {'text': 'Look up the account status'}, {}),
            ('model_call', {'response': {'text': 'Checking the account'},
                            'usage': {'input_tokens': 10, 'output_tokens': 3, 'basis': 'recorded'}}, {}),
            ('tool_call', {'name': 'lookup', 'arguments': {'account': 'A'}}, {'tool_call_id': 'lookup-1', 'parent_id': trace+'-1'}),
            ('tool_result', {'content': 'active'}, {'tool_call_id': 'lookup-1'}),
            ('model_call', {'response': {'text': 'The account is active'},
                            'usage': {'input_tokens': 12, 'output_tokens': 4, 'basis': 'recorded'}}, {}),
        ]
        for index, (kind, payload, links) in enumerate(rows):
            events.append(normalize_event({'event_id': f'{trace}-{index}', 'trace_id': trace,
                'session_id': trace, 'index': index, 'kind': kind, 'payload': payload,
                'source': {'importer': 'synthetic-demo'}, **links}))
    return events


def run_demo(output):
    output = Path(output)
    events = synthetic_events()
    store = merge_events(empty_event_store(), events)
    graph = build_execution_graph(events)
    candidates = discover_workflows(graph)
    candidate = candidates['candidates'][0]
    draft = build_workflow(candidate, graph)
    # Explicit checks/reviewer are part of this synthetic fixture, not inferred approval.
    spec = review_workflow(draft, reviewer='synthetic-demo-fixture', success_checks=[
        {'type': 'no_error'}, {'type': 'output_equals', 'text': 'The account is active'},
        {'type': 'tool_sequence', 'calls': [{'name': 'lookup', 'arguments': {'account': 'A'}}]},
    ])
    suite = build_workflow_cases(spec, candidate, graph)
    baseline = make_recipe('positive-control', 'demo', '1', spec['revision'], ['lookup'])
    proposals = propose_recipes(spec, baseline, [{'name': 'negative-control', 'config': {'negative_control': True}}])
    runtimes = RuntimeRegistry()
    runtimes.register('demo', DemoRuntime)
    experiment = evaluate_candidates(spec, suite, proposals['candidates'], runtimes, objective='cost')
    registry = RecipeRegistry(output/'recipes')
    for recipe in proposals['candidates']:
        registry.put(recipe)
    promotions = PromotionStore(output/'promotion-history.json')
    promotions.promote(experiment, expected_generation=promotions.read()['generation'])
    for name, document in {'events': store, 'graph': graph, 'dependencies': profile_dependencies(events),
                           'workflow-candidates': candidates, 'workflow-draft': draft, 'workflow': spec,
                           'cases': suite, 'proposals': proposals, 'experiment': experiment}.items():
        atomic_write_json(output/(name+'.json'), document)
    summary = {'synthetic': True, 'events': len(events), 'cases': len(suite['cases']),
               'trials': experiment['trials_used'], 'selected_recipe': experiment['selected_recipe'],
               'passing_candidates': sum(c['eligible'] for c in experiment['candidates'])}
    atomic_write_json(output/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Directory for synthetic demo artifacts')
    args = parser.parse_args(argv)
    import json
    print(json.dumps(run_demo(args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
