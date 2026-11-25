from copy import deepcopy
import unittest

from switchit.contracts import ContractError
from switchit.dependencies import profile_dependencies
from switchit.execution_graph import build_execution_graph
from switchit.fixtures import FixtureSession, build_fixtures
from switchit.workflow_builder import build_workflow, review_workflow, validate_workflow, workflow_revision
from switchit.workflow_cases import build_workflow_cases
from switchit.workflow_patterns import discover_workflows
from tests.workflow_support import event, evidence, pipeline


class DependencyTests(unittest.TestCase):
    def test_explicit_requirements_keep_evidence(self):
        record = event('e', payload={'skills': ['research'], 'services': ['memory'],
                'permissions': ['read'], 'environment': {'python': '3.11'},
                'request': {'messages': [{'role': 'system', 'content': 'Be precise'}]}})
        manifest = profile_dependencies([record])
        self.assertEqual(manifest['unknown'], [])
        self.assertTrue(all(item['evidence'] == ['e'] for item in manifest['dependencies']))

    def test_permissions_are_not_invented_from_tool_names(self):
        manifest = profile_dependencies([event('e', 'tool_call', payload={'name': 'send_email'})])
        self.assertIn('permission_requirements_unobserved', manifest['unknown'])
        self.assertFalse(any(item['kind'] == 'permission' for item in manifest['dependencies']))


class WorkflowDiscoveryTests(unittest.TestCase):
    def test_repeated_tasks_group_and_task_changes_segment(self):
        events = evidence() + [event('new-task', 'user_input', index=8, payload={'text': 'Delete abandoned files'}),
                               event('new-answer', index=9, payload={'response': {'text': 'done'}})]
        candidates = discover_workflows(build_execution_graph(events))['candidates']
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(candidates[0]['segments']), 2)

    def test_unknown_tasks_are_not_grouped_by_tools_alone(self):
        graph = build_execution_graph([event('a', trace='a'), event('b', trace='b')])
        candidates = discover_workflows(graph)['candidates']
        self.assertEqual(len(candidates), 2)
        self.assertIn('task_boundary_unknown', candidates[0]['uncertainty'])

    def test_failed_variants_are_retained(self):
        events = evidence()
        events[-1]['payload']['response']['error'] = 'failed'
        candidates = discover_workflows(build_execution_graph(events))['candidates']
        self.assertTrue(any(s['failed'] for s in candidates[0]['segments']))


class BuilderTests(unittest.TestCase):
    def test_draft_requires_review_and_explicit_outcome(self):
        _, graph, candidate, _, _ = pipeline()
        draft = build_workflow(candidate, graph)
        self.assertEqual(draft['status'], 'draft')
        with self.assertRaises(ContractError):
            validate_workflow(draft, require_reviewed=True)
        with self.assertRaises(ContractError):
            review_workflow(draft, reviewer='owner', success_checks=[{'type': 'no_error'}])

    def test_spec_preserves_evidence_and_detects_mutation(self):
        _, graph, _, spec, _ = pipeline()
        self.assertTrue(all(step['evidence'] for step in spec['steps']))
        self.assertTrue(all(e in {n['event_id'] for n in graph['nodes']} for e in spec['evidence']))
        changed = deepcopy(spec)
        changed['name'] = 'changed'
        with self.assertRaises(ContractError):
            validate_workflow(changed)

    def test_cycles_and_dangling_dependencies_are_rejected(self):
        _, _, _, spec, _ = pipeline()
        for dependency in ('missing', spec['steps'][0]['id']):
            changed = deepcopy(spec)
            changed['steps'][0]['depends_on'] = [dependency]
            changed['revision'] = workflow_revision(changed)
            with self.assertRaises(ContractError):
                validate_workflow(changed)

    def test_metadata_only_records_cannot_build_executable_workflow(self):
        graph = build_execution_graph([event('stub', evidence='metadata_only')])
        candidate = discover_workflows(graph)['candidates'][0]
        with self.assertRaises(ContractError):
            build_workflow(candidate, graph)

    def test_branch_contract_is_validated(self):
        _, _, _, spec, _ = pipeline()
        spec['steps'][-1]['when'] = {'path': 'inputs.enabled', 'equals': True}
        spec['revision'] = workflow_revision(spec)
        self.assertEqual(validate_workflow(spec), spec)
        spec['steps'][-1]['when'] = {'python': 'arbitrary code'}
        spec['revision'] = workflow_revision(spec)
        with self.assertRaises(ContractError):
            validate_workflow(spec)


class FixtureTests(unittest.TestCase):
    def test_exact_arguments_and_consumption(self):
        _, graph, _, _, _ = pipeline()
        fixtures = build_fixtures(graph, ['trace-a'])
        session = FixtureSession(fixtures)
        self.assertEqual(session.resolve('lookup', {'account': 'B'})['status'], 'missing')
        self.assertEqual(session.resolve('lookup', {'account': 'A'})['result'], 'active')
        self.assertEqual(session.resolve('lookup', {'account': 'A'})['status'], 'missing')

    def test_repeated_arguments_match_each_occurrence(self):
        records = []
        for index in range(2):
            records.extend([event(f'call-{index}', 'tool_call', index=index*2, tool_call_id=f'id-{index}',
                                  payload={'name': 'lookup', 'arguments': {}}),
                            event(f'result-{index}', 'tool_result', index=index*2+1, tool_call_id=f'id-{index}',
                                  payload={'content': f'value-{index}'})])
        session = FixtureSession(build_fixtures(build_execution_graph(records)))
        self.assertEqual(session.resolve('lookup', {})['result'], 'value-0')
        self.assertEqual(session.resolve('lookup', {})['result'], 'value-1')

    def test_missing_results_do_not_create_fake_fixtures(self):
        graph = build_execution_graph([event('tool', 'tool_call', payload={'name': 'lookup'})])
        fixtures = build_fixtures(graph)
        self.assertEqual(fixtures['fixtures'], [])
        self.assertEqual(fixtures['issues'][0]['reason'], 'missing_or_ambiguous_result')

    def test_cross_trace_results_are_ambiguous_without_scope(self):
        _, graph, _, _, _ = pipeline()
        session = FixtureSession(build_fixtures(graph))
        self.assertEqual(session.resolve('lookup', {'account': 'A'})['status'], 'ambiguous')


class WorkflowCaseTests(unittest.TestCase):
    def test_whole_segments_and_holdout_have_no_trace_leakage(self):
        _, _, _, spec, suite = pipeline()
        self.assertEqual({case['split'] for case in suite['cases']}, {'train', 'holdout'})
        self.assertTrue(all(len(case['events']) == 5 for case in suite['cases']))
        self.assertTrue(all(case['checks'] == spec['success_checks'] for case in suite['cases']))
        self.assertTrue(all(case['fixtures']['fixtures'] for case in suite['cases']))

    def test_construction_evidence_stays_out_of_holdout(self):
        _, graph, _, spec, suite = pipeline()
        constructed = {node['trace_id'] for node in graph['nodes'] if node['event_id'] in spec['evidence']}
        held_out = {case['trace_id'] for case in suite['cases'] if case['split'] == 'holdout'}
        self.assertFalse(constructed & held_out)
        self.assertIn('discovery_used_source_corpus', suite['uncertainty'])

    def test_single_trace_does_not_fabricate_holdout(self):
        _, graph, candidate, spec, _ = pipeline()
        candidate['segments'] = candidate['segments'][:1]
        suite = build_workflow_cases(spec, candidate, graph)
        self.assertIn('independent_holdout_unavailable', suite['uncertainty'])
        self.assertEqual(suite['cases'][0]['split'], 'train')


class SessionSplitTests(unittest.TestCase):
    def test_subtraces_from_one_session_do_not_become_holdout(self):
        events = evidence()
        for event in events:
            event['session_id'] = 'same-session'
        graph = build_execution_graph(events)
        candidate = discover_workflows(graph)['candidates'][0]
        draft = build_workflow(candidate, graph)
        spec = review_workflow(draft, reviewer='owner', success_checks=[{'type': 'output_contains', 'text': 'active'}])
        suite = build_workflow_cases(spec, candidate, graph)
        self.assertEqual({case['split'] for case in suite['cases']}, {'train'})
        self.assertIn('independent_holdout_unavailable', suite['uncertainty'])

    def test_dependencies_bridge_nonexecuting_tool_result_nodes(self):
        _, _, _, spec, _ = pipeline()
        final = spec['steps'][-1]
        tool = next(step for step in spec['steps'] if step['kind'] == 'tool')
        self.assertIn(tool['id'], final['depends_on'])
