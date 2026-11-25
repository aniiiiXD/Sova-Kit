from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from switchit.contracts import ContractError
from switchit.runtime_adapters import CommandRuntime, FixtureRuntime, ProviderRuntime, RuntimeAdapter, RuntimeRegistry
from switchit.recipes import make_recipe
from switchit.workflow_builder import workflow_revision
from switchit.workflow_evaluation import evaluate_workflow
from switchit.workflow_runner import run_workflow
from tests.workflow_support import pipeline


class WorkflowRuntimeTests(unittest.TestCase):
    def setUp(self):
        _, _, _, self.spec, suite = pipeline()
        self.case = suite['cases'][0]

    def run_fixture(self, **options):
        return run_workflow(self.spec, self.case, FixtureRuntime(self.case), **options)

    def test_recorded_workflow_runs_without_network(self):
        run = self.run_fixture()
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(run['result']['usage']['input_tokens'], 24)
        self.assertEqual(run['result']['usage']['output_tokens'], 6)
        self.assertEqual(evaluate_workflow(self.spec['success_checks'], run)['verdict'], 'pass')
        self.assertEqual(run['metrics']['cost'], 0)

    def test_wrong_revision_and_draft_are_rejected(self):
        self.case['workflow_revision'] = 'different'
        with self.assertRaises(ContractError):
            self.run_fixture()

    def test_step_and_token_budgets_stop_execution(self):
        self.assertEqual(self.run_fixture(max_steps=1)['status'], 'budget_exceeded')
        run = self.run_fixture(max_tokens=5)
        self.assertEqual(run['status'], 'budget_exceeded')
        self.assertEqual(len(run['steps']), 1)

    def test_unknown_usage_does_not_become_zero(self):
        first = self.spec['steps'][0]['id']
        self.case['step_outputs'][first]['usage'] = None
        run = self.run_fixture(max_tokens=100)
        self.assertEqual(run['status'], 'budget_exceeded')
        self.assertIsNone(run['result']['usage']['input_tokens'])

    def test_recipe_cannot_expand_tool_scope(self):
        with self.assertRaises(ContractError):
            self.run_fixture(recipe={'allowed_tools': ['lookup', 'delete']})

    def test_missing_fixture_fails_without_placeholder_execution(self):
        self.case['fixtures']['fixtures'] = []
        run = self.run_fixture()
        self.assertEqual(run['status'], 'error')
        self.assertIn('fixture missing', run['error'])

    def test_branch_missing_input_is_uncertain(self):
        self.spec['steps'][0]['when'] = {'path': 'inputs.enabled', 'equals': True}
        self.spec['revision'] = workflow_revision(self.spec)
        self.case['workflow_revision'] = self.spec['revision']
        self.assertEqual(self.run_fixture()['status'], 'uncertain')

    def test_output_path_cannot_escape_workspace(self):
        self.spec['outputs'] = ['../outside.txt']
        self.spec['revision'] = workflow_revision(self.spec)
        self.case['workflow_revision'] = self.spec['revision']
        with self.assertRaises(ContractError):
            self.run_fixture()

    def test_input_bindings_resolve_before_execution(self):
        step = next(s for s in self.spec['steps'] if s['kind'] == 'tool')
        step['arguments'] = {'account': {'$ref': 'inputs.account'}}
        self.spec['revision'] = workflow_revision(self.spec)
        self.case['workflow_revision'] = self.spec['revision']
        self.case['inputs']['account'] = 'A'
        self.assertEqual(self.run_fixture()['status'], 'completed')

    def test_runtime_requires_explicit_remote_and_tool_authorization(self):
        runtime = FixtureRuntime(self.case)
        runtime.remote = True
        with self.assertRaises(ContractError):
            run_workflow(self.spec, self.case, runtime)
        runtime.remote = False
        runtime.executes_tools = True
        with self.assertRaises(ContractError):
            run_workflow(self.spec, self.case, runtime)


class EvaluationTests(unittest.TestCase):
    def test_missing_artifact_and_state_evidence_stay_unknown(self):
        run = {'status': 'completed', 'result': {'text': 'done'}}
        for kind in ('artifact_equals', 'state_equals'):
            evaluation = evaluate_workflow([{'type': kind, 'path': 'x', 'value': 'done'}], run)
            self.assertEqual(evaluation['verdict'], 'uncertain')

    def test_artifact_paths_with_dots_are_literal_keys(self):
        run = {'status': 'completed', 'result': {}, 'artifacts': {'report.txt': 'correct'}}
        result = evaluate_workflow([{'type': 'artifact_equals', 'path': 'report.txt', 'value': 'correct'}], run)
        self.assertEqual(result['verdict'], 'pass')

    def test_empty_checks_cannot_pass(self):
        self.assertEqual(evaluate_workflow([], {'status': 'completed'})['verdict'], 'uncertain')

    def test_repeated_tools_and_order_are_checked(self):
        calls = [{'name': 'lookup', 'arguments': {'id': n}} for n in (1, 2)]
        run = {'status': 'completed', 'result': {'tool_calls': calls[:1]}}
        result = evaluate_workflow([{'type': 'tool_sequence', 'calls': calls}], run)
        self.assertEqual(result['verdict'], 'fail')


class CommandBridgeTests(unittest.TestCase):
    def test_bridge_is_opt_in_and_unsupported_agents_are_explicit(self):
        with self.assertRaises(ContractError):
            CommandRuntime('codex', ['codex'], version='pinned')
        registry = RuntimeRegistry()
        recipe = make_recipe('test', 'devin', 'pinned', 'revision', [])
        with self.assertRaisesRegex(ContractError, 'unsupported execution runtime'):
            registry.create(recipe, {})

    def test_json_bridge_uses_temporary_workspace(self):
        import tempfile
        with tempfile.TemporaryDirectory() as workspace:
            script = 'import json,sys; x=json.load(sys.stdin); print(json.dumps({"text":x["inputs"]["task"]}))'
            runtime = CommandRuntime('local-test', [sys.executable, '-c', script], version='1', allow_execution=True)
            result = runtime.execute({'kind': 'model'}, {'task': 'hello'}, {'workspace': workspace})
            self.assertEqual(result['text'], 'hello')

    def test_bridge_timeout_and_output_bounds(self):
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as workspace:
            runtime = CommandRuntime('local-test', [sys.executable, '-c', 'import time; time.sleep(1)'],
                                     version='1', allow_execution=True, timeout=0.05)
            with self.assertRaises(subprocess.TimeoutExpired):
                runtime.execute({}, {}, {'workspace': workspace})
            runtime = CommandRuntime('local-test', [sys.executable, '-c', 'print("x"*100)'], version='1',
                                     allow_execution=True, max_output_bytes=10)
            with self.assertRaises(ContractError):
                runtime.execute({}, {}, {'workspace': workspace})


class ModelProposalTests(unittest.TestCase):
    def test_wrong_model_arguments_cannot_be_hidden_by_correct_fixture(self):
        _, _, _, spec, suite = pipeline()
        case = suite['cases'][0]
        first_model = next(step['id'] for step in spec['steps'] if step['kind'] == 'model')
        case['step_outputs'][first_model]['tool_calls'] = [{'name': 'lookup', 'arguments': {'account': 'WRONG'}}]
        run = run_workflow(spec, case, FixtureRuntime(case))
        self.assertEqual(run['status'], 'error')
        self.assertIn('proposal does not match', run['error'])
        self.assertFalse(any(step['kind'] == 'tool' and step['status'] == 'completed' for step in run['steps']))

    def test_unconsumed_model_proposals_make_run_uncertain(self):
        _, _, _, spec, suite = pipeline()
        case = suite['cases'][0]
        final_model = [step['id'] for step in spec['steps'] if step['kind'] == 'model'][-1]
        case['step_outputs'][final_model]['tool_calls'] = [{'name': 'lookup', 'arguments': {'account': 'A'}}]
        run = run_workflow(spec, case, FixtureRuntime(case))
        self.assertEqual(run['status'], 'uncertain')


class ProviderRecipeTests(unittest.TestCase):
    def test_retry_context_and_full_usage_accounting(self):
        from tests.test_replay_correctness import ScriptedAdapter, response
        first = response(tokens=10, latency=4)
        first['error'] = 'temporary failure'
        provider = ScriptedAdapter([first, response('done', tokens=5, latency=2)])
        case = {'request': {'messages': [{'role': 'user', 'content': 'old'}, {'role': 'user', 'content': 'new'}]}}
        runtime = ProviderRuntime(provider, case)
        result = runtime.execute({'kind': 'model'}, {}, {'steps': {}, 'recipe': {
            'system_prompt': 'Be precise', 'config': {'max_attempts': 2, 'context_policy': 'last_user'}}})
        self.assertEqual(result['usage']['input_tokens'], 15)
        self.assertEqual(result['latency_ms'], 6)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(provider.requests[0]['messages'], [{'role': 'system', 'content': 'Be precise'},
                                                           {'role': 'user', 'content': 'new'}])


class StateCaptureTests(unittest.TestCase):
    def test_observed_state_can_be_evaluated_without_inference(self):
        _, _, _, spec, suite = pipeline()
        case = suite['cases'][0]
        case['state_after'] = {'account': {'status': 'active'}}
        run = run_workflow(spec, case, FixtureRuntime(case))
        evaluated = evaluate_workflow([{'type': 'state_equals', 'path': 'account.status', 'value': 'active'}], run)
        self.assertEqual(evaluated['verdict'], 'pass')
