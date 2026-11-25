from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from switchit.composer import evaluate_candidates
from switchit.contracts import ContractError, fingerprint
from switchit.promotions import PromotionStore, validate_selection
from switchit.recipes import RecipeRegistry, make_recipe
from switchit.runtime_adapters import FixtureRuntime, RuntimeRegistry
from tests.workflow_support import pipeline


class ConfiguredFixtureRuntime(FixtureRuntime):
    def __init__(self, case, recipe):
        super().__init__(case)
        self.config = recipe.get('config', {})

    def execute(self, step, inputs, context):
        result = super().execute(step, inputs, context)
        if self.config.get('fail') and step['kind'] == 'model':
            result['text'] = 'incorrect'
        if self.config.get('holdout_fail') and self.case['split'] == 'holdout' and step['kind'] == 'model':
            result['text'] = 'incorrect'
        result['cost'] = self.config.get('cost', 1)
        return result


class ComposerTests(unittest.TestCase):
    def setUp(self):
        _, _, _, self.spec, self.suite = pipeline()
        self.registry = RuntimeRegistry()
        self.registry.register('fixtures', lambda recipe, case: ConfiguredFixtureRuntime(case, recipe))

    def recipe(self, name, **config):
        return make_recipe(name, 'fixtures', '1', self.spec['revision'], ['lookup'], config=config)

    def evaluate(self, recipes, **options):
        return evaluate_candidates(self.spec, self.suite, recipes, self.registry, objective='cost', **options)

    def test_failed_cheap_recipe_loses_to_passing_recipe(self):
        cheap = self.recipe('cheap', cost=0, fail=True)
        passing = self.recipe('passing', cost=2)
        result = self.evaluate([cheap, passing])
        self.assertEqual(result['selected_recipe'], passing['recipe_id'])
        self.assertFalse(result['candidates'][0]['eligible'])
        self.assertEqual(validate_selection(result), passing)

    def test_holdout_failure_disqualifies_candidate(self):
        result = self.evaluate([self.recipe('overfit', holdout_fail=True)])
        self.assertIsNone(result['selected_recipe'])
        self.assertTrue(any(t['evaluation']['verdict'] == 'fail' for t in result['candidates'][0]['trials']))

    def test_budgets_never_select_partial_evaluations(self):
        result = self.evaluate([self.recipe('one'), self.recipe('two')], max_trials=1)
        self.assertEqual(result['trials_used'], 1)
        self.assertIsNone(result['selected_recipe'])
        result = self.evaluate([self.recipe('one'), self.recipe('two')], max_candidates=1)
        self.assertEqual(len(result['candidates']), 1)

    def test_rejects_leakage_and_changed_criteria(self):
        original = deepcopy(self.suite)
        self.suite['cases'][1]['trace_id'] = self.suite['cases'][0]['trace_id']
        with self.assertRaisesRegex(ContractError, 'trace leakage'):
            self.evaluate([self.recipe('test')])
        self.suite = original
        self.suite['cases'][0]['checks'] = [{'type': 'no_error'}]
        with self.assertRaisesRegex(ContractError, 'fixed evaluation'):
            self.evaluate([self.recipe('test')])

    def test_runtime_version_mismatch_is_ineligible(self):
        recipe = make_recipe('mismatch', 'fixtures', '2', self.spec['revision'], ['lookup'])
        result = self.evaluate([recipe])
        self.assertIsNone(result['selected_recipe'])
        self.assertIn('runtime version mismatch', result['candidates'][0]['errors'][0])

    def test_inputs_are_not_mutated_by_trials(self):
        before = deepcopy((self.spec, self.suite))
        self.evaluate([self.recipe('test')])
        self.assertEqual(before, (self.spec, self.suite))

    def test_missing_holdout_is_rejected(self):
        self.suite['cases'] = [self.suite['cases'][0]]
        with self.assertRaisesRegex(ContractError, 'holdout'):
            self.evaluate([self.recipe('test')])

    def test_unknown_cost_cannot_win_cost_selection(self):
        registry = RuntimeRegistry()
        class UnknownCost(FixtureRuntime):
            def execute(self, *args):
                return {**super().execute(*args), 'cost': None}
        registry.register('fixtures', lambda recipe, case: UnknownCost(case))
        result = evaluate_candidates(self.spec, self.suite, [self.recipe('test')], registry, objective='cost')
        self.assertIsNone(result['selected_recipe'])

    def test_registry_is_immutable_and_path_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = RecipeRegistry(folder)
            recipe = self.recipe('test')
            key = registry.put(recipe)
            self.assertEqual(registry.get(key), recipe)
            with self.assertRaises(ContractError):
                registry.get('../../outside')
            recipe['config']['cost'] = 100
            with self.assertRaises(ContractError):
                registry.put(recipe)

    def test_promotion_and_rollback_keep_history(self):
        experiment = self.evaluate([self.recipe('test')])
        with tempfile.TemporaryDirectory() as folder:
            promotions = PromotionStore(Path(folder)/'active.json')
            state = promotions.promote(experiment, expected_generation=0)
            self.assertEqual(state['generation'], 1)
            with self.assertRaises(ContractError):
                promotions.promote(experiment, expected_generation=0)
            state = promotions.rollback(self.spec['workflow_id'], expected_generation=1)
            self.assertEqual(state['active'], {})
            self.assertEqual(len(state['history']), 2)

    def test_tampered_or_failed_evidence_cannot_promote(self):
        experiment = self.evaluate([self.recipe('test')])
        experiment['candidates'][0]['trials'][0]['run']['result']['text'] = 'wrong'
        experiment['experiment_id'] = fingerprint({k:v for k,v in experiment.items() if k != 'experiment_id'})
        with self.assertRaises(ContractError):
            validate_selection(experiment)
        failed = self.evaluate([self.recipe('bad', fail=True)])
        with self.assertRaisesRegex(ContractError, 'retain the baseline'):
            validate_selection(failed)


class RecipeProposalTests(unittest.TestCase):
    def test_variants_are_bounded_and_cannot_change_checks_or_permissions(self):
        from switchit.composer import propose_recipes
        _, _, _, spec, suite = pipeline()
        baseline = make_recipe('base', 'fixtures', '1', spec['revision'], ['lookup'])
        proposals = propose_recipes(spec, baseline, [
            {'allowed_tools': ['lookup', 'delete']},
            {'success_checks': []},
            {'name': 'second', 'system_prompt': 'Use evidence'},
            {'name': 'third'},
        ], limit=2)
        self.assertEqual(len(proposals['candidates']), 2)
        self.assertEqual(len(proposals['rejected']), 2)
        self.assertEqual(proposals['candidates'][1]['name'], 'second')
