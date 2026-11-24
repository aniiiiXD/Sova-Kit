"""Evidence-gated configuration activation and rollback; never undo business effects."""
from copy import deepcopy
from pathlib import Path
import json
from statistics import mean
from switchit.contracts import fingerprint, require, version
from switchit.recipes import RecipeRegistry, validate_recipe
from switchit.storage import atomic_write_json
from switchit.workflow_evaluation import case_checks, evaluate_workflow


def validate_selection(experiment):
    version(experiment, 'experiment_version')
    require(experiment['experiment_id'] == fingerprint({k:v for k,v in experiment.items() if k != 'experiment_id'}), 'experiment content changed')
    require(experiment['checks_digest'] == fingerprint(experiment['checks']), 'evaluation criteria changed')
    chosen = experiment.get('selected_recipe')
    require(chosen, 'no accepted configuration; retain the baseline')
    candidates = [e for e in experiment['candidates'] if e['recipe']['recipe_id'] == chosen]
    require(len(candidates) == 1, 'selection does not identify exactly one recipe')
    candidate = candidates[0]
    validate_recipe(candidate['recipe'])
    require(candidate['recipe']['workflow_revision'] == experiment['workflow_revision'], 'workflow revision mismatch')
    require(candidate['eligible'] and not candidate['errors'], 'candidate is not eligible')
    require([t['case_id'] for t in candidate['trials']] == experiment['case_ids'], 'incomplete evaluation coverage')
    if experiment['require_holdout']:
        require({t['split'] for t in candidate['trials']} == {'train', 'holdout'}, 'holdout evaluation missing')
    require(experiment['case_ids'] and len(set(experiment['case_ids'])) == len(experiment['case_ids']), 'invalid case coverage')
    expected = experiment.get('expected_checks', {})
    require(isinstance(expected, dict) and set(expected) <= set(experiment['case_ids']), 'invalid case-specific expectations')
    if expected:
        require(experiment.get('expected_checks_digest') == fingerprint(expected), 'case-specific expectations changed')
    observed_scores = [trial['run']['metrics'].get(experiment['objective']) for trial in candidate['trials']]
    require(all(value is not None for value in observed_scores) and mean(observed_scores) == candidate['score'], 'selection score does not match run measurements')
    for trial in candidate['trials']:
        require(trial['run']['workflow_id'] == experiment['workflow_id'] and trial['run']['workflow_revision'] == experiment['workflow_revision'], 'run belongs to a different workflow')
        require(trial['run']['case_id'] == trial['case_id'], 'run belongs to a different case')
        require(trial['run']['runtime']['runtime_id'] == candidate['recipe']['runtime'] and trial['run']['runtime']['version'] == candidate['recipe']['runtime_version'], 'run used a different runtime')
        require(trial['run']['recipe_digest'] == fingerprint(candidate['recipe']), 'run evaluated a different recipe')
        checks = case_checks(experiment['checks'], {'expected_checks': expected.get(trial['case_id'], [])})
        evaluation = evaluate_workflow(checks, trial['run'])
        require(evaluation == trial['evaluation'] and evaluation['verdict'] == 'pass', 'evaluation evidence failed verification')
    return candidate['recipe']


class PromotionStore:
    """Single-writer local configuration history; revision checks detect stale callers."""
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'promotions_version': 1, 'generation': 0, 'active': {}, 'history': []}

    def promote(self, experiment, *, expected_generation):
        recipe = validate_selection(experiment)
        state = self.read()
        require(state['generation'] == expected_generation, 'promotion state changed; reload before applying')
        # Persist the exact accepted recipe and evidence before the active pointer.
        RecipeRegistry(self.path.parent / 'recipes').put(recipe)
        evidence_path = self.path.parent / 'experiments' / (experiment['experiment_id'] + '.json')
        if evidence_path.exists():
            require(json.loads(evidence_path.read_text()) == experiment, 'immutable experiment collision')
        else:
            atomic_write_json(evidence_path, experiment)
        workflow = experiment['workflow_id']
        previous = state['active'].get(workflow)
        active = {'recipe_id': recipe['recipe_id'], 'workflow_revision': experiment['workflow_revision'],
                  'experiment_id': experiment['experiment_id']}
        state['history'].append({'action': 'promote', 'workflow_id': workflow, 'previous': previous, 'next': active})
        state['active'][workflow] = active
        state['generation'] += 1
        atomic_write_json(self.path, state)
        return deepcopy(state)

    def rollback(self, workflow_id, *, expected_generation):
        state = self.read()
        require(state['generation'] == expected_generation, 'promotion state changed; reload before applying')
        current = state['active'].get(workflow_id)
        last = next((h for h in reversed(state['history']) if h['workflow_id'] == workflow_id), None)
        require(last and last['action'] == 'promote' and last['next'] == current, 'no current promotion to roll back')
        previous = last['previous']
        state['history'].append({'action': 'rollback', 'workflow_id': workflow_id, 'previous': current, 'next': previous})
        if previous is None:
            state['active'].pop(workflow_id, None)
        else:
            state['active'][workflow_id] = previous
        state['generation'] += 1
        atomic_write_json(self.path, state)
        return deepcopy(state)
