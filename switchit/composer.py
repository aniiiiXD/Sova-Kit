"""Bounded experiments over supplied recipes, with an independent fixed evaluator."""
from copy import deepcopy
from itertools import zip_longest
from statistics import mean
from switchit.contracts import fingerprint, require, version
from switchit.recipes import validate_recipe
from switchit.workflow_builder import validate_workflow
from switchit.workflow_evaluation import case_checks, evaluate_workflow
from switchit.workflow_runner import run_workflow


def validate_suite(spec, suite, *, require_holdout=True):
    validate_workflow(spec, require_reviewed=True)
    version(suite, 'workflow_cases_version')
    require(suite['workflow_revision'] == spec['revision'], 'suite workflow revision mismatch')
    cases = suite['cases']
    require(cases, 'evaluation suite is empty')
    require(len({c['case_id'] for c in cases}) == len(cases), 'duplicate evaluation case IDs')
    require(all(c.get('split') in ('train', 'holdout') for c in cases), 'each case needs a train/holdout split')
    train = {c['trace_id'] for c in cases if c['split'] == 'train'}
    holdout = {c['trace_id'] for c in cases if c['split'] == 'holdout'}
    require(not train & holdout, 'trace leakage across train/holdout split')
    train_groups = {c.get('split_group', c['trace_id']) for c in cases if c['split'] == 'train'}
    holdout_groups = {c.get('split_group', c['trace_id']) for c in cases if c['split'] == 'holdout'}
    require(not train_groups & holdout_groups, 'session leakage across train/holdout split')
    require(not require_holdout or (train and holdout), 'independent train and holdout traces are required')
    for case in cases:
        require(case.get('workflow_revision') == spec['revision'], 'case revision mismatch')
        require(case.get('checks') == spec['success_checks'], 'case attempts to change the fixed evaluation criteria')
        case_checks(spec['success_checks'], case)
    return suite


def compare_trial(baseline, candidate):
    """Summarize the paired observations without treating a missing run as a result."""
    pair = {'baseline_verdict': baseline['evaluation']['verdict'] if baseline else None,
            'candidate_verdict': candidate['evaluation']['verdict'] if candidate else None,
            'initial_state_matches': None, 'final_state_matches': None, 'first_tool_difference': None}
    if not baseline or not candidate:
        return pair
    left, right = baseline['run'], candidate['run']
    left_replay, right_replay = left.get('stateful_replay'), right.get('stateful_replay')
    if left_replay is None or right_replay is None:
        return pair
    pair['initial_state_matches'] = left_replay['initial_state_digest'] == right_replay['initial_state_digest']
    if not pair['initial_state_matches']:
        return pair
    if left['status'] == right['status'] == 'completed':
        pair['final_state_matches'] = left_replay['final_state_digest'] == right_replay['final_state_digest']
    for index, (before, after) in enumerate(zip_longest(left_replay['executed_tools'], right_replay['executed_tools'])):
        if (before is None and left['status'] != 'completed') or (after is None and right['status'] != 'completed'):
            break
        if before is None or after is None or before['call_digest'] != after['call_digest']:
            pair['first_tool_difference'] = {'index': index, 'baseline': before, 'candidate': after}
            break
    return pair


def evaluate_candidates(spec, suite, recipes, runtimes, *, max_candidates=3, max_trials=30,
                        objective='latency_ms', allow_remote=False, allow_tool_execution=False,
                        max_steps=100, max_seconds=60, max_tokens=None, require_holdout=True):
    validate_suite(spec, suite, require_holdout=require_holdout)
    require(objective in ('latency_ms', 'cost'), 'unsupported selection objective')
    require(max_candidates > 0 and max_trials > 0, 'experiment budgets must be positive')
    cases = deepcopy(suite['cases'])
    checks = deepcopy(spec['success_checks'])
    experiment = {'experiment_version': 1, 'workflow_id': spec['workflow_id'], 'workflow_revision': spec['revision'],
                  'suite_digest': fingerprint(suite), 'checks': checks, 'checks_digest': fingerprint(checks),
                  'case_ids': [c['case_id'] for c in cases], 'objective': objective,
                  'require_holdout': require_holdout, 'candidates': [], 'selected_recipe': None}
    expected_checks = {case['case_id']: deepcopy(case['expected_checks']) for case in cases if case.get('expected_checks')}
    if expected_checks:
        experiment.update(expected_checks=expected_checks, expected_checks_digest=fingerprint(expected_checks))
    trials = 0
    require(len({r['recipe_id'] for r in recipes}) == len(recipes), 'duplicate recipe candidates')
    for recipe in list(recipes)[:max_candidates]:
        validate_recipe(recipe)
        require(recipe['workflow_revision'] == spec['revision'], 'recipe workflow revision mismatch')
        entry = {'recipe': deepcopy(recipe), 'trials': [], 'eligible': False, 'score': None, 'errors': []}
        for case in cases:
            if trials >= max_trials:
                entry['errors'].append('experiment trial budget exhausted')
                break
            trials += 1
            try:
                runtime = runtimes.create(recipe, case)
                run = run_workflow(deepcopy(spec), deepcopy(case), runtime, deepcopy(recipe),
                                   allow_remote=allow_remote, allow_tool_execution=allow_tool_execution,
                                   max_steps=max_steps, max_seconds=max_seconds, max_tokens=max_tokens)
                evaluation = evaluate_workflow(case_checks(checks, case), run)
                entry['trials'].append({'case_id': case['case_id'], 'split': case['split'],
                                        'run': run, 'evaluation': evaluation})
            except Exception as error:
                entry['errors'].append(f'{case["case_id"]}: {error}')
        metrics = [t['run']['metrics'].get(objective) for t in entry['trials']]
        entry['eligible'] = (len(entry['trials']) == len(cases) and not entry['errors']
                             and all(t['evaluation']['verdict'] == 'pass' for t in entry['trials'])
                             and all(value is not None for value in metrics))
        if entry['eligible']:
            entry['score'] = mean(metrics)
        experiment['candidates'].append(entry)
    accepted = [entry for entry in experiment['candidates'] if entry['eligible']]
    if accepted:
        selected = min(accepted, key=lambda e: (e['score'], e['recipe']['recipe_id']))
        experiment['selected_recipe'] = selected['recipe']['recipe_id']
    if spec.get('tool_simulation') and experiment['candidates']:
        baseline, *variants = experiment['candidates']
        original_trials = {trial['case_id']: trial for trial in baseline['trials']}
        comparisons = []
        for variant in variants:
            alternative_trials = {trial['case_id']: trial for trial in variant['trials']}
            comparisons.append({'recipe_id': variant['recipe']['recipe_id'], 'cases': [
                {'case_id': case_id, **compare_trial(original_trials.get(case_id), alternative_trials.get(case_id))}
                for case_id in experiment['case_ids']]})
        experiment['paired_replay'] = {'baseline_recipe_id': baseline['recipe']['recipe_id'], 'candidates': comparisons}
    experiment['trials_used'] = trials
    experiment['experiment_id'] = fingerprint(experiment)
    return experiment


def propose_recipes(spec, baseline, variants, *, capabilities=None, limit=3):
    """Compose a bounded set from explicit parameter variants, never changing checks.

    Capabilities are keyed by runtime ID and supplied by the caller's registry.
    Proposals are candidates only; they carry no claim of improved performance.
    """
    from switchit.recipes import recipe_id
    validate_workflow(spec, require_reviewed=True)
    validate_recipe(baseline)
    require(isinstance(limit, int) and limit > 0, 'candidate limit must be positive')
    require(baseline['workflow_revision'] == spec['revision'], 'baseline workflow revision mismatch')
    candidates, rejected, seen = [], [], set()
    for variant in [{}] + list(variants):
        if len(candidates) >= limit:
            break
        try:
            require(isinstance(variant, dict), 'variant must be an object')
            require(set(variant) <= {'name', 'runtime', 'runtime_version', 'model', 'system_prompt', 'allowed_tools', 'config'},
                    'variant attempts to modify protected recipe fields')
            recipe = {**deepcopy(baseline), **deepcopy(variant)}
            recipe['recipe_id'] = recipe_id(recipe)
            validate_recipe(recipe)
            require(set(recipe['allowed_tools']) <= set(spec['allowed_tools']), 'variant expands workflow permissions')
            needed = {step['tool'] for step in spec['steps'] if step['kind'] == 'tool'}
            require(needed <= set(recipe['allowed_tools']), 'variant omits required tools')
            if capabilities is not None:
                caps = capabilities.get(recipe['runtime'])
                require(caps is not None, 'runtime capabilities unavailable')
                require(caps['version'] == recipe['runtime_version'], 'runtime version mismatch')
                require({s['kind'] for s in spec['steps']} <= set(caps['step_kinds']), 'runtime lacks workflow capabilities')
                if recipe.get('model'):
                    require(recipe['model'] in caps['models'], 'model unsupported by runtime')
            if recipe['recipe_id'] not in seen:
                candidates.append(recipe)
                seen.add(recipe['recipe_id'])
        except ValueError as error:
            rejected.append({'variant': deepcopy(variant), 'reason': str(error)})
    return {'recipe_proposals_version': 1, 'workflow_revision': spec['revision'],
            'candidates': candidates, 'rejected': rejected}
