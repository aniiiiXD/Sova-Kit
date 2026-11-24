"""Public CLI for evidence reconstruction and reviewed workflow execution."""
from pathlib import Path
import json

from .contracts import require
from .storage import atomic_write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_bundle(document, output):
    from .events import normalize_event, events_from_calls
    from .execution_graph import build_execution_graph
    from .dependencies import profile_dependencies
    from .workflow_patterns import discover_workflows
    from .workflow_builder import build_workflow
    if isinstance(document, dict) and 'event_store_version' in document:
        from .import_pipeline import active_events
        document = active_events(document)
    if isinstance(document, list):
        events = [normalize_event(event) for event in document]
    elif isinstance(document, dict) and 'calls' in document:
        events = events_from_calls(document['calls'])
    else:
        require(isinstance(document, dict) and isinstance(document.get('events'), list),
                'provide an events array, event-store document, or legacy calls store')
        events = [normalize_event(event) for event in document['events']]
    require(events, 'no events supplied')
    graph = build_execution_graph(events)
    candidates = discover_workflows(graph)
    drafts, diagnostics = [], []
    for candidate in candidates['candidates']:
        try:
            drafts.append(build_workflow(candidate, graph))
        except ValueError as error:
            diagnostics.append({'candidate_id': candidate['candidate_id'], 'error': str(error)})
    docs = {'events': {'events': events}, 'graph': graph, 'dependencies': profile_dependencies(events),
            'candidates': candidates, 'drafts': {'workflows': drafts, 'diagnostics': diagnostics}}
    for name, value in docs.items():
        atomic_write_json(Path(output)/(name+'.json'), value)
    return {'output': str(output), 'events': len(events), 'candidates': len(candidates['candidates']),
            'drafts': len(drafts), 'diagnostics': diagnostics}


def cmd_build(args):
    print(json.dumps(build_bundle(read_json(args.events), args.output), indent=2))
    return 0


def cmd_review(args):
    from .workflow_builder import review_workflow
    doc = read_json(args.workflow)
    if 'workflows' in doc:
        drafts = doc['workflows']
        if args.workflow_id:
            drafts = [d for d in drafts if d['workflow_id'] == args.workflow_id]
        require(len(drafts) == 1, 'select a workflow with --workflow-id')
        doc = drafts[0]
    checks = read_json(args.checks)
    require(isinstance(checks, list), 'checks file must contain an array')
    patch = read_json(args.bindings) if args.bindings else {}
    require(isinstance(patch, dict) and set(patch) <= {'inputs', 'outputs', 'steps', 'allowed_tools', 'tool_simulation'},
            'bindings may contain inputs, outputs, steps, allowed_tools, and tool_simulation')
    spec = review_workflow(doc, reviewer=args.reviewer, success_checks=checks, **patch)
    atomic_write_json(args.output, spec)
    print(json.dumps({'workflow': args.output, 'revision': spec['revision']}))
    return 0


def cmd_cases(args):
    from .workflow_cases import build_workflow_cases
    spec, graph, candidates = read_json(args.workflow), read_json(args.graph), read_json(args.candidates)
    matches = [c for c in candidates['candidates'] if c['candidate_id'] == spec['workflow_id']]
    require(len(matches) == 1, 'workflow candidate missing or ambiguous')
    suite = build_workflow_cases(spec, matches[0], graph, args.holdout_fraction)
    atomic_write_json(args.output, suite)
    print(json.dumps({'cases': len(suite['cases']), 'output': args.output, 'uncertainty': suite['uncertainty']}))
    return 0


def execute_case(spec, case, *, runtime='fixture', provider='mock', model=None, remote=False,
                 command=None, allow_execution=False, max_seconds=60, max_tokens=None, recipe=None):
    from .runtime_adapters import FixtureRuntime, ProviderRuntime, CommandRuntime
    from .workflow_runner import run_workflow
    from .workflow_evaluation import case_checks, evaluate_workflow
    from . import providers
    require(case.get('workflow_id') == spec['workflow_id'], 'case belongs to another workflow')
    require(case.get('checks') == spec['success_checks'], 'case checks differ from reviewed criteria')
    if recipe:
        from .recipes import validate_recipe
        validate_recipe(recipe)
        require(recipe['workflow_revision'] == spec['revision'], 'recipe belongs to another workflow revision')
    if runtime == 'fixture':
        adapter = FixtureRuntime(case)
    elif runtime == 'provider':
        from .cli import _live_key
        config = {'model': model, 'timeout': max_seconds}
        if provider in providers.PROVIDER_PRESETS:
            config = providers.resolve_preset(provider, config)
            config['api_key'] = providers.read_preset_key(provider)
        else:
            config['api_key'] = _live_key(provider)
        adapter = ProviderRuntime(providers.build_adapter(provider, config, remote=remote), case)
    elif runtime == 'command':
        require(isinstance(command, dict), 'provide --command-config JSON for command execution')
        require(set(command) <= {'runtime_id', 'argv', 'version', 'remote', 'env', 'models'}, 'unknown command config field')
        adapter = CommandRuntime(**command, allow_execution=allow_execution, allow_remote=remote, timeout=max_seconds)
    else:
        raise ValueError('unknown runtime')
    if recipe:
        require(recipe['runtime'] == adapter.runtime_id and recipe['runtime_version'] == adapter.version,
                'recipe runtime identity/version mismatch')
    run = run_workflow(spec, case, adapter, recipe=recipe, max_seconds=max_seconds, max_tokens=max_tokens,
                       allow_remote=remote, allow_tool_execution=allow_execution)
    return {'run': run, 'evaluation': evaluate_workflow(case_checks(spec['success_checks'], case), run)}


def cmd_run(args):
    spec, case = read_json(args.workflow), read_json(args.case)
    if 'cases' in case:
        selected = [c for c in case['cases'] if not args.case_id or c['case_id'] == args.case_id]
        require(len(selected) == 1, 'select exactly one case with --case-id')
        case = selected[0]
    result = execute_case(spec, case, runtime=args.runtime, provider=args.provider, model=args.model,
                          remote=args.remote, allow_execution=args.allow_execution,
                          command=read_json(args.command_config) if args.command_config else None,
                          recipe=read_json(args.recipe) if args.recipe else None,
                          max_seconds=args.max_seconds, max_tokens=args.max_tokens)
    atomic_write_json(args.output, result)
    print(json.dumps({'output': args.output, 'verdict': result['evaluation']['verdict']}))
    return 0 if result['evaluation']['verdict'] == 'pass' else 1


def cmd_export(args):
    from .skill_export import export_skill
    result = export_skill(read_json(args.workflow), args.output, name=args.name,
                          suite=read_json(args.cases) if args.cases else None)
    print(json.dumps(result, indent=2))
    return 0


def register(subparsers):
    workflow = subparsers.add_parser('workflow', help='Build, review, evaluate, and export complete workflows')
    commands = workflow.add_subparsers(dest='workflow_command', required=True)
    sub = commands.add_parser('build', help='Reconstruct evidence graph and draft workflow contracts')
    sub.add_argument('--events', required=True)
    sub.add_argument('--output', required=True)
    sub.set_defaults(func=cmd_build)
    sub = commands.add_parser('review', help='Pin explicitly supplied reviewer and outcome criteria')
    sub.add_argument('--workflow', required=True)
    sub.add_argument('--workflow-id')
    sub.add_argument('--checks', required=True)
    sub.add_argument('--reviewer', required=True)
    sub.add_argument('--bindings', help='Optional JSON inputs/outputs/steps/allowed_tools/tool_simulation overrides')
    sub.add_argument('--output', required=True)
    sub.set_defaults(func=cmd_review)
    sub = commands.add_parser('cases', help='Build exact fixtures and session-separated cases')
    for flag in ('workflow', 'graph', 'candidates', 'output'):
        sub.add_argument('--'+flag, required=True)
    sub.add_argument('--holdout-fraction', type=float, default=.25)
    sub.set_defaults(func=cmd_cases)
    sub = commands.add_parser('run', help='Execute one reviewed case and evaluate fixed criteria')
    for flag in ('workflow', 'case', 'output'):
        sub.add_argument('--'+flag, required=True)
    sub.add_argument('--case-id')
    sub.add_argument('--runtime', choices=['fixture', 'provider', 'command'], default='fixture')
    sub.add_argument('--provider', default='mock')
    sub.add_argument('--model')
    sub.add_argument('--recipe')
    sub.add_argument('--remote', action='store_true')
    sub.add_argument('--command-config')
    sub.add_argument('--allow-execution', action='store_true')
    sub.add_argument('--max-seconds', type=float, default=60)
    sub.add_argument('--max-tokens', type=int)
    sub.set_defaults(func=cmd_run)
    sub = commands.add_parser('export-skill', help='Export reviewed workflow contract as an agent skill')
    sub.add_argument('--workflow', required=True)
    sub.add_argument('--output', required=True)
    sub.add_argument('--name')
    sub.add_argument('--cases', help='Optional private evidence suite for fixture verification')
    sub.set_defaults(func=cmd_export)
