"""Bounded recipe comparison and local promotion controls."""
import json
from .contracts import require
from .storage import atomic_write_json
from .workflow_commands import read_json


def cmd_propose(args):
    from .composer import propose_recipes
    result = propose_recipes(read_json(args.workflow), read_json(args.baseline), read_json(args.variants), limit=args.limit)
    atomic_write_json(args.output, result)
    print(json.dumps({'output':args.output,'candidates':len(result['candidates']),'rejected':result['rejected']}))
    return 0


def cmd_compare(args):
    from .composer import evaluate_candidates
    from .runtime_adapters import RuntimeRegistry, FixtureRuntime, ProviderRuntime
    from . import providers
    from .cli import _live_key
    config = read_json(args.runtimes) if args.runtimes else {}
    require(isinstance(config, dict) and set(config) <= {'provider'}, 'runtime config supports provider only; fixture is built in')
    registry = RuntimeRegistry()
    registry.register('fixtures', lambda recipe, case: FixtureRuntime(case))
    if 'provider' in config:
        settings = config['provider']
        require(isinstance(settings,dict) and set(settings) <= {'kind','base_url'}, 'provider settings accept kind/base_url only')
        kind = settings.get('kind','mock')
        def factory(recipe, case):
            options = {'model':recipe.get('model'), 'timeout':args.max_seconds, 'base_url':settings.get('base_url')}
            if kind in providers.PROVIDER_PRESETS:
                options = providers.resolve_preset(kind, options)
                options['api_key'] = providers.read_preset_key(kind)
            else:
                options['api_key'] = _live_key(kind)
            return ProviderRuntime(providers.build_adapter(kind, options, remote=args.remote), case)
        registry.register('provider', factory)
    recipes = read_json(args.recipes)
    if isinstance(recipes,dict):
        recipes = recipes['candidates']
    experiment = evaluate_candidates(read_json(args.workflow), read_json(args.cases), recipes, registry,
        max_candidates=args.max_candidates, max_trials=args.max_trials, objective=args.objective,
        allow_remote=args.remote, max_seconds=args.max_seconds, max_tokens=args.max_tokens)
    atomic_write_json(args.output, experiment)
    print(json.dumps({'output':args.output,'selected_recipe':experiment['selected_recipe'],'trials':experiment['trials_used']}))
    return 0 if experiment['selected_recipe'] else 1


def cmd_promotion(args):
    from .promotions import PromotionStore
    store = PromotionStore(args.store)
    if args.harness_command == 'promote':
        result = store.promote(read_json(args.experiment), expected_generation=args.expected_generation)
    elif args.harness_command == 'rollback':
        result = store.rollback(args.workflow_id, expected_generation=args.expected_generation)
    else:
        result = store.read()
    print(json.dumps(result, indent=2))
    return 0


def register(subparsers):
    sub = subparsers.add_parser('harness', help='Propose and compare pinned recipes, then promote locally')
    children = sub.add_subparsers(dest='harness_command', required=True)
    p = children.add_parser('propose')
    for key in ('workflow','baseline','variants','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--limit',type=int,default=3)
    p.set_defaults(func=cmd_propose)
    p = children.add_parser('compare')
    for key in ('workflow','cases','recipes','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--runtimes',help='Provider connection config JSON; keys remain in environment')
    p.add_argument('--remote',action='store_true')
    p.add_argument('--max-candidates',type=int,default=3)
    p.add_argument('--max-trials',type=int,default=30)
    p.add_argument('--max-seconds',type=float,default=60)
    p.add_argument('--max-tokens',type=int)
    p.add_argument('--objective',choices=['cost','latency_ms'],default='cost')
    p.set_defaults(func=cmd_compare)
    for action in ('promote','rollback','status'):
        p = children.add_parser(action)
        p.add_argument('--store',required=True)
        if action != 'status':
            p.add_argument('--expected-generation',required=True,type=int)
        if action == 'rollback':
            p.add_argument('--workflow-id',required=True)
        if action == 'promote':
            p.add_argument('--experiment',required=True)
        p.set_defaults(func=cmd_promotion)
