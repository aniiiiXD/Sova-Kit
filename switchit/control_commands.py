"""CLI entry points for local orchestration and platform service."""
import json
from pathlib import Path


def cmd_orchestrate(args):
    from .orchestration import run_plan, status, cancel
    root = Path(args.output)
    if args.action == 'run':
        plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
        if root.joinpath('state.json').exists() and not args.resume:
            raise ValueError('existing run: use --resume with the same immutable plan')
        result = run_plan(plan, root, max_parallel=args.max_parallel)
    elif args.action == 'cancel':
        result = cancel(root)
    else:
        result = status(root)
    print(json.dumps(result, indent=2))
    return 1 if result.get('status') in ('failed', 'cancelled', 'blocked') else 0


def cmd_plan(args):
    from .orchestration import generate_development_plan, smoke_plan
    from .storage import atomic_write_json
    if args.action == 'smoke-plan':
        plan = smoke_plan(args.workspace)
    else:
        config = json.loads(Path(args.workers).read_text(encoding='utf-8'))
        if set(config) - {'workers', 'sources', 'max_parallel', 'repairs'}:
            raise ValueError('worker config accepts workers, sources, max_parallel and repairs')
        plan = generate_development_plan(config['workers'], sources=config['sources'],
                                          max_parallel=config.get('max_parallel',3), repairs=config.get('repairs'))
    atomic_write_json(args.output, plan)
    print(json.dumps({'output':args.output,'tasks':len(plan['tasks']),'publish':False}))
    return 0


def cmd_repair(args):
    from .orchestration.repair import run_repair_loop
    root = Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('existing repair run: use --resume with the same config')
    result = run_repair_loop(json.loads(Path(args.config).read_text(encoding='utf-8')), root)
    print(json.dumps(result, indent=2))
    return 0 if result.get('status') == 'succeeded' else 1


def cmd_serve(args):
    from .platform import serve
    serve(host=args.host, port=args.port, data_dir=args.data_dir,
          switchup_dir=args.switchup_dir, allow_live_models=args.allow_live_models)
    return 0


def register(subparsers):
    sub = subparsers.add_parser('orchestrate', help='Run a durable parallel command DAG')
    commands = sub.add_subparsers(dest='action', required=True)
    for action in ('run', 'status', 'cancel'):
        p = commands.add_parser(action)
        p.add_argument('--output', required=True, help='Persistent graph run directory')
        p.set_defaults(func=cmd_orchestrate)
        if action == 'run':
            p.add_argument('--plan', required=True)
            p.add_argument('--resume', action='store_true')
            p.add_argument('--max-parallel', type=int)
    p = commands.add_parser('plan', help='Generate the development graph from configured worker wrappers')
    p.add_argument('--workers',required=True,help='JSON document with workers, sources and optional max_parallel')
    p.add_argument('--output',required=True)
    p.set_defaults(func=cmd_plan)
    p = commands.add_parser('smoke-plan', help='Generate an executable offline fanout/join example')
    p.add_argument('--workspace',required=True)
    p.add_argument('--output',required=True)
    p.set_defaults(func=cmd_plan)
    p = commands.add_parser('repair', help='Run bounded coder/verifier rounds with frozen criteria')
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p.set_defaults(func=cmd_repair)
    sub = subparsers.add_parser('serve', help='Serve the SwitchUP workspace and workflow evidence API')
    sub.add_argument('--host', default='127.0.0.1')
    sub.add_argument('--port', type=int, default=8787)
    sub.add_argument('--data-dir', default='.switchit/platform')
    sub.add_argument('--switchup-dir', help='Override the SwitchUP workspace directory')
    sub.add_argument('--allow-live-models', action='store_true', help='Enable connected model requests on loopback')
    sub.set_defaults(func=cmd_serve)
