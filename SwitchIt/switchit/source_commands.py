"""Explicitly scoped source inventory/import command surface."""
import json
from pathlib import Path
from .storage import atomic_write_json
from .workflow_commands import read_json


def cmd_inventory(args):
    from .source_inventory import inventory_sources
    roots = read_json(args.roots)
    if not isinstance(roots, dict) or not all(isinstance(paths, list) and all(isinstance(p, str) for p in paths) for paths in roots.values()):
        raise ValueError('roots must map agent names to arrays of directory paths')
    from .importers.registry import IMPORTERS_BY_AGENT
    if not roots or set(roots) - set(IMPORTERS_BY_AGENT):
        raise ValueError('roots must use supported agent importer names')
    inventory = inventory_sources(defaults=roots, limit=args.limit, previous=read_json(args.previous) if args.previous else None)
    atomic_write_json(args.output, inventory)
    print(json.dumps({'sources': len(inventory['sources']), 'complete': inventory['complete'], 'output':args.output}))
    return 0


def cmd_import(args):
    from .import_pipeline import import_inventory, active_events
    result = import_inventory(read_json(args.inventory), store=read_json(args.store) if args.store else None)
    output = Path(args.output)
    atomic_write_json(output/'store.json', result['store'])
    atomic_write_json(output/'events.json', {'events':active_events(result['store'])})
    atomic_write_json(output/'diagnostics.json', result['diagnostics'])
    print(json.dumps({'output':str(output), 'diagnostics':result['diagnostics']}))
    return 1 if any(d['status'] in ('error','unsupported','partial') for d in result['diagnostics']) else 0


def cmd_github(args):
    from .github_source import collect_github, GitHubClient
    from .importers.github import import_github
    from .execution_graph import build_execution_graph
    from .event_store import empty_event_store, merge_events
    from .contracts import fingerprint
    from .redaction import redact_value
    if args.file:
        if args.repo or args.pr or args.issue or args.run or args.remote:
            raise ValueError('--file cannot be combined with remote repository selection')
        document=read_json(args.file)
    else:
        if not args.repo or not args.remote:
            raise ValueError('API collection requires --repo owner/name and --remote; or use --file EXPORT.json offline')
        document=collect_github(args.repo,pr=args.pr,issue=args.issue,run=args.run,
            client=GitHubClient(max_pages=args.max_pages,max_requests=args.max_requests))
    result=import_github(document)
    source_id=fingerprint(['github',result['repository'],document.get('scope',{})])[:24]
    for event in result['events']:
        event['source']['source_id']=source_id
    previous=read_json(args.store) if args.store else empty_event_store()
    store=merge_events(previous,result['events'],{source_id:{'digest':result['revision'],
        'event_ids':[e['event_id'] for e in result['events']],'agent':'github'}})
    output=Path(args.output)
    for name,value in {'export':redact_value(document),'events':{'events':result['events']},'store':store,
        'graph':build_execution_graph(result['events']),'relationships':result['relationships'],
        'diagnostics':{'coverage':result['coverage'],'unresolved_references':result['diagnostics'],'limitations':result['limitations']}}.items():
        atomic_write_json(output/(name+'.json'),value)
    partial=any(item.get('status')!='complete' for item in result['coverage'])
    print(json.dumps({'output':str(output),'events':len(result['events']),'partial':partial,
                      'execution_evidence':False,'repository':result['repository']}))
    return 1 if partial else 0


def cmd_aegis(args):
    from .aegis_source import collect_aegis
    from .importers.aegis import import_aegis
    from .execution_graph import build_execution_graph
    from .event_store import empty_event_store, merge_events
    from .redaction import redact_value
    if args.file:
        if args.url or args.task:
            raise ValueError('--file cannot be combined with --url or --task')
        document = read_json(args.file)
    else:
        if not args.url or not args.task:
            raise ValueError('use --file EXPORT.json or --url http://127.0.0.1:PORT --task TASK_ID')
        document = collect_aegis(args.url, args.task, max_pages=args.max_pages)
    result = import_aegis(document)
    previous = read_json(args.store) if args.store else empty_event_store()
    store = merge_events(previous, result['events'], {result['source_id']: {
        'digest': result['revision'], 'event_ids': [e['event_id'] for e in result['events']], 'agent': 'aegis'}})
    output = Path(args.output)
    diagnostics = {key: result[key] for key in ('coverage', 'diagnostics', 'omissions', 'limitations', 'replay_ready')}
    for name, value in {'export': redact_value(document), 'events': {'events': result['events']},
                        'store': store, 'graph': build_execution_graph(result['events']), 'diagnostics': diagnostics}.items():
        atomic_write_json(output / (name + '.json'), redact_value(value) if name == 'diagnostics' else value)
    partial = any(c['status'] in ('partial', 'limited') for c in result['coverage'])
    print(json.dumps({'output': str(output), 'events': len(result['events']), 'partial': partial,
                      'replay_ready': False, 'source': 'aegis'}))
    return 1 if partial else 0


def cmd_correlate(args):
    from .evidence_correlation import combine_evidence
    from .execution_graph import build_execution_graph
    events=combine_evidence([read_json(args.agent_events),read_json(args.github_events)])
    from .contracts import require
    require(any(e['source'].get('importer') == 'github' for e in events), 'no GitHub evidence supplied')
    require(any(e['source'].get('importer') != 'github' for e in events), 'no agent evidence supplied')
    graph=build_execution_graph(events)
    output=Path(args.output)
    report=graph['correlation']
    for name,document in {'events':{'events':events},'graph':graph,'correlation':report}.items():
        atomic_write_json(output/(name+'.json'),document)
    print(json.dumps({'output':str(output),'events':len(events),'links':len(report['links']),
                      'unresolved':len(report['unresolved'])}))
    return 0


def register(subparsers):
    sub = subparsers.add_parser('sources', help='Inventory explicitly scoped agent roots and import revisioned evidence')
    children = sub.add_subparsers(dest='source_command', required=True)
    p = children.add_parser('inventory')
    p.add_argument('--roots', required=True, help='JSON mapping agent names to directories')
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--previous')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_inventory)
    p = children.add_parser('import')
    p.add_argument('--inventory', required=True)
    p.add_argument('--store', help='Prior revisioned store, if any')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_import)

    p = children.add_parser('github', help='Import GitHub activity evidence from an export or explicit API scope')
    p.add_argument('--file',help='Offline GitHub export JSON')
    p.add_argument('--repo',help='GitHub.com owner/repository')
    p.add_argument('--pr',type=int)
    p.add_argument('--issue',type=int)
    p.add_argument('--run',type=int,help='Actions run ID')
    p.add_argument('--remote',action='store_true',help='Explicitly read selected GitHub API resources')
    p.add_argument('--max-pages',type=int,default=3)
    p.add_argument('--max-requests',type=int,default=60)
    p.add_argument('--store',help='Previous revisioned event store')
    p.add_argument('--output',required=True)
    p.set_defaults(func=cmd_github)

    p = children.add_parser('aegis', help='Import an Aegis task bundle or read an explicitly selected local daemon task')
    p.add_argument('--file', help='Offline aegis-task bundle or aegis-snapshot JSON')
    p.add_argument('--url', help='Local Aegis daemon URL, including port')
    p.add_argument('--task', help='Aegis task ID to read')
    p.add_argument('--max-pages', type=int, default=10)
    p.add_argument('--store', help='Previous revisioned event store')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_aegis)

    p = children.add_parser('correlate',help='Link explicit GitHub references in agent events to imported repository evidence')
    p.add_argument('--agent-events',required=True)
    p.add_argument('--github-events',required=True)
    p.add_argument('--output',required=True)
    p.set_defaults(func=cmd_correlate)
