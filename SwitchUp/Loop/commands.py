"""CLI and MCP entry points to the SwitchUP workspace."""
import json
from pathlib import Path

from switchit.contracts import require
from switchit.data_paths import workspace_dir
from switchit.storage import atomic_write_json
from switchit.workflow_commands import read_json

from . import loop, traces
from .contracts import chat_workflow
from .store import Workspace, project_summary, round_summary


def _emit(value, output=None):
    if output:
        atomic_write_json(output, value)
        print(json.dumps({'output': output}))
    else:
        print(json.dumps(value, indent=2, ensure_ascii=False))


def create(args, workspace):
    if args.workflow:
        require(not args.reviewer and not args.tools, 'reviewer and tools are supplied by the reviewed workflow')
        workflow = read_json(args.workflow)
    else:
        workflow = chat_workflow(args.project, read_json(args.checks), args.reviewer,
                                 read_json(args.tools) if args.tools else [])
    prompt = Path(args.prompt).read_text(encoding='utf-8') if args.prompt else ''
    project = loop.initialize(workspace, args.project, read_json(args.connection), workflow, prompt,
                              read_json(args.cases) if args.cases else None)
    _emit(project_summary(project))


def ingest(args, workspace):
    _emit(traces.ingest_events(workspace, args.project, traces.read_events(args.events)))


def feedback(args, workspace):
    _emit(workspace.feedback(args.project, args.trace, args.correction, args.event_id))


def snapshot(args, workspace):
    if args.cases:
        project = workspace.install_suite(args.project, read_json(args.cases),
                                          expected_generation=args.expected_generation)
    else:
        project = traces.pin_chat_suite(workspace, args.project, read_json(args.holdout),
                                        expected_generation=args.expected_generation, max_cases=args.max_cases)
    _emit(project_summary(project))


def prepare(args, workspace):
    record = workspace.prepare_round(args.project, args.evidence_limit)
    if args.output:
        atomic_write_json(args.output, {'round': round_summary(record), 'baseline': record['baseline'], 'evidence': record['evidence']})
    _emit({**round_summary(record), 'event_ids': [event['event_id'] for event in record['evidence']], 'output': args.output})


def propose(args, workspace):
    record = loop.propose(workspace, args.project, args.round,
                          patch=read_json(args.proposal) if args.proposal else None,
                          optimizer=read_json(args.optimizer) if args.optimizer else None,
                          allow_live=args.live, timeout=args.timeout)
    _emit({**round_summary(record), 'candidate_recipe_id': record['candidate']['recipe_id']})


def verify(args, workspace):
    record = loop.verify(workspace, args.project, args.round, allow_live=args.live,
                         timeout=args.timeout, max_model_calls=args.max_model_calls, max_tokens=args.max_tokens)
    _emit(round_summary(record))
    return 0 if record['status'] == 'verified' else 1


def cycle(args, workspace):
    require(args.live, 'authorize the optimizer and replay legs with --live')
    record = loop.run_round(workspace, args.project, optimizer=read_json(args.optimizer), allow_live=True,
                            timeout=args.timeout, max_model_calls=args.max_model_calls, max_tokens=args.max_tokens,
                            evidence_limit=args.evidence_limit, activate_verified=args.activate)
    _emit(round_summary(record))
    return 0 if record['status'] in ('verified', 'activated') else 1


def activate(args, workspace):
    _emit(project_summary(loop.activate(workspace, args.project, args.round, expected_generation=args.expected_generation)))


def rollback(args, workspace):
    _emit(project_summary(workspace.rollback(args.project, args.expected_generation)))


def abandon(args, workspace):
    record = workspace.round(args.project, args.round)
    require(record['status'] in ('prepared', 'proposing', 'proposed', 'verifying'), 'this round is already finished')
    _, record = workspace.transition(args.project, args.round, record['status'], 'abandoned')
    _emit(round_summary(record))


def status(args, workspace):
    _emit({'projects': workspace.status(args.project)})


def active(args, workspace):
    project = workspace.project(args.project)
    _emit({'project': project['name'], 'generation': project['generation'], 'connection': project['connection'],
           'recipe': project['active_recipe']}, args.output)


def show(args, workspace):
    _emit(workspace.round(args.project, args.round), args.output)


def export(args, workspace):
    value = workspace.suite(args.project, args.suite_digest) if args.suite_digest else {'events': list(workspace.events(args.project))}
    _emit(value, args.output)


def chat(args, workspace):
    result = traces.invoke(workspace, args.project, read_json(args.request), allow_live=args.live, timeout=args.timeout)
    _emit(result, args.output)
    return 0 if result['status'] == 'completed' else 1


def dispatch(args):
    try:
        return args.action(args, Workspace(args.state_dir)) or 0
    except KeyError as error:
        raise ValueError(f'missing SwitchUP project, artifact, or field {error}') from error


def register(subparsers):
    section = subparsers.add_parser('improve', help='SwitchUP projects, trace-driven updates, and gated releases')
    commands = section.add_subparsers(dest='improve_command', required=True)
    state_dir = str(workspace_dir() / 'platform' / 'switchup')

    def add(name, handler, help_text, *, round_arg=False, generation=False):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument('--state-dir', default=state_dir, help='SwitchUP workspace directory')
        command.add_argument('--project', required=name != 'status', help='Project name')
        if round_arg:
            command.add_argument('--round', required=True, help='Improvement round ID')
        if generation:
            command.add_argument('--expected-generation', required=True, type=int, help='Current project generation')
        command.set_defaults(func=dispatch, action=handler)
        return command

    command = add('init', create, 'Create a project with a pinned target model and reviewed checks')
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument('--workflow', help='Reviewed workflow JSON')
    source.add_argument('--checks', help='Outcome check array for a captured chat workflow')
    command.add_argument('--connection', required=True, help='Model connection JSON')
    command.add_argument('--prompt', help='Initial project instructions as a text file')
    command.add_argument('--reviewer', help='Reviewer for chat outcome checks')
    command.add_argument('--tools', help='Chat tool allowlist JSON')
    command.add_argument('--cases', help='Reviewed replay suite JSON')

    command = add('ingest', ingest, 'Import canonical evidence from SwitchIt or an instrumented agent')
    command.add_argument('--events', required=True, help='Canonical events, event store, or calls JSON')
    command = add('feedback', feedback, 'Link a correction to a captured trace')
    command.add_argument('--trace', required=True)
    command.add_argument('--correction', required=True)
    command.add_argument('--event-id', help='Stable ID for idempotent feedback')
    command = add('snapshot', snapshot, 'Pin a train and holdout case suite', generation=True)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument('--cases', help='Reviewed replay suite JSON')
    source.add_argument('--holdout', help='JSON array of held-out trace IDs for chat replay')
    command.add_argument('--max-cases', type=int, default=30)

    command = add('prepare', prepare, 'Reserve training evidence for a new improvement round')
    command.add_argument('--evidence-limit', type=int, default=20)
    command.add_argument('--output', help='Write the training packet to this JSON file')
    command = add('propose', propose, 'Submit written instructions or ask a configured model for a candidate', round_arg=True)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument('--proposal', help='Written proposal JSON')
    source.add_argument('--optimizer', help='Optimizer model connection JSON')
    command.add_argument('--live', action='store_true', help='Authorize sending the training packet to the configured model')
    command.add_argument('--timeout', type=float, default=45)
    verifier = add('verify', verify, 'Replay both versions against the same reviewed cases', round_arg=True)
    automatic = add('cycle', cycle, 'Prepare one model-generated improvement and verify it')
    automatic.add_argument('--optimizer', required=True, help='Optimizer model connection JSON')
    automatic.add_argument('--evidence-limit', type=int, default=20)
    automatic.add_argument('--activate', action='store_true', help='Activate the candidate if it passes the release gate')
    for command in (verifier, automatic):
        command.add_argument('--live', action='store_true', help='Authorize connected model requests for this run')
        command.add_argument('--timeout', type=float, default=45)
        command.add_argument('--max-model-calls', type=int, default=80)
        command.add_argument('--max-tokens', type=int, help='Optional observed token budget per case')

    add('activate', activate, 'Activate a verified candidate', round_arg=True, generation=True)
    add('rollback', rollback, 'Restore the previous active recipe', generation=True)
    add('abandon', abandon, 'Close an unfinished round and preserve its evidence', round_arg=True)
    add('status', status, 'Inspect projects, active versions, and recent rounds')
    add('active', active, 'Read the version used for new requests').add_argument('--output')
    add('show', show, 'Inspect an improvement round and its evidence', round_arg=True).add_argument('--output')
    command = add('export', export, 'Export captured events or a pinned replay suite')
    command.add_argument('--suite-digest', help='Export an immutable replay suite by digest')
    command.add_argument('--output', required=True)
    command = add('invoke', chat, 'Send one captured request through the active model recipe')
    command.add_argument('--request', required=True, help='Canonical invocation JSON')
    command.add_argument('--live', action='store_true', help='Authorize the connected model request')
    command.add_argument('--timeout', type=float, default=45)
    command.add_argument('--output')
