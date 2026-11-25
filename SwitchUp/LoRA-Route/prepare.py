"""Build a local SageMaker launch bundle for the SwitchUP weights pilot."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import hashlib
import json
import re
import shlex
import shutil
import sys
import zipfile


SWITCHUP_ROOT = Path(__file__).resolve().parents[1]
SWITCHIT_ROOT = SWITCHUP_ROOT.parent / 'SwitchIt'
sys.path.insert(0, str(SWITCHIT_ROOT))

from switchit.composer import validate_suite
from switchit.data_paths import kit_data_dir
from switchit.workflow_builder import workflow_revision


MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
IMAGE = ('763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm@sha256:'
         '23c809c9f619014b1be84e1c35f0c8e7f4e47756cab96261b3b2087fe955a107')
REGION = 'us-east-1'
RATE = 1.408
RUNTIME_SECONDS = 2700
PENDING_SECONDS = 7200
SYSTEM = ('You route order support requests. Respond with exactly one action: ASK to request confirmation, '
          'CANCEL to cancel the order, or KEEP to leave the order in place. Use the order details and the '
          'routing policy you know. Return the action only.')

TRAIN_TEMPLATES = (
    'Order {order}. Customer wants to {intent}. Lane is {lane}. Authorization is {auth}. Which action?',
    'Route this ticket: order={order}, intent={intent}, lane={lane}, authorization={auth}.',
    'Support queue entry for {order}: {auth} authorization and {lane} fulfillment. The request is {intent}.',
    'For the {intent} request on {order}, the warehouse lane is {lane}; the authorization status is {auth}.',
)
HOLDOUT_TEMPLATES = (
    'Warehouse report for order {order} lists {lane}. A customer asks to {intent}; their authorization reads {auth}. Decide.',
    'Resolve {order}. We have a {intent} request, a fulfillment code of {lane}, and authorization marked {auth}.',
    'Which action should support take on this order? {order}, {lane} warehouse lane, {auth} authorization, request to {intent}.',
)


def _rows(templates, prefix, repeats):
    scenarios = list(product(('track', 'cancel'), ('MESA', 'TIDE'), ('GRANTED', 'PENDING')))
    rows = []
    for template in templates:
        for _ in range(repeats):
            for intent, lane, auth in scenarios:
                order = f'{prefix}{len(rows):04d}'
                if intent == 'track':
                    action = 'KEEP'
                else:
                    action = 'CANCEL' if lane == 'MESA' and auth == 'GRANTED' else 'ASK'
                rows.append({'order_id': order, 'text': template.format(order=order, intent=intent, lane=lane, auth=auth),
                             'corrected_action': action, 'trace_id': f'synthetic-{order}'})
    return rows


def documents():
    train = _rows(TRAIN_TEMPLATES, 'tr', 6)
    holdout = _rows(HOLDOUT_TEMPLATES, 'ho', 1)
    checks = [{'type': 'output_regex', 'pattern': '^(ASK|CANCEL|KEEP)$'}]
    workflow = {
        'workflow_version': 1, 'workflow_id': 'switchup-weights-pilot', 'name': 'Order routing pilot',
        'status': 'reviewed', 'reviewed_by': 'Synthetic pilot specification', 'inputs': {}, 'outputs': [],
        'allowed_tools': ['cancel_order'], 'success_checks': checks,
        'steps': [
            {'id': 'decide', 'kind': 'model', 'depends_on': [], 'request': {'$ref': 'inputs.request'}},
            {'id': 'cancel', 'kind': 'tool', 'tool': 'cancel_order', 'depends_on': ['decide'],
             'requires_proposal': True, 'arguments': {'id': {'$ref': 'inputs.order_id'}},
             'when': {'path': 'steps.decide.tool_calls.0.name', 'equals': 'cancel_order', 'on_missing': 'skip'}},
        ],
        'tool_simulation': {'cancel_order': {'operation': 'set',
                            'path': ['orders', {'$ref': 'arguments.id'}, 'status'], 'value': 'cancelled'}},
    }
    workflow['revision'] = workflow_revision(workflow)
    suite = {'workflow_cases_version': 1, 'workflow_revision': workflow['revision'], 'cases': []}
    for split, rows in (('train', train[:8]), ('holdout', holdout)):
        for row in rows:
            order, action = row['order_id'], row['corrected_action']
            suite['cases'].append({
                'case_id': order, 'trace_id': row['trace_id'], 'split_group': row['trace_id'], 'split': split,
                'workflow_id': workflow['workflow_id'], 'workflow_revision': workflow['revision'], 'checks': checks,
                'inputs': {'order_id': order, 'request': {'messages': [{'role': 'user', 'content': row['text']}]}},
                'initial_state': {'orders': {order: {'status': 'open'}}},
                'expected_checks': [{'type': 'output_equals', 'text': action},
                                    {'type': 'state_equals', 'path': f'orders.{order}.status',
                                     'value': 'cancelled' if action == 'CANCEL' else 'open'}],
            })
    validate_suite(workflow, suite)
    return train, workflow, suite


def _write(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _aws_command(profile, *args):
    return shlex.join(['aws', '--profile', profile, '--region', REGION, *args])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--role-arn', required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,61}[a-z0-9]', args.job_name):
        parser.error('job name must be lowercase alphanumeric with internal hyphens')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', args.bucket):
        parser.error('bucket must be an S3 bucket name')
    if not re.fullmatch(r'[A-Za-z0-9_/-]+', args.prefix.strip('/')):
        parser.error('prefix must be a nonempty S3 path')
    train, workflow, suite = documents()
    local = kit_data_dir(SWITCHUP_ROOT.parent) / 'runs' / 'sagemaker' / args.job_name
    payload = local / 'payload'
    payload.mkdir(parents=True, exist_ok=False)
    root = f's3://{args.bucket}/{args.prefix.strip("/")}/{args.job_name}'
    shutil.copyfile(SWITCHUP_ROOT / 'Retraining' / 'worker.py', payload / 'worker.py')
    with zipfile.ZipFile(payload / 'switchit.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((SWITCHIT_ROOT / 'switchit').rglob('*.py')):
            archive.write(path, path.relative_to(SWITCHIT_ROOT).as_posix())
    _write(payload / 'training.json', train)
    _write(payload / 'workflow.json', workflow)
    _write(payload / 'suite.json', suite)
    _write(payload / 'manifest.json', {'model': MODEL, 'revision': REVISION, 'system_prompt': SYSTEM,
          'seed': 814, 'epochs': 4, 'batch_size': 4, 'rank': 8, 'alpha': 16, 'learning_rate': 0.0004,
          'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(payload.iterdir())}})
    job = {
        'TrainingJobName': args.job_name, 'RoleArn': args.role_arn,
        'AlgorithmSpecification': {'TrainingImage': IMAGE, 'TrainingInputMode': 'File',
                                   'ContainerEntrypoint': ['python3', '/opt/ml/input/data/payload/worker.py']},
        'InputDataConfig': [{'ChannelName': 'payload', 'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix', 'S3Uri': f'{root}/input/', 'S3DataDistributionType': 'FullyReplicated'}}}],
        'OutputDataConfig': {'S3OutputPath': f'{root}/output/'},
        'ResourceConfig': {'InstanceType': 'ml.g5.xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 40},
        'StoppingCondition': {'MaxRuntimeInSeconds': RUNTIME_SECONDS, 'MaxPendingTimeInSeconds': PENDING_SECONDS},
        'EnableNetworkIsolation': False,
        'Environment': {'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': '/tmp/switchup-hf'},
        'Tags': [{'Key': 'Project', 'Value': 'SwitchUP'}, {'Key': 'Scope', 'Value': 'synthetic-weights-pilot'}],
    }
    request = local / 'training-job.json'
    _write(request, job)
    stage = _aws_command(args.profile, 's3', 'cp', f'{payload}/', f'{root}/input/', '--recursive', '--only-show-errors')
    submit = _aws_command(args.profile, 'sagemaker', 'create-training-job', '--cli-input-json', f'file://{request}', '--no-cli-pager')
    launch = local / 'launch.sh'
    launch.write_text(f'#!/bin/bash\nset -euo pipefail\n{stage}\n{submit}\n', encoding='utf-8')
    launch.chmod(0o700)
    print(json.dumps({'model': MODEL, 'revision': REVISION, 'job': args.job_name, 'training_rows': len(train),
          'heldout_cases': sum(case['split'] == 'holdout' for case in suite['cases']),
          'runtime_minutes': RUNTIME_SECONDS // 60, 'compute_at_runtime_limit_usd': round(RATE * RUNTIME_SECONDS / 3600, 3),
          'launch': shlex.join(['bash', str(launch)]), 'submit': submit,
          'results': f'{root}/output/{args.job_name}/output/model.tar.gz'}, indent=2))


if __name__ == '__main__':
    main()
