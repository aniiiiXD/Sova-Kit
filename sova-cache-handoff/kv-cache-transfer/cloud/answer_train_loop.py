"""Run one SageMaker candidate through the durable development graph."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from kvverify.agent_graph import Controller, Node, RetryableError, digest
from .answer_train_worker import development_gate
from .autonomous_budget import RATE_LIMITS
from .harvey_pilot_submit import INSTANCE, PAIR, ROOT, dispatch, sha
from .package_integrity import verify_worker_imports
from .submit import BUCKET, aws


def prepare(max_runtime_seconds=4800):
    parent = json.loads((ROOT/'reports/quality-8b-v1/submission.json').read_text())
    if parent['config']['models'] != PAIR:
        raise ValueError('Archived checkpoint does not match the intended model pair')
    state = json.loads(aws('sagemaker','describe-training-job','--training-job-name',parent['job_name']))
    if state['TrainingJobStatus'] != 'Completed':
        raise ValueError('Expected completed base job')
    origin = f"s3://{BUCKET}/kv-transfer/{parent['job_name']}/live/ridge/"
    base = json.loads(aws('s3','cp',origin+'manifest.json','-','--only-show-errors'))
    benchmark = ROOT/'datasets/reference-benchmarks-v1'
    reused = ROOT/'datasets/loop-holdout-v1'
    benchmark_manifest = json.loads((benchmark/'manifest.json').read_text())
    previous_manifest = json.loads((reused/'manifest.json').read_text())
    train_name = 'processed/quality/train.jsonl'
    inputs = {'inputs/train.jsonl': benchmark/train_name,
              'inputs/development.jsonl': reused/'quality.jsonl'}
    expected = {'inputs/train.jsonl': benchmark_manifest['processed'][train_name]['sha256'],
                'inputs/development.jsonl': previous_manifest['files']['quality.jsonl']['sha256']}
    if {key:sha(path) for key,path in inputs.items()} != expected:
        raise ValueError('Frozen training or development file changed')
    files = {path.relative_to(ROOT).as_posix(): path for folder in ('kvverify','cloud')
             for path in (ROOT/folder).glob('*.py')}
    files['cloud/recovery_bootstrap.sh'] = ROOT/'cloud/recovery_bootstrap.sh'
    files.update(inputs)
    config = {'mode':'answer_train','models':PAIR,'seed':19,'rank':16,'steps':192,
              'validate_at':[32,96,192],'learning_rate':.0006,'native_prefix_tokens':256,
              'max_prompt_tokens':8192,'max_runtime_seconds':max_runtime_seconds,
              'worker_timeout_seconds':max_runtime_seconds - 360,
              'mapper_origin':origin,'input_mapper_dir':'/opt/ml/input/data/mapper',
              'initial_weights_sha256':base['weights_sha256'],'input_sha256':expected,
              'assessment':'Archived evaluation articles are development only. No confirmation data.',
              'verified_worker_imports':verify_worker_imports(files,'cloud/answer_train_worker.py')}
    return config, files


def optional_live(uri, *, region=None):
    try:
        args = ('s3','cp',uri,'-','--only-show-errors')
        return json.loads(aws(*args, region=region) if region else aws(*args))
    except subprocess.CalledProcessError as exc:
        if any(word in exc.stderr for word in ('404','NoSuchKey','Not Found')):
            return None
        raise RetryableError('S3 read error: '+exc.stderr.strip()) from exc


def describe(job, *, absent_ok=False, region=None):
    try:
        args = ('sagemaker','describe-training-job','--training-job-name',job)
        return json.loads(aws(*args, region=region) if region else aws(*args))
    except subprocess.CalledProcessError as exc:
        if absent_ok and any(word in exc.stderr.lower() for word in ('could not find','not found','does not exist')):
            return None
        raise RetryableError('SageMaker status error: '+exc.stderr.strip()) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance', choices=tuple(i for i in RATE_LIMITS if i.startswith('ml.g6e.')), default=INSTANCE)
    parser.add_argument('--max-runtime-seconds', type=int, default=4800)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--submit', action='store_true')
    actions.add_argument('--attach-existing', action='store_true',
                         help='Use the persisted run spec and exact AWS receipt without submitting')
    args = parser.parse_args()
    directory = args.output.resolve()
    receipt_path = directory/'training'/'submission.json'
    database = directory/'graph.sqlite3'
    if args.attach_existing:
        if not database.is_file() or not receipt_path.is_file():
            raise ValueError('Existing journal and submission receipt are required')
        with sqlite3.connect(database) as connection:
            persisted = connection.execute('SELECT value FROM run_spec WHERE id = 1').fetchone()
        spec = json.loads(persisted[0])['inputs']
        config = spec['config']
        files = {}
        job = json.loads(receipt_path.read_text())['job_name']
    else:
        if not 1800 <= args.max_runtime_seconds <= 4800:
            parser.error('--max-runtime-seconds must be between 1800 and 4800')
        if args.max_runtime_seconds * RATE_LIMITS[args.instance] > 6 * 3600:
            parser.error('The requested runtime can exceed $6 at the instance rate ceiling')
        config, files = prepare(args.max_runtime_seconds)
        spec = {'config':config,'file_sha256':{name:sha(path) for name,path in files.items()},
                'instance_type':args.instance}
        job = 'sova-kv-answer-train-'+digest(spec)[:20]
        if not args.submit:
            print(json.dumps({'status':'prepared', 'job_name':job, 'config':config}, indent=2))
            return

    def schedule(_):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['job_name'] != job or receipt['config'] != config:
                raise ValueError('A different run owns the training receipt')
            if describe(job, absent_ok=True) is None:
                # The original transaction may have stopped just before CreateTrainingJob.
                # Keep its receipt for an operator to reconcile rather than allocating twice.
                raise RuntimeError('Training intent has no AWS job. Reconcile the saved receipt before retrying.')
            return {'job':job,'s3':receipt['live_results']}
        if args.attach_existing:
            raise RuntimeError('Attached runs cannot submit a job')
        receipt = dispatch(config, dict(files), instance=args.instance, output=receipt_path.parent, submit=True,
                           job_kind='answer-train', mapper_input=True, job_name=job)
        return {'job':job,'s3':receipt['live_results']}

    def observe(inputs):
        launched = inputs['schedule']
        last = None
        while True:
            state = describe(launched['job'])
            status = state['TrainingJobStatus']
            progress = optional_live(launched['s3']+'progress.json') if status == 'InProgress' else None
            snapshot = {'status':status, 'phase':state['SecondaryStatus'], 'progress':progress}
            if snapshot != last:
                print(json.dumps(snapshot), flush=True)
                last = snapshot
            if status in ('Failed','Stopped'):
                return {'job':launched['job'],'state':status, 'diagnosis':state.get('FailureReason'),
                        's3':launched['s3'],'billable_seconds':state.get('BillableTimeInSeconds',0)}
            if status == 'Completed':
                return {'job':launched['job'],'state':status,'s3':launched['s3'],
                        'billable_seconds':state['BillableTimeInSeconds']}
            time.sleep(20)

    def verify(inputs):
        job_result = inputs['observe']
        result = {**job_result,'product_goal_achieved':False,'confirmation_opened':False}
        if result['state'] != 'Completed':
            return {**result,'verdict':'worker_failed'}
        summary = optional_live(result['s3']+'summary.json')
        if not summary or summary['locked_confirmation_accessed'] or summary['product_goal_achieved']:
            raise ValueError('Inconsistent development evidence')
        if summary['selected']['passes_development_gate'] != development_gate(summary['selected']):
            raise ValueError('Development gate cannot be reproduced')
        return {**result, 'training':summary,
                'verdict':'ready_for_timing' if development_gate(summary['selected']) else 'development_rejected'}

    graph = [Node('schedule',(),schedule,max_attempts=3),
             Node('observe',('schedule',),observe,max_attempts=3),
             Node('verify',('observe',),verify,max_attempts=3)]
    result = asyncio.run(Controller(database, graph, spec).run())['verify']
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
