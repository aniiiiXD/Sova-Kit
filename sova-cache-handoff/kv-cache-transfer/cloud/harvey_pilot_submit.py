"""Package and launch a bounded Qwen3 cache transfer evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import tempfile

from .autonomous_budget import TOTAL_LIMIT_USD, reconcile
from .full_submit import make_bundle
from .package_integrity import verify_worker_imports
from .recovery_submit import request
from .submit import ACCOUNT, BUCKET, MODELS, PROFILE, REGION, aws

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = 'ml.g6e.2xlarge'
PAIR = [MODELS[1], {'repo': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218'}]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(benchmark='harvey'):
    reference = ROOT / 'datasets/reference-benchmarks-v1'
    reference_manifest = json.loads((reference / 'manifest.json').read_text())
    calibration = 'processed/quality/train.jsonl'
    data = {'inputs/quality-train.jsonl': reference / calibration,
            'inputs/calibration-manifest.json': reference / 'manifest.json'}
    if benchmark == 'quality':
        dataset = ROOT / 'datasets/loop-holdout-v1'
        manifest = json.loads((dataset / 'manifest.json').read_text())
        data.update({'inputs/evaluation.jsonl': dataset / 'quality.jsonl',
                     'inputs/evaluation-manifest.json': dataset / 'manifest.json'})
        evaluation_sha = manifest['files']['quality.jsonl']['sha256']
        documents, questions, calibration_count, calibration_limit = 15, 267, 32, 8192
    elif benchmark == 'harvey':
        dataset = ROOT / 'datasets/harvey-contract-facts-v1'
        manifest = json.loads((dataset / 'manifest.json').read_text())
        data.update({'inputs/evaluation.jsonl': dataset / 'pilot.jsonl',
                     'inputs/evaluation-manifest.json': dataset / 'manifest.json',
                     'inputs/LICENSE': dataset / 'LICENSE.harvey.txt'})
        evaluation_sha = manifest['pilot_sha256']
        documents, questions, calibration_count, calibration_limit = 4, 12, 12, 4096
    else:
        raise ValueError('Unknown evaluation dataset')
    if (sha(data['inputs/evaluation.jsonl']) != evaluation_sha or
            sha(data['inputs/quality-train.jsonl']) != reference_manifest['processed'][calibration]['sha256']):
        raise ValueError('Dataset bytes differ from the frozen manifest')
    files = {path.relative_to(ROOT).as_posix(): path
             for folder in ('kvverify', 'cloud') for path in (ROOT / folder).glob('*.py')}
    files['cloud/recovery_bootstrap.sh'] = ROOT / 'cloud/recovery_bootstrap.sh'
    files.update(data)
    config = {'mode': 'harvey_pilot', 'benchmark': benchmark, 'models': PAIR, 'seed': 19,
              'expected_documents': documents, 'expected_questions': questions,
              'max_runtime_seconds': 2700, 'worker_timeout_seconds': 2280,
              'calibration_documents': calibration_count, 'calibration_prefix_tokens': calibration_limit, 'ridge_alpha': .01,
              'max_prompt_tokens': 30720, 'mapper_origin': 'unused',
              'input_sha256': {name: sha(path) for name, path in data.items()},
              'verified_worker_imports': verify_worker_imports(files, 'cloud/harvey_pilot_worker.py')}
    return config, files


def dispatch(config, files, *, instance, output, submit, job_kind, mapper_input=False, job_name=None,
             preferences=(), allow_active_workers=0, authorized_total_usd=TOTAL_LIMIT_USD):
    ledger = (reconcile(config['max_runtime_seconds'], instance_type=instance,
                        allow_active_workers=allow_active_workers,
                        authorized_total_usd=authorized_total_usd) if submit else None)
    job = job_name or f'sova-kv-{job_kind}-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
    req = request(job, config)
    if not mapper_input:
        req['InputDataConfig'] = req['InputDataConfig'][:1]
    if config.get('corrected_mapper_origin'):
        if not mapper_input:
            raise ValueError('Corrected replay requires both comparison checkpoints')
        req['InputDataConfig'].append({
            'ChannelName': 'corrected',
            'DataSource': {'S3DataSource': {'S3Uri': config['corrected_mapper_origin'], 'S3DataType': 'S3Prefix'}},
        })
    req.pop('CheckpointConfig')
    if preferences:
        req['ResourceConfig'].pop('InstanceType')
        req['ResourceConfig']['InstancePreferences'] = [{'InstanceType': name} for name in preferences]
        req['StoppingCondition']['MaxPendingTimeInSeconds'] = 7200
        req.pop('EnableManagedSpotTraining')
    else:
        req['ResourceConfig']['InstanceType'] = instance
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='sova-cache-eval-') as work:
        work = Path(work)
        frozen = work / 'config.json'
        frozen.write_text(json.dumps(config))
        files['cloud/recovery-config.json'] = frozen
        bundle = work / 'bundle.tar.gz'
        hashes = make_bundle(bundle, files, config)
        if any(hashes[name] != expected for name, expected in config['input_sha256'].items()):
            raise ValueError('Packaged input drift')
        destination = output / 'submission.json'
        receipt = {'status': 'prepared', 'job_name': job, 'mode': config['mode'], 'region': REGION,
                   'profile': PROFILE, 'config': config, 'request': req, 'budget_ledger': ledger,
                   'file_sha256': hashes, 'bundle_sha256': sha(bundle),
                   'live_results': f's3://{BUCKET}/kv-transfer/{job}/live/'}
        if ledger:
            receipt['estimated_max_worker_compute_usd'] = ledger['proposed_max_runtime_compute_usd']
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        if submit:
            if json.loads(aws('sts', 'get-caller-identity'))['Account'] != ACCOUNT:
                raise ValueError('Wrong AWS account')
            spec = work / 'request.json'
            spec.write_text(json.dumps(req))
            try:
                aws('s3', 'cp', str(bundle), f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz',
                    '--only-show-errors')
                response = json.loads(aws('sagemaker', 'create-training-job', '--cli-input-json', 'file://' + str(spec)))
                receipt.update(status='submitted', job_arn=response['TrainingJobArn'])
            except Exception as exc:
                receipt.update(status='submission_failed', error=getattr(exc, 'stderr', None) or str(exc))
                raise
            finally:
                destination.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({key: receipt.get(key) for key in
                          ('status', 'job_name', 'estimated_max_worker_compute_usd', 'live_results')}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', choices=('harvey', 'quality'), default='harvey')
    parser.add_argument('--instance', choices=(INSTANCE, 'ml.g6e.4xlarge'), default=INSTANCE)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    config, files = prepare(args.benchmark)
    dispatch(config, files, instance=args.instance, output=args.output, submit=args.submit,
             job_kind=args.benchmark)


if __name__ == '__main__':
    main()
