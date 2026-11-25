"""Prepare or explicitly submit one bounded mapper-training/evaluation job.

Default is plan-only: it reads prepared data and public prices but performs no
AWS mutation. --submit is the sole job-creation gate. There are no retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse

try:
    from .submit import ACCOUNT, BUCKET, IMAGE, MODELS, PROFILE, REGION, aws, price
except ImportError:
    from submit import ACCOUNT, BUCKET, IMAGE, MODELS, PROFILE, REGION, aws, price

ROOT = Path(__file__).resolve().parents[1]
MAX_RUNTIME = 7200
DEFAULTS = {
    'train_steps': 400, 'train_prefix_tokens': 1024, 'val_prefix_tokens': 1024,
    'rank': 16, 'max_eval_tokens': 8192, 'eval_repeats': 1,
    'freegen_max_new_tokens': 16, 'max_eval_examples': 1037,
    'train_max_seconds': 4500, 'latency_examples': 24, 'latency_repeats': 5,
    'freegen_documents': 61, 'seed': 19,
}
SOURCE_FILES = ('cloud/full_worker.py', 'cloud/full_bootstrap.sh',
                'cloud/full_submit.py', 'cloud/full_collect.py')
ENTRY = ('mkdir -p /opt/ml/kvverify && '
         'tar -xzf /opt/ml/input/data/code/bundle.tar.gz -C /opt/ml/kvverify && '
         'exec bash /opt/ml/kvverify/cloud/full_bootstrap.sh')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_relative(name: str) -> str:
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Expected a normalized relative POSIX filename')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('Unsafe or non-normalized bundle filename: '+name)
    if any(part.startswith('.') for part in path.parts):
        raise ValueError('Hidden files cannot be packaged')
    if path.suffix.lower() not in ('.json', '.jsonl', '.py', '.sh') and path.name not in ('LICENSE', 'TERMS', 'SOURCE_README.md'):
        raise ValueError('Unsupported file type: '+name)
    return name


def prepared_files(data_dir: Path) -> tuple[dict, dict[str, Path]]:
    root = data_dir.resolve(strict=True)
    manifest_file = root/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Prepared manifest schema_version=1 and a files map are required')
    required = {'train.jsonl', 'mapper_validation.jsonl', 'dev.jsonl'}
    if not required.issubset(manifest['files']):
        raise ValueError('Prepared mapper train/validation and development files are required')
    splits = manifest.get('splits', {})
    for split, official, record in (('train', 'train', 'calibration'),
                                    ('mapper_validation', 'train', 'calibration'), ('dev', 'dev', 'eval')):
        info = splits.get(split, {})
        if (info.get('file') != split+'.jsonl' or info.get('official_split') != official
                or info.get('record_split') != record):
            raise ValueError('Prepared splits do not preserve train-only mapper fitting and dev-only evaluation')
        if info.get('sha256') != manifest['files'][info['file']].get('sha256'):
            raise ValueError('Split and file checksum manifests disagree')
    files = {'inputs/manifest.json': manifest_file}
    for name, info in manifest['files'].items():
        valid_relative(name)
        if name not in required | {'LICENSE', 'TERMS', 'SOURCE_README.md'}:
            raise ValueError('Only declared training/development data and attribution may be bundled')
        path = root/name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError('Dataset members must be ordinary files directly inside data-dir')
        if not isinstance(info, dict) or digest(path) != info.get('sha256'):
            raise ValueError('Prepared file checksum mismatch: '+name)
        if info.get('bytes') is not None and path.stat().st_size != info['bytes']:
            raise ValueError('Prepared byte-count mismatch: '+name)
        files['inputs/'+name] = path
    return manifest, files


def resume_uri(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if (parsed.scheme != 's3' or parsed.netloc != BUCKET or parsed.query or parsed.fragment
            or not parsed.path.startswith('/kv-transfer/') or '..' in PurePosixPath(parsed.path).parts):
        raise ValueError('Resume must reference this account\'s kv-transfer S3 prefix')
    return value.rstrip('/')+'/'


def configuration(stage: str, manifest_sha: str, runtime: int = MAX_RUNTIME,
                  resume: str | None = None) -> dict:
    if stage not in ('train-eval', 'train', 'eval'):
        raise ValueError('Unknown stage')
    if type(runtime) is not int or not 600 <= runtime <= MAX_RUNTIME:
        raise ValueError('Runtime must be 600-7200 seconds')
    resume = resume_uri(resume)
    if stage == 'eval' and resume is None:
        raise ValueError('Evaluation-only requires an explicit existing mapper/checkpoint prefix')
    return {'schema_version': 1, 'models': MODELS, 'stage': stage, **DEFAULTS,
            'data_manifest': 'inputs/manifest.json', 'data_manifest_sha256': manifest_sha,
            'max_runtime_seconds': runtime, 'worker_timeout_seconds': runtime-300,
            'checkpoint_dir': '/opt/ml/checkpoints',
            'resume_dir': '/opt/ml/input/data/resume' if resume else None,
            'resume_s3': resume, 'output_dir': '/opt/ml/model', 'dtype': 'float16',
            'evaluation_scope': 'All eligible official dev records; official test remains untouched',
            'compute_policy': 'Explicit --submit only; one bounded GPU job, no endpoint or automatic retries.'}


def build_request(job: str, config: dict) -> dict:
    if not re.fullmatch(r'sova-kv-full-[A-Za-z0-9-]{1,48}', job) or len(job) > 63:
        raise ValueError('Invalid job name')
    prefix = 'kv-transfer/'+job
    channels = [{'ChannelName': 'code', 'DataSource': {'S3DataSource': {
        'S3Uri': f's3://{BUCKET}/{prefix}/input/', 'S3DataType': 'S3Prefix'}}}]
    if config['resume_s3']:
        channels.append({'ChannelName': 'resume', 'DataSource': {'S3DataSource': {
            'S3Uri': config['resume_s3'], 'S3DataType': 'S3Prefix'}}})
    return {
        'TrainingJobName': job, 'RoleArn': f'arn:aws:iam::{ACCOUNT}:role/sova-sagemaker-exec',
        'AlgorithmSpecification': {'TrainingImage': IMAGE, 'TrainingInputMode': 'File',
                                   'ContainerEntrypoint': ['bash', '-c', ENTRY]},
        'ResourceConfig': {'InstanceType': 'ml.g5.2xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 50},
        'InputDataConfig': channels,
        'OutputDataConfig': {'S3OutputPath': f's3://{BUCKET}/{prefix}/output/'},
        'CheckpointConfig': {'S3Uri': f's3://{BUCKET}/{prefix}/checkpoints/',
                             'LocalPath': '/opt/ml/checkpoints'},
        'StoppingCondition': {'MaxRuntimeInSeconds': config['max_runtime_seconds']},
        'Environment': {'KV_BUCKET': BUCKET, 'KV_PREFIX': prefix, 'KV_STAGE': config['stage'],
                        'KV_CONFIG': '/opt/ml/kvverify/cloud/full-config.json',
                        'KV_WORKER_TIMEOUT': str(config['worker_timeout_seconds'])},
        'EnableManagedSpotTraining': False,
        'Tags': [{'Key': 'Project', 'Value': 'sova-kv-transfer'},
                 {'Key': 'Scope', 'Value': 'mapper-training-development-evaluation'}],
    }


def make_bundle(destination: Path, files: dict[str, Path], config: dict) -> dict[str, str]:
    payloads = {}
    for name, path in files.items():
        valid_relative(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Bundle contains a symlink or missing file: '+name)
        payloads[name] = path.read_bytes()
    payloads['cloud/full-config.json'] = (json.dumps(config, sort_keys=True, indent=2)+'\n').encode()
    # Stable archive bytes: names, content, timestamps, modes and gzip header.
    with destination.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, value in sorted(payloads.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mtime, info.mode = len(value), 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    archive.addfile(info, io.BytesIO(value))
    return {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payloads.items())}


def tested_source_hashes(root: Path = ROOT) -> dict[str, str]:
    files = [*sorted((root/'kvverify').glob('*.py')),
             *sorted((root/'cloud').glob('*.py')), *sorted((root/'cloud').glob('*.sh')),
             *sorted((root/'tests').glob('test*.py'))]
    return {path.relative_to(root).as_posix(): digest(path) for path in files}


def smoke_gate():
    evidence = ROOT/'reports/smoke-full-v1/test-results.json'
    report = json.loads(evidence.read_text())
    if (report.get('status') != 'passed' or report.get('tests_run', 0) < 50
            or any(report.get(key, -1) != 0 for key in ('failures', 'errors', 'skipped'))):
        raise ValueError('The complete current full suite must pass >=50 tests with no skips/errors/failures')
    actual = tested_source_hashes(ROOT)
    if not actual or report.get('file_sha256') != actual:
        raise ValueError('Full smoke evidence is stale or incomplete for current source/test checksums')
    return {'path': evidence.relative_to(ROOT).as_posix(), 'sha256': digest(evidence),
            'tests_run': report['tests_run'], 'tested_file_sha256': actual,
            'scope': 'Current full mapper, data, evaluation, adapter and cloud suite with frozen source/test hashes.'}


def compute_budget(rate: float, seconds: int, prior: float, budget: float) -> tuple[float, float]:
    if (not all(math.isfinite(value) for value in (rate, prior, budget))
            or rate <= 0 or not 0 <= prior <= 3.25 or not 0 < budget <= 3.25):
        raise ValueError('Positive finite rate/budget and bounded nonnegative finite prior spend required')
    estimate = rate*seconds/3600
    total = prior+estimate
    if not math.isfinite(estimate) or estimate <= 0 or total > budget:
        raise RuntimeError('Prior compute plus the current runtime estimate exceeds the compute guard')
    return estimate, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'datasets/prepared-v1')
    parser.add_argument('--stage', choices=('train-eval', 'train', 'eval'), default='train-eval')
    parser.add_argument('--resume-s3')
    parser.add_argument('--max-runtime-seconds', type=int, default=MAX_RUNTIME)
    parser.add_argument('--max-compute-usd', type=float, default=3.25,
                        help='Runtime-priced worker compute guard; storage/billing overhead/tax are extra')
    parser.add_argument('--prior-compute-usd', type=float, default=0.,
                        help='Known earlier compute in this experiment; included in the same guard')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.max_compute_usd) or not 0 < args.max_compute_usd <= 3.25:
        raise ValueError('This run supports a positive compute guard no greater than $3.25')
    manifest, files = prepared_files(args.data_dir)
    config = configuration(args.stage, digest(args.data_dir/'manifest.json'),
                           args.max_runtime_seconds, args.resume_s3)
    for name in SOURCE_FILES:
        files[name] = ROOT/name
    for path in sorted((ROOT/'kvverify').glob('*.py')):
        files[path.relative_to(ROOT).as_posix()] = path
    smoke = smoke_gate()
    rate = price()
    estimate, total_estimate = compute_budget(rate['usd_per_hour'], args.max_runtime_seconds,
                                              args.prior_compute_usd, args.max_compute_usd)
    job = 'sova-kv-full-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    prefix = 'kv-transfer/'+job
    request = build_request(job, config)
    with tempfile.TemporaryDirectory(prefix='sova-kv-full-') as temporary:
        temp = Path(temporary)
        bundle = temp/'bundle.tar.gz'
        hashes = make_bundle(bundle, files, config)
        receipt = {
            'status': 'prepared', 'job_name': job, 'profile': PROFILE, 'region': REGION,
            'request': request, 'config': config, 'price': rate,
            'estimated_max_worker_compute_usd': estimate, 'runtime_compute_guard_usd': args.max_compute_usd,
            'prior_compute_usd': args.prior_compute_usd,
            'total_estimated_compute_at_runtime_limit_usd': total_estimate,
            'cost_note': 'Runtime-bound public-price estimate, not invoice or total cost ceiling; storage, transfer, billing overhead and tax are extra.',
            'data_manifest_sha256': config['data_manifest_sha256'], 'data_splits': manifest.get('splits'),
            'bundle_sha256': digest(bundle), 'file_sha256': hashes, 'smoke_evidence': smoke,
            'live_results': f's3://{BUCKET}/{prefix}/live/',
            'checkpoints': f's3://{BUCKET}/{prefix}/checkpoints/',
            'cloud_mutation': 'None in plan mode; --submit uploads one bundle and creates exactly one training job.'}
        if args.submit:
            if json.loads(aws('sts', 'get-caller-identity'))['Account'] != ACCOUNT:
                raise RuntimeError('Unexpected AWS account')
            report = ROOT/'reports'/job
            report.mkdir(parents=True, exist_ok=False)
            saved = report/'submission.json'
            saved.write_text(json.dumps(receipt, indent=2)+'\n')
            request_file = temp/'request.json'
            request_file.write_text(json.dumps(request))
            try:
                aws('s3', 'cp', str(bundle), f's3://{BUCKET}/{prefix}/input/bundle.tar.gz', '--only-show-errors')
                result = json.loads(aws('sagemaker', 'create-training-job', '--cli-input-json', 'file://'+str(request_file)))
                receipt.update(status='submitted', job_arn=result['TrainingJobArn'])
            except subprocess.CalledProcessError as exc:
                receipt.update(status='submission_failed', error=exc.stderr.strip())
                saved.write_text(json.dumps(receipt, indent=2)+'\n')
                raise
            saved.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
