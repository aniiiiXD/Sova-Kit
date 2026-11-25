"""Package one 30-minute SageMaker pilot. No launch without --submit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROFILE, REGION, ACCOUNT = 'passkarwado', 'us-east-2', '169970912263'
BUCKET = 'sova-finance1-artifacts-use2-169970912263'
IMAGE = '763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training@sha256:263daec9f68067640794b4e82c24752c96f869dc5aa67eb6878c03893325eb54'
MODELS = [
    {'repo': 'Qwen/Qwen3-0.6B', 'revision': 'c1899de289a04d12100db370d81485cdf75e47ca'},
    {'repo': 'Qwen/Qwen3-1.7B', 'revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'},
]
PRICE_URL = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-2/index.json'
SAGEMAKER_CLI_MODEL = ROOT / 'artifacts' / 'aws-cli-model'


def aws(*args, region=REGION):
    environment = None
    if args[:2] in (('sagemaker', 'create-training-job'), ('sagemaker', 'describe-training-job')) and SAGEMAKER_CLI_MODEL.is_dir():
        environment = os.environ.copy()
        environment.setdefault('AWS_DATA_PATH', str(SAGEMAKER_CLI_MODEL))
    return subprocess.run(['aws', '--profile', PROFILE, '--region', region or REGION, *args],
                          check=True, capture_output=True, text=True, env=environment).stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_prices(instance_types, *, region=REGION):
    price_url = PRICE_URL.replace('/'+REGION+'/', '/'+region+'/')
    with urllib.request.urlopen(price_url, timeout=30) as response:
        data = json.load(response)
    requested = {name + '-Training': name for name in instance_types}
    prices = {}
    for sku, product in data['products'].items():
        a = product['attributes']
        instance = requested.get(a.get('instanceType'))
        if instance is not None and a.get('component') == 'Training':
            for term in data['terms']['OnDemand'][sku].values():
                for dimension in term['priceDimensions'].values():
                    if dimension['unit'] == 'Hrs':
                        if instance in prices:
                            raise RuntimeError('Multiple public training prices for ' + instance)
                        prices[instance] = {'usd_per_hour': float(dimension['pricePerUnit']['USD']),
                                            'sku': sku, 'source': price_url, 'effective_date': term['effectiveDate']}
    if set(prices) != set(instance_types):
        raise RuntimeError('Missing public training price')
    return prices


def price():
    return training_prices(('ml.g5.2xlarge',))['ml.g5.2xlarge']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    smoke = json.loads((ROOT/'reports/smoke-v1/test-results.json').read_text())
    if smoke['status'] != 'passed' or smoke['tests_run'] != 33 or smoke['skipped']:
        raise RuntimeError('All 33 scoped smoke tests must pass')
    rate = price()
    if rate['usd_per_hour'] > 1.60:
        raise RuntimeError('Price exceeds this pilot scope')
    if json.loads(aws('sts', 'get-caller-identity'))['Account'] != ACCOUNT:
        raise RuntimeError('Unexpected AWS account')
    job = 'sova-kv-pilot-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    prefix = 'kv-transfer/'+job
    request = {
        'TrainingJobName': job, 'RoleArn': f'arn:aws:iam::{ACCOUNT}:role/sova-sagemaker-exec',
        'AlgorithmSpecification': {'TrainingImage': IMAGE, 'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['bash', '-c', 'mkdir -p /opt/ml/kvverify && tar -xzf /opt/ml/input/data/code/bundle.tar.gz -C /opt/ml/kvverify && exec bash /opt/ml/kvverify/cloud/bootstrap.sh']},
        'ResourceConfig': {'InstanceType': 'ml.g5.2xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 50},
        'InputDataConfig': [{'ChannelName': 'code', 'DataSource': {'S3DataSource': {
            'S3Uri': f's3://{BUCKET}/{prefix}/input/', 'S3DataType': 'S3Prefix'}}}],
        'OutputDataConfig': {'S3OutputPath': f's3://{BUCKET}/{prefix}/output/'},
        'StoppingCondition': {'MaxRuntimeInSeconds': 1800}, 'EnableManagedSpotTraining': False,
        'Environment': {'KV_BUCKET': BUCKET, 'KV_PREFIX': prefix},
        'Tags': [{'Key': 'Project', 'Value': 'sova-kv-transfer'}, {'Key': 'Scope', 'Value': 'mechanism-pilot'}],
    }
    source = ROOT.parent/'sova-sys1/datasets/processed/contractnli'
    files = {p.relative_to(ROOT).as_posix(): p for folder in ('kvverify', 'cloud')
             for p in sorted((ROOT/folder).glob('*.py'))}
    files['cloud/bootstrap.sh'] = ROOT/'cloud/bootstrap.sh'
    for src, dst in [('documents/train_clean.jsonl','train-documents.jsonl'),
                     ('documents/dev.jsonl','dev-documents.jsonl'), ('decisions/dev.jsonl','dev-decisions.jsonl')]:
        files['inputs/'+dst] = source/src
    receipt = {'status': 'prepared', 'job_name': job, 'region': REGION, 'profile': PROFILE,
        'request': request, 'price': rate, 'max_worker_seconds': 1800,
        'estimated_max_worker_compute_usd': rate['usd_per_hour']/2,
        'cost_note': 'Runtime-bound estimate; storage, transfer, provisioning/billing overhead and tax are extra.',
        'models': MODELS, 'live_results': f's3://{BUCKET}/{prefix}/live/',
        'file_sha256': {name: sha(path) for name, path in files.items()},
        'scope': '12 calibration documents, 12 distinct held-out documents/questions, 5 arms x 2 repetitions; fp16',
        'approval': 'User explicitly requested executing smoke tests and a bounded SageMaker pilot in this conversation.'}
    with tempfile.TemporaryDirectory(prefix='sova-kv-package-') as temporary:
        temp = Path(temporary)
        config = temp/'config.json'
        config.write_text(json.dumps({'models': MODELS}, indent=2)+'\n')
        receipt['file_sha256']['cloud/config.json'] = sha(config)
        bundle = temp/'bundle.tar.gz'
        with tarfile.open(bundle, 'w:gz') as archive:
            for name, path in files.items():
                archive.add(path, arcname=name, recursive=False)
            archive.add(config, arcname='cloud/config.json', recursive=False)
        receipt['bundle_sha256'] = sha(bundle)
        if args.submit:
            report = ROOT/'reports'/job
            report.mkdir(parents=True, exist_ok=False)
            saved = report/'submission.json'
            saved.write_text(json.dumps(receipt, indent=2)+'\n')
            aws('s3', 'cp', str(bundle), f's3://{BUCKET}/{prefix}/input/bundle.tar.gz', '--only-show-errors')
            path = temp/'request.json'
            path.write_text(json.dumps(request))
            try:
                result = json.loads(aws('sagemaker', 'create-training-job', '--cli-input-json', 'file://'+str(path)))
                receipt.update(status='submitted', job_arn=result['TrainingJobArn'])
            except subprocess.CalledProcessError as exc:
                receipt.update(status='submission_failed', error=exc.stderr.strip())
                saved.write_text(json.dumps(receipt, indent=2)+'\n')
                raise
            saved.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
