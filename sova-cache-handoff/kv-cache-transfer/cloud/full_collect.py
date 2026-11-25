"""Collect an existing full-job receipt, artifacts and logs. Never start compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

try:
    from .full_submit import ACCOUNT, BUCKET, PROFILE, REGION, aws, resume_uri
except ImportError:
    from full_submit import ACCOUNT, BUCKET, PROFILE, REGION, aws, resume_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--logs', action='store_true')
    parser.add_argument('--status-only', action='store_true', help='Skip artifact downloads; return live job metadata quickly')
    parser.add_argument('--progress', action='store_true', help='Also read the small latest progress.json object from S3')
    parser.add_argument('--checkpoints', action='store_true', help='Also download existing checkpoints')
    parser.add_argument('--model-artifact', action='store_true', help='Download output model.tar.gz without extracting')
    args = parser.parse_args()
    receipt = args.receipt.resolve(strict=True)
    spec = json.loads(receipt.read_text())
    if spec['region'] != REGION or spec['profile'] != PROFILE:
        raise ValueError('Unexpected receipt account/region/profile')
    expected_role = f'arn:aws:iam::{ACCOUNT}:role/sova-sagemaker-exec'
    if spec['request']['RoleArn'] != expected_role:
        raise ValueError('Unexpected receipt role')
    job = spec['job_name']
    live = resume_uri(spec['live_results'])
    checkpoints = resume_uri(spec['checkpoints'])
    description = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', job))
    billable = description.get('BillableTimeInSeconds')
    status = {'job_name': job, 'status': description['TrainingJobStatus'],
        'secondary_status': description.get('SecondaryStatus'),
        'failure_reason': description.get('FailureReason'),
        'training_seconds': description.get('TrainingTimeInSeconds'), 'billable_seconds': billable,
        'estimated_billed_compute_usd': None if billable is None else billable*spec['price']['usd_per_hour']/3600,
        'cost_note': 'Public-rate estimate from AWS billable seconds, not an invoice; storage, transfer and tax are extra.',
        'output': description.get('ModelArtifacts', {}),
        'transitions': description.get('SecondaryStatusTransitions', [])}
    # Persist status before optional transfers so even a log/API failure is visible.
    (receipt.parent/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    if args.progress:
        try:
            progress = json.loads(aws('s3', 'cp', live+'progress.json', '-', '--only-show-errors'))
            status['live_progress'] = progress
            (receipt.parent/'live-progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            status['live_progress_available'] = False
            status['live_progress_note'] = 'No readable progress artifact yet; job metadata is authoritative.'
        (receipt.parent/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    if args.status_only:
        if args.logs or args.checkpoints or args.model_artifact:
            parser.error('--status-only cannot be combined with downloads or --logs')
        print(json.dumps(status, indent=2))
        return
    aws('s3', 'sync', live, str(receipt.parent/'collected'), '--only-show-errors')
    if args.checkpoints:
        aws('s3', 'sync', checkpoints, str(receipt.parent/'checkpoints'), '--only-show-errors')
    if args.model_artifact:
        artifact = description.get('ModelArtifacts', {}).get('S3ModelArtifacts')
        if artifact:
            resume_uri(artifact)  # Same project/bucket, but retain the exact object URI.
            aws('s3', 'cp', artifact, str(receipt.parent/'model.tar.gz'), '--only-show-errors')
    if args.logs:
        try:
            streams = json.loads(aws('logs', 'describe-log-streams', '--log-group-name', '/aws/sagemaker/TrainingJobs',
                '--log-stream-name-prefix', job))
            events = []
            for stream in streams.get('logStreams', []):
                page = json.loads(aws('logs', 'get-log-events', '--log-group-name', '/aws/sagemaker/TrainingJobs',
                    '--log-stream-name', stream['logStreamName'], '--limit', '10000', '--no-start-from-head'))
                events.extend(page.get('events', []))
            events.sort(key=lambda item: item['timestamp'])
            (receipt.parent/'cloudwatch.json').write_text(json.dumps(events, indent=2)+'\n')
            for event in events[-40:]:
                print(event['message'])
        except subprocess.CalledProcessError as exc:
            status['log_collection_error'] = exc.stderr.strip()
            (receipt.parent/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
