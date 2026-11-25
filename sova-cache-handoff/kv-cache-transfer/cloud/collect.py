"""Fetch status and existing artifacts; never launch or restart compute."""
import argparse
import json
from pathlib import Path
from submit import aws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--logs', action='store_true')
    args = parser.parse_args()
    receipt = Path(args.receipt).resolve()
    spec = json.loads(receipt.read_text())
    job = spec['job_name']
    description = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', job))
    seconds = description.get('BillableTimeInSeconds')
    status = {'job_name': job, 'status': description['TrainingJobStatus'],
        'secondary_status': description.get('SecondaryStatus'),
        'failure_reason': description.get('FailureReason'),
        'training_seconds': description.get('TrainingTimeInSeconds'),
        'billable_seconds': seconds,
        'estimated_billed_compute_usd': None if seconds is None else seconds*spec['price']['usd_per_hour']/3600,
        'cost_note': 'Estimate from reported billable seconds and public rate; not an AWS invoice; excludes storage, transfer and tax.',
        'output': description.get('ModelArtifacts', {}),
        'transitions': description.get('SecondaryStatusTransitions', [])}
    (receipt.parent/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    aws('s3', 'sync', spec['live_results'], str(receipt.parent/'collected'), '--only-show-errors')
    if args.logs:
        streams = json.loads(aws('logs', 'describe-log-streams', '--log-group-name', '/aws/sagemaker/TrainingJobs',
            '--log-stream-name-prefix', job, '--output', 'json'))
        events = []
        for stream in streams.get('logStreams', []):
            page = json.loads(aws('logs', 'get-log-events', '--log-group-name', '/aws/sagemaker/TrainingJobs',
                '--log-stream-name', stream['logStreamName'], '--limit', '1000', '--no-start-from-head'))
            events.extend(page.get('events', []))
        events.sort(key=lambda x: x['timestamp'])
        (receipt.parent/'cloudwatch.json').write_text(json.dumps(events, indent=2)+'\n')
        for event in events[-35:]:
            print(event['message'])
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
