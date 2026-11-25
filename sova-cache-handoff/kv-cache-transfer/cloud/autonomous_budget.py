"""Reconcile this project's managed jobs before each autonomous allocation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from .submit import REGION, aws, training_prices

DEFAULT_INSTANCE = "ml.g5.2xlarge"
RATE_LIMITS = {
    DEFAULT_INSTANCE: 1.60,
    "ml.g6e.xlarge": 2.85,
    "ml.g6e.2xlarge": 3.00,
    "ml.g6e.4xlarge": 4.00,
    "ml.g6e.8xlarge": 6.00,
    "ml.g6e.16xlarge": 9.80,
}

ROOT=Path(__file__).resolve().parents[1]
TOTAL_LIMIT_USD = 30.0
BILLING_RESERVE_USD = 2.0
PROJECT_REGIONS = (REGION, "ap-south-1")


def reconcile(proposed_seconds=0,*,allow_active_workers=0,instance_type=DEFAULT_INSTANCE,
              authorized_total_usd=TOTAL_LIMIT_USD):
    if type(allow_active_workers)is not int or not 0<=allow_active_workers<=1:raise ValueError('At most one explicitly reserved concurrent control is allowed')
    if instance_type not in RATE_LIMITS:
        raise ValueError('Unknown proposed instance type')
    if not math.isfinite(authorized_total_usd) or not 0 < authorized_total_usd:
        raise ValueError('Invalid authorized project cap')
    summaries=[(region, row) for region in PROJECT_REGIONS
               for row in json.loads(aws('sagemaker','list-training-jobs','--name-contains','sova-kv',region=region))['TrainingJobSummaries']]
    def inspect(entry):
        region,row=entry
        d=json.loads(aws('sagemaker','describe-training-job','--training-job-name',row['TrainingJobName'],region=region))
        resources = d['ResourceConfig']
        if resources.get('SelectedInstanceCount', resources.get('InstanceCount')) != 1:
            raise ValueError('Expected single-instance historical job')
        instance = resources.get('SelectedInstanceType') or resources.get('InstanceType')
        if not instance:
            preferences = [item['InstanceType'] for item in resources.get('InstancePreferences', [])]
            if not preferences or d.get('TrainingStartTime'):
                raise ValueError('Unknown instance allocation for project job')
            instance = max(preferences, key=RATE_LIMITS.__getitem__)
        if instance not in RATE_LIMITS:
            raise ValueError('Unknown instance price in project ledger')
        terminal=d['TrainingJobStatus']in('Completed','Failed','Stopped')
        seconds=d.get('BillableTimeInSeconds',0)if terminal else d['StoppingCondition']['MaxRuntimeInSeconds']
        return {'job_name':d['TrainingJobName'],'status':d['TrainingJobStatus'],
            'billable_seconds':d.get('BillableTimeInSeconds'),'reserved_or_actual_seconds':seconds,
            'instance_type':instance,'region':region,'terminal':terminal}
    with ThreadPoolExecutor(max_workers=5)as pool:jobs=list(pool.map(inspect,summaries))
    types_by_region = {region:{job['instance_type'] for job in jobs if job['region']==region} for region in PROJECT_REGIONS}
    types_by_region[REGION].add(instance_type)
    rates = {(region, name):price for region,names in types_by_region.items() if names
             for name,price in training_prices(names,region=region).items()}
    for (region,instance), pricing in rates.items():
        hourly = pricing['usd_per_hour']
        limit=3.50 if (region,instance)==('ap-south-1','ml.g6e.2xlarge') else RATE_LIMITS[instance]
        if not math.isfinite(hourly) or not 0 < hourly <= limit:
            raise ValueError('Unexpected worker price for ' + instance)
    for job in jobs:
        job['runtime_compute_usd'] = job['reserved_or_actual_seconds'] * rates[job['region'],job['instance_type']]['usd_per_hour'] / 3600
    rate = rates[REGION,instance_type]
    total=sum(j['runtime_compute_usd']for j in jobs)
    proposed=proposed_seconds*rate['usd_per_hour']/3600
    ledger={'updated_utc':datetime.now(timezone.utc).isoformat(),'user_authorized_total_usd':authorized_total_usd,
        'billing_extras_reserve_usd':BILLING_RESERVE_USD,'runtime_compute_usd':total,'proposed_max_runtime_compute_usd':proposed,
        'estimated_with_reserve_and_next_job_usd':total+BILLING_RESERVE_USD+proposed,'price':rate,'jobs':jobs,
        'policy':f'The ${authorized_total_usd:.2f} cap includes historical compute in both regions, the proposed job at its ceiling, and the billing reserve.'}
    path=ROOT/'research/autonomous-loop/ledger.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(ledger,indent=2)+'\n')
    if ledger['estimated_with_reserve_and_next_job_usd']>authorized_total_usd:
        raise RuntimeError(f'BUDGET_ALERT: next run plus reserved extras would exceed ${authorized_total_usd:.2f}')
    if sum(not j['terminal']for j in jobs)>allow_active_workers:raise RuntimeError('Too many active project workers for this scoped experiment')
    return ledger
