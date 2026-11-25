"""Autonomous frozen-receiver distillation, subject to reconciled$50 budget."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import tempfile
from .linear_search_submit import sources as prior_sources,TESTS as PRIOR_TESTS
from .state_diagnostic_submit import inputs
from .submit import aws,ACCOUNT,BUCKET,PROFILE,REGION,MODELS
from .recovery_submit import request
from .full_submit import make_bundle,digest
from .autonomous_budget import reconcile
from .package_integrity import verify_worker_imports
ROOT=Path(__file__).resolve().parents[1]
TESTS=(*PRIOR_TESTS,'test_continuation_training.py','test_task_distillation.py','test_native_prefix.py','test_layer_readout.py','test_key_geometry.py','test_worker_package.py')


def sources():
    files=prior_sources()
    # Include the complete small project module tree; no data, credentials,
    # virtual environments or external model weights enter this source bundle.
    for folder in('kvverify','cloud'):
        for path in(ROOT/folder).glob('*.py'):files[path.relative_to(ROOT).as_posix()]=path
    for n in('kvverify/continuation_training.py','cloud/continuation_worker.py','cloud/continuation_submit.py','scripts/run_continuation_smoke.py'):
        files[n]=ROOT/n
    return files


def tested_hashes():
    return {n:digest(p)for n,p in sorted({**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}.items())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-receipt',type=Path,required=True);p.add_argument('--smoke-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--submit',action='store_true')
    p.add_argument('--objective',choices=('continuation','question_readout'),default='continuation')
    p.add_argument('--native-prefix-policy',choices=('none','header','64','256'),default='none')
    p.add_argument('--steps',type=int);p.add_argument('--validate-every',type=int);p.add_argument('--learning-rate',type=float)
    args=p.parse_args()
    steps=args.steps if args.steps is not None else(1500 if args.objective=='continuation'else 300)
    interval=args.validate_every if args.validate_every is not None else(250 if args.objective=='continuation'else 50)
    lr=args.learning_rate if args.learning_rate is not None else(3e-5 if args.objective=='continuation'else 1e-5)
    if not 1<=steps<=1500 or not 1<=interval<=250 or not 0<lr<=3e-5:raise ValueError('Training bounds exceeded')
    if args.objective=='continuation'and args.native_prefix_policy!='none':raise ValueError('Native prefix is a task-readout policy')
    smoke=json.loads(args.smoke_report.read_text())
    if smoke.get('status')!='passed'or smoke.get('mode')!='continuation'or smoke.get('file_sha256')!=tested_hashes()or any(smoke.get(k)!=0 for k in('errors','failures','skipped')):
        raise ValueError('Fresh complete unchanged-source smoke required')
    parent=json.loads(args.parent_receipt.read_text());name=parent['job_name']
    if parent['mode']not in('linear_search','continuation')or parent['profile']!=PROFILE or parent['region']!=REGION:raise ValueError('Wrong parent scope')
    status=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    if status['TrainingJobStatus']not in('Completed','Stopped','Failed'):raise ValueError('Parent must be terminal before a new training stage')
    origin=f's3://{BUCKET}/kv-transfer/{name}/live/'
    if parent['mode']=='linear_search':
        if status['TrainingJobStatus']!='Completed':raise ValueError('Linear selection must complete')
        result=json.loads(aws('s3','cp',origin+'summary.json','-','--only-show-errors'))
        if result.get('status')!='completed'or result.get('validation_documents')!=30:raise ValueError('Incomplete parent selection')
        mapper_origin=origin+'selected/'+result['selected_support']+'/'
        manifest=json.loads(aws('s3','cp',mapper_origin+'manifest.json','-','--only-show-errors'))
        if manifest.get('trained_steps')!=0 or manifest.get('selected_candidate')!=result['selected_candidate']:raise ValueError('Selected initializer mismatch')
    else:
        result=json.loads(aws('s3','cp',origin+'best-selection.json','-','--only-show-errors'))
        mapper_origin=f's3://{BUCKET}/kv-transfer/{name}/checkpoints/step-{result["step"]:06d}/'
        manifest=json.loads(aws('s3','cp',mapper_origin+'manifest.json','-','--only-show-errors'))
        if manifest['selected_at_step']!=result['step']or manifest['weights_sha256']!=result['weights_sha256']:
            raise ValueError('Validated weights-only restart identity mismatch')
    if manifest.get('mapper_format')!='affine-cache-v1':raise ValueError('Unsupported initializer')
    ancestry=manifest.get('total_weight_updates',manifest.get('selected_at_step',0))
    ledger=reconcile(7200)
    cfg={'schema_version':1,'mode':'continuation','models':MODELS,'dtype':'float32','seed':19,
        'max_runtime_seconds':7200,'worker_timeout_seconds':6900,'mapper_origin':mapper_origin,
        'input_mapper_dir':'/opt/ml/input/data/mapper','initial_weights_sha256':manifest['weights_sha256'],
        'steps':steps,'validate_every':interval,'learning_rate':lr,'train_max_seconds':5400,'training_enabled':True,
        'training_objective':args.objective,'native_prefix_policy':args.native_prefix_policy,
        'initial_ancestor_updates':ancestry,'sampling_seed':parent.get('config',{}).get('sampling_seed',parent['config']['seed'])+(1 if parent['mode']=='continuation'else 0),
        'restart_policy':'Weights-only restart with fresh optimizer; stage counters local, total_weight_updates records selected ancestry',
        'parent_job_name':name,'authorization':'User autonomous loop; notify before$50; no per-run questions.'}
    files={**sources(),**inputs()};cfg['input_sha256']={n:digest(path)for n,path in files.items()if n.startswith('reference/')}
    cfg['verified_worker_imports']=verify_worker_imports(files,'cloud/continuation_worker.py')
    job='sova-kv-continuation-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2);req=request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-continuation-')as temporary:
        temp=Path(temporary);configuration=temp/'config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration;bundle=temp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[n]!=smoke['file_sha256'][n]for n in sources()):raise ValueError('Packaged source drift')
        receipt={'status':'prepared','job_name':job,'mode':'continuation','profile':PROFILE,'region':REGION,
            'request':req,'config':cfg,'price':ledger['price'],'budget_ledger':ledger,'smoke_evidence':smoke,
            'estimated_max_worker_compute_usd':7200*ledger['price']['usd_per_hour']/3600,
            'file_sha256':hashes,'bundle_sha256':digest(bundle),'parent_selection':result,
            'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False);saved=args.output/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong account')
            request_file=temp/'request.json';request_file.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                response=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(request_file)))
                receipt.update(status='submitted',job_arn=response['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','estimated_max_worker_compute_usd','live_results')},indent=2))


if __name__=='__main__':main()
