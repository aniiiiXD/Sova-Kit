"""Promote one frozen development-qualified artifact to reserved acceptance."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import secrets
import tempfile
from .continuation_submit import sources as prior_sources,TESTS as PRIOR_TESTS
from .submit import aws,ACCOUNT,BUCKET,PROFILE,REGION,MODELS
from .recovery_submit import request
from .full_submit import make_bundle,digest
from .autonomous_budget import reconcile
from .package_integrity import verify_worker_imports
ROOT=Path(__file__).resolve().parents[1]
TESTS=(*PRIOR_TESTS,'test_holdout_gate.py','test_transfer_economics.py','test_prefix_competitor.py')


def promotion(summary):
    selected=[row for row in summary.get('validation_history',[])if row['step']==summary.get('selected_step')]
    if len(selected)!=1:raise ValueError('One selected development checkpoint required')
    row=selected[0]
    if any(type(row.get(k))is not int or not 0<=row[k]<=30 for k in('documents','native_correct','correct','native_agreement')):
        raise ValueError('Finite integral development denominators required')
    if not isinstance(row.get('mean_native_kl'),(int,float))or not math.isfinite(row['mean_native_kl'])or row['mean_native_kl']<0:
        raise ValueError('Finite nonnegative native KL required')
    if row['documents']!=30 or row['native_correct']<12:raise ValueError('Expected useful native baseline on30development articles')
    if(row['correct']<.9*row['native_correct']or row['native_agreement']<.8*row['documents']or row['mean_native_kl']>.5):
        raise ValueError('Development promotion gate not met; keep acceptance sealed: '+json.dumps(row))
    return row


def stopped_checkpoint_summary(best,history):
    """A validated immutable checkpoint can outlive an intentionally stopped run."""
    matches=[r for r in history if r.get('step')==best.get('step')]
    if len(matches)!=1 or any(matches[0].get(k)!=best.get(k)for k in(
        'documents','correct','native_correct','native_agreement','mean_native_kl')):
        raise ValueError('Checkpoint record must match its complete validation row')
    summary={'selected_step':best['step'],'validation_history':history,
             'native_prefix_policy':best.get('native_prefix_policy','none'),
             'key_projection':best.get('key_projection','none'),
             'parent_training_state':'Stopped; selected checkpoint only, not completed requested training'}
    promotion(summary)
    return summary


def sources():
    files=prior_sources();files['scripts/run_holdout_smoke.py']=ROOT/'scripts/run_holdout_smoke.py'
    return files


def tested_hashes():
    return {n:digest(p)for n,p in sorted({**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}.items())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-receipt',type=Path,required=True)
    p.add_argument('--smoke-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--submit',action='store_true');args=p.parse_args()
    smoke=json.loads(args.smoke_report.read_text())
    if smoke.get('status')!='passed'or smoke.get('mode')!='holdout'or smoke.get('file_sha256')!=tested_hashes()or any(smoke.get(k)!=0 for k in('failures','errors','skipped')):
        raise ValueError('Fresh unchanged-source complete smoke evidence required')
    parent=json.loads(args.parent_receipt.read_text());name=parent['job_name']
    if parent['mode']not in('continuation','layer_readout','matched_fit')or parent['profile']!=PROFILE or parent['region']!=REGION:raise ValueError('Wrong parent scope')
    state=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    if state['TrainingJobStatus']not in('Completed','Stopped'):raise ValueError('Training parent must be completed or intentionally stopped with a validated checkpoint')
    prefix=f's3://{BUCKET}/kv-transfer/{name}/live/'
    best=json.loads(aws('s3','cp',prefix+'best-selection.json','-','--only-show-errors'))
    if state['TrainingJobStatus']=='Completed':
        summary=json.loads(aws('s3','cp',prefix+'summary.json','-','--only-show-errors'))
        mapper_origin=prefix+'best/'
    else:
        history=json.loads(aws('s3','cp',prefix+'validation-history.json','-','--only-show-errors'))
        summary=stopped_checkpoint_summary(best,history)
        mapper_origin=f's3://{BUCKET}/kv-transfer/{name}/checkpoints/step-{best["step"]:06d}/'
    gate=promotion(summary)
    manifest=json.loads(aws('s3','cp',mapper_origin+'manifest.json','-','--only-show-errors'))
    if manifest['selected_at_step']!=summary['selected_step']or manifest['mapper_format']!='affine-cache-v1':raise ValueError('Frozen candidate identity mismatch')
    if manifest['weights_sha256']!=best['weights_sha256']or best['step']!=summary['selected_step']:raise ValueError('Selected artifact hash or step differs')
    policy=manifest.get('native_prefix_policy','none')
    if policy not in('none','256')or policy!=summary.get('native_prefix_policy','none'):
        raise ValueError('Explicit native-prefix policy differs between selected artifact and development summary')
    key_policy=manifest.get('key_projection','none')
    if key_policy not in('none','target_rms_ellipsoid')or key_policy!=summary.get('key_projection','none'):
        raise ValueError('Frozen key-projection policy differs from selected validation')
    ledger=reconcile(10800)
    cfg={'schema_version':1,'mode':'holdout','models':MODELS,'seed':20261119,'dtype':'float32','max_runtime_seconds':10800,
        'worker_timeout_seconds':10500,'mapper_origin':mapper_origin,'input_mapper_dir':'/opt/ml/input/data/mapper',
        'mapper_weights_sha256':manifest['weights_sha256'],'native_prefix_policy':policy,'key_projection':key_policy,
        'parent_training_state':state['TrainingJobStatus'],
        'training_enabled':False,'parent_job_name':name,'promotion_record':gate,
        'authorization':'User autonomous loop under$50; one frozen candidate on reserved public sources'}
    files=sources();base=ROOT/'datasets/loop-holdout-v1';dataset=json.loads((base/'manifest.json').read_text())
    for n in('quality.jsonl','hellaswag.jsonl'):
        if digest(base/n)!=dataset['files'][n]['sha256']:raise ValueError('Reserved evaluation bytes changed')
        files['acceptance/'+n]=base/n
    files['acceptance/manifest.json']=base/'manifest.json'
    cfg['input_sha256']={n:digest(path)for n,path in files.items()if n.startswith('acceptance/')}
    cfg['verified_worker_imports']=verify_worker_imports(files,'cloud/holdout_worker.py')
    job='sova-kv-holdout-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2);req=request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-holdout-')as temporary:
        temp=Path(temporary);configuration=temp/'config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration;bundle=temp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[n]!=smoke['file_sha256'][n]for n in sources()):raise ValueError('Packaged source drift')
        receipt={'status':'prepared','job_name':job,'mode':'holdout','region':REGION,'profile':PROFILE,'request':req,'config':cfg,
            'price':ledger['price'],'budget_ledger':ledger,'estimated_max_worker_compute_usd':10800*ledger['price']['usd_per_hour']/3600,
            'file_sha256':hashes,'bundle_sha256':digest(bundle),'smoke_evidence':smoke,'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False);saved=args.output/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong AWS account')
            reqfile=temp/'request.json';reqfile.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                response=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(reqfile)))
                receipt.update(status='submitted',job_arn=response['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','estimated_max_worker_compute_usd')},indent=2))


if __name__=='__main__':main()
