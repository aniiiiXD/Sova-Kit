"""One bounded layer-ordered value-calibration job under the autonomous budget."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import tempfile
from .holdout_submit import sources as prior_sources,TESTS
from .state_diagnostic_submit import inputs
from .submit import aws,ACCOUNT,BUCKET,PROFILE,REGION,MODELS
from .recovery_submit import request
from .full_submit import make_bundle,digest
from .autonomous_budget import reconcile
from .package_integrity import verify_worker_imports
ROOT=Path(__file__).resolve().parents[1]


def sources():
    files=prior_sources();files['scripts/run_layer_readout_smoke.py']=ROOT/'scripts/run_layer_readout_smoke.py'
    return files


def tested_hashes():
    return {n:digest(p)for n,p in sorted({**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}.items())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-receipt',type=Path,required=True);p.add_argument('--smoke-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--submit',action='store_true');args=p.parse_args()
    smoke=json.loads(args.smoke_report.read_text())
    if smoke.get('status')!='passed'or smoke.get('mode')!='layer_readout'or smoke.get('file_sha256')!=tested_hashes()or any(smoke.get(k)!=0 for k in('errors','failures','skipped')):
        raise ValueError('Fresh complete source-frozen calibration smoke required')
    parent=json.loads(args.parent_receipt.read_text());name=parent['job_name']
    if parent['mode']!='continuation'or parent['profile']!=PROFILE or parent['region']!=REGION:raise ValueError('Wrong selected parent scope')
    state=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    if state['TrainingJobStatus']not in('Completed','Stopped'):raise ValueError('Parent must be terminal')
    result=json.loads(aws('s3','cp',f's3://{BUCKET}/kv-transfer/{name}/live/best-selection.json','-','--only-show-errors'))
    origin=f's3://{BUCKET}/kv-transfer/{name}/checkpoints/step-{result["step"]:06d}/'
    manifest=json.loads(aws('s3','cp',origin+'manifest.json','-','--only-show-errors'))
    if manifest['weights_sha256']!=result['weights_sha256']or manifest['mapper_format']!='affine-cache-v1'or manifest.get('native_prefix_policy')!='256':
        raise ValueError('Selected native256initialization changed')
    ledger=reconcile(5400)
    cfg={'schema_version':1,'mode':'layer_readout','models':MODELS,'seed':19,'dtype':'float32',
        'max_runtime_seconds':5400,'worker_timeout_seconds':5100,'mapper_origin':origin,
        'input_mapper_dir':'/opt/ml/input/data/mapper','initial_weights_sha256':manifest['weights_sha256'],
        'energy_fraction':.01,'native_prefix_policy':'256','optimizer_updates':0,
        'parent_job_name':name,'authorization':'User autonomous analyze/fix/test loop under$50; no per-run approval'}
    files={**sources(),**inputs()};cfg['input_sha256']={n:digest(path)for n,path in files.items()if n.startswith('reference/')}
    cfg['verified_worker_imports']=verify_worker_imports(files,'cloud/layer_readout_worker.py')
    job='sova-kv-layer-readout-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    req=request(job,cfg);req['ResourceConfig']['VolumeSizeInGB']=128
    with tempfile.TemporaryDirectory(prefix='sova-layer-readout-')as td:
        temp=Path(td);configuration=temp/'config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration;bundle=temp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[n]!=smoke['file_sha256'][n]for n in sources()):raise ValueError('Source drift during packaging')
        receipt={'status':'prepared','job_name':job,'mode':'layer_readout','profile':PROFILE,'region':REGION,
            'request':req,'config':cfg,'price':ledger['price'],'budget_ledger':ledger,'smoke_evidence':smoke,
            'estimated_max_worker_compute_usd':5400*ledger['price']['usd_per_hour']/3600,'parent_selection':result,
            'file_sha256':hashes,'bundle_sha256':digest(bundle),'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False);saved=args.output/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong AWS account')
            request_file=temp/'request.json';request_file.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                response=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(request_file)))
                receipt.update(status='submitted',job_arn=response['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','estimated_max_worker_compute_usd')},indent=2))


if __name__=='__main__':main()
