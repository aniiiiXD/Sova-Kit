"""One bounded native-prefix or component-oracle control; serial managed GPU."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import tempfile
from .holdout_submit import sources as previous_sources,TESTS as PREVIOUS_TESTS
from .state_diagnostic_submit import inputs
from .submit import aws,ACCOUNT,BUCKET,PROFILE,REGION,MODELS
from .recovery_submit import request
from .full_submit import make_bundle,digest
from .autonomous_budget import reconcile
from .package_integrity import verify_worker_imports
ROOT=Path(__file__).resolve().parents[1]
TESTS=(*PREVIOUS_TESTS,'test_native_prefix.py')


def sources():
    files=previous_sources();files['scripts/run_prefix_probe_smoke.py']=ROOT/'scripts/run_prefix_probe_smoke.py';return files


def tested_hashes():
    return {n:digest(p)for n,p in sorted({**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}.items())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-receipt',type=Path,required=True)
    p.add_argument('--checkpoint-step',type=int,required=True);p.add_argument('--smoke-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--submit',action='store_true')
    p.add_argument('--components',action='store_true');p.add_argument('--project-keys',action='store_true');args=p.parse_args()
    if args.project_keys and not args.components:raise ValueError('Key geometry is an explicit component-diagnostic intervention')
    smoke=json.loads(args.smoke_report.read_text())
    if smoke.get('status')!='passed'or smoke.get('mode')!='prefix_probe'or smoke.get('file_sha256')!=tested_hashes()or any(smoke.get(k)!=0 for k in('failures','errors','skipped')):
        raise ValueError('Fresh complete unchanged-source smoke evidence required')
    parent=json.loads(args.parent_receipt.read_text());name=parent['job_name']
    if parent['mode']not in('continuation','layer_readout','matched_fit','linear_search')or parent['profile']!=PROFILE or parent['region']!=REGION:raise ValueError('Unexpected checkpoint parent scope')
    state=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    if state['TrainingJobStatus']not in('Completed','Stopped'):raise ValueError('Parent must be terminal before a serial diagnostic')
    if args.checkpoint_step<0:raise ValueError('Explicit frozen nonnegative step required')
    if parent['mode']=='linear_search':
        if args.checkpoint_step!=0 or state['TrainingJobStatus']!='Completed':raise ValueError('Closed-form initializer must be completed/untrained')
        summary=json.loads(aws('s3','cp',f's3://{BUCKET}/kv-transfer/{name}/live/summary.json','-','--only-show-errors'))
        support=summary['selected_support']
        if support not in('same_head','all_head'):raise ValueError('Unknown selected affine support')
        origin=f's3://{BUCKET}/kv-transfer/{name}/live/selected/{support}/'
    else:origin=f's3://{BUCKET}/kv-transfer/{name}/checkpoints/step-{args.checkpoint_step:06d}/'
    manifest=json.loads(aws('s3','cp',origin+'manifest.json','-','--only-show-errors'))
    if manifest.get('selected_at_step',0)!=args.checkpoint_step or manifest['mapper_format']!='affine-cache-v1':
        raise ValueError('Checkpoint metadata does not match frozen step')
    if parent['mode']=='linear_search'and(manifest.get('trained_steps')!=0 or manifest.get('selected_candidate')!=summary['selected_candidate']):
        raise ValueError('Closed-form selected initializer differs from its report')
    if parent['mode']=='layer_readout'and manifest.get('selection_axis')!='corrected_layers':raise ValueError('Readout counter must mean corrected layers')
    if parent['mode']=='matched_fit'and manifest.get('selection_axis')!='candidate_index':raise ValueError('Matched fit counter must mean candidate index')
    if manifest.get('key_projection','none')!='none':raise ValueError('Probe requires unprojected frozen weights; request projection as an explicit intervention')
    ledger=reconcile(1800)
    cfg={'schema_version':1,'mode':'prefix_probe','models':MODELS,'dtype':'float32','seed':19,
        'max_runtime_seconds':1800,'worker_timeout_seconds':1500,'mapper_origin':origin,'input_mapper_dir':'/opt/ml/input/data/mapper',
        'mapper_weights_sha256':manifest['weights_sha256'],'component_oracles':args.components,'project_keys':args.project_keys,
        'training_enabled':False,'parent_job_name':name,'checkpoint_step':args.checkpoint_step,
        'authorization':'User autonomous loop; bounded development control; all active runtime maxima reserved under$50'}
    files={**sources(),**inputs()};cfg['input_sha256']={n:digest(path)for n,path in files.items()if n.startswith('reference/')}
    cfg['verified_worker_imports']=verify_worker_imports(files,'cloud/prefix_probe_worker.py')
    job=('sova-kv-component-probe-'if args.components else'sova-kv-prefix-probe-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2);req=request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-prefix-control-')as temporary:
        temp=Path(temporary);configuration=temp/'config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration;bundle=temp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[n]!=smoke['file_sha256'][n]for n in sources()):raise ValueError('Source drift after tests')
        receipt={'status':'prepared','job_name':job,'mode':'prefix_probe','region':REGION,'profile':PROFILE,'config':cfg,'request':req,
            'price':ledger['price'],'budget_ledger':ledger,'estimated_max_worker_compute_usd':1800*ledger['price']['usd_per_hour']/3600,
            'file_sha256':hashes,'bundle_sha256':digest(bundle),'smoke_evidence':smoke,'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False);saved=args.output/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong account')
            path=temp/'request.json';path.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                result=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(path)))
                receipt.update(status='submitted',job_arn=result['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','estimated_max_worker_compute_usd')},indent=2))


if __name__=='__main__':main()
