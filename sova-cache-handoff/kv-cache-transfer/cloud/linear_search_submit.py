"""Bounded autonomous linear-reference experiment under the user's$50 cap."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import tempfile
from .state_diagnostic_submit import sources as previous_sources,TESTS as PREVIOUS_TESTS,inputs,TRAINED_MAPPER,EXPECTED_WEIGHTS
from .submit import aws,ACCOUNT,BUCKET,PROFILE,REGION,MODELS
from .recovery_submit import request
from .full_submit import make_bundle,digest
from .autonomous_budget import reconcile

ROOT=Path(__file__).resolve().parents[1]
TESTS=(*PREVIOUS_TESTS,'test_affine_cache.py')


def sources():
    files=previous_sources()
    for name in('kvverify/affine_cache.py','cloud/linear_search_worker.py','cloud/linear_search_submit.py',
                'cloud/autonomous_budget.py','scripts/run_linear_search_smoke.py'):
        files[name]=ROOT/name
    return files


def tested_hashes():
    return {n:digest(p)for n,p in sorted({**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}.items())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-runtime-seconds',type=int,default=5400);p.add_argument('--submit',action='store_true');args=p.parse_args()
    if not 1800<=args.max_runtime_seconds<=7200:raise ValueError('Bounded30–120minute calibration/selection worker')
    smoke=json.loads(args.smoke_report.read_text())
    if(smoke.get('status')!='passed'or smoke.get('mode')!='linear_search'or smoke.get('file_sha256')!=tested_hashes()
       or any(smoke.get(k)!=0 for k in('errors','failures','skipped'))):raise ValueError('Fresh unchanged-source smoke evidence required')
    ledger=reconcile(args.max_runtime_seconds)
    cfg={'schema_version':1,'mode':'linear_search','models':MODELS,'dtype':'float32','seed':19,
        'max_runtime_seconds':args.max_runtime_seconds,'worker_timeout_seconds':args.max_runtime_seconds-300,
        'mapper_origin':TRAINED_MAPPER,'input_mapper_dir':'/opt/ml/input/data/mapper',
        'old_mapper_weights_sha256':EXPECTED_WEIGHTS['step48'],'training_enabled':False,
        'alphas':[1e-7,1e-5,.001,.01],'max_prompt_tokens':8192,'query_block_size':256,
        'authorization':'User authorizes autonomous diagnose/fix/test loop; notify at$50 total; no per-run approval.'}
    files={**sources(),**inputs()};cfg['input_sha256']={n:digest(path)for n,path in files.items()if n.startswith('reference/')}
    job='sova-kv-linear-search-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    req=request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-linear-search-')as temporary:
        temp=Path(temporary);configuration=temp/'config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration;bundle=temp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[n]!=smoke['file_sha256'][n]for n in sources()):raise ValueError('Packaged source drift')
        receipt={'status':'prepared','job_name':job,'profile':PROFILE,'region':REGION,'mode':'linear_search',
            'config':cfg,'request':req,'price':ledger['price'],'budget_ledger':ledger,
            'estimated_max_worker_compute_usd':args.max_runtime_seconds*ledger['price']['usd_per_hour']/3600,
            'file_sha256':hashes,'bundle_sha256':digest(bundle),'smoke_evidence':smoke,
            'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False);saved=args.output/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong AWS account')
            req_file=temp/'request.json';req_file.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                response=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(req_file)))
                receipt.update(status='submitted',job_arn=response['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','estimated_max_worker_compute_usd','live_results')},indent=2))


if __name__=='__main__':main()
