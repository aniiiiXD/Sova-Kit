"""Prepare or submit one inference-only24-document state diagnostic.

No training code is invoked. --submit is required to allocate the managed GPU.
Default operation verifies metadata, price and test evidence and writes a plan.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
import tempfile

from .submit import ACCOUNT,BUCKET,MODELS,PROFILE,REGION,aws,price
from .full_submit import digest,make_bundle
from .recovery_submit import request,source_files as recovery_sources,MODE_TESTS,STEP_ZERO_MAPPER

ROOT=Path(__file__).resolve().parents[1]
TRAINED_MAPPER=f's3://{BUCKET}/kv-transfer/sova-kv-repair-20261118-173137-4f04/checkpoints/best-final/'
EXPECTED_WEIGHTS={'step0':'00494f97de95d0b180dbd5849d08e12f264fd826d5493161eb91815548dcd98b',
                  'step48':'e2f6390f176a0c4ef262edfaf18a133e5fe88a5a33a898c5415c757269d53c51'}
TESTS=(*MODE_TESTS['precision'],'test_state_diagnostic.py')


def sources():
    files=recovery_sources('precision')
    for name in('kvverify/state_diagnostic.py','cloud/state_diagnostic_worker.py',
                'cloud/state_diagnostic_submit.py','scripts/run_state_diagnostic_smoke.py'):
        files[name]=ROOT/name
    return files


def tested_hashes():
    files={**sources(),**{'tests/'+n:ROOT/'tests'/n for n in TESTS}}
    return {name:digest(path)for name,path in sorted(files.items())}


def verify_tests(path):
    result=json.loads(Path(path).read_text())
    if(result.get('mode')!='state_diagnostic'or result.get('status')!='passed'
       or result.get('tests_run',0)<1 or any(result.get(k)!=0 for k in('failures','errors','skipped'))
       or result.get('sources_unchanged_during_tests')is not True or result.get('file_sha256')!=tested_hashes()):
        raise ValueError('Fresh, complete, unchanged-source diagnostic smoke evidence required')
    return {'path':str(Path(path).resolve()),'sha256':digest(Path(path)),**result}


def checkpoint_config(name,manifest):
    expected_step=0 if name=='step0'else 48
    if (manifest.get('weights_sha256')!=EXPECTED_WEIGHTS[name]or manifest.get('trained_steps')!=expected_step
        or manifest.get('selected_at_step')!=expected_step or manifest.get('format_version')!=2):
        raise ValueError('Unexpected exact checkpoint: '+name)
    return {'origin':STEP_ZERO_MAPPER if name=='step0'else TRAINED_MAPPER,
        'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'manifest_hash_rule':'canonical-json-sortkeys-compact-ascii','weights_sha256':EXPECTED_WEIGHTS[name],
        'trained_steps':expected_step}


def configuration(runtime,checkpoints):
    if type(runtime)is not int or not 600<=runtime<=1800:raise ValueError('One diagnostic is bounded to600–1800seconds')
    return {'schema_version':1,'mode':'state_diagnostic','models':MODELS,'dtype':'float32','seed':19,
        'max_runtime_seconds':runtime,'worker_timeout_seconds':runtime-300,
        'mapper_origin':TRAINED_MAPPER,'input_mapper_dir':'/opt/ml/input/data/mapper',
        'input_initializer_dir':'/opt/ml/input/data/initializer','checkpoints':checkpoints,
        'training_enabled':False,'max_prompt_tokens':8192,'query_block_size':256,
        'native_oracle_protocol':'oracle-probability-v2','documents':24,
        'purpose':'Validation-only state interventions; no final quality or serving-speed claim'}


def diagnostic_request(job,cfg):
    req=request(job,cfg)
    req['InputDataConfig'].append({'ChannelName':'initializer','DataSource':{'S3DataSource':{
        'S3Uri':STEP_ZERO_MAPPER,'S3DataType':'S3Prefix'}}})
    return req


def cost_guard(rate,seconds,cap):
    if not math.isfinite(rate)or not 0<rate<=1.60 or type(seconds)is not int or not 600<=seconds<=1800:
        raise ValueError('Price or runtime outside the diagnostic scope')
    if not math.isfinite(cap)or not 0<cap<=.80:raise ValueError('Explicit runtime-compute cap must be <=$0.80')
    estimate=rate*seconds/3600
    if estimate>cap:raise ValueError('Maximum worker estimate exceeds approved cap')
    return estimate


def inputs():
    base=ROOT/'datasets/reference-benchmarks-v1'
    manifest=json.loads((base/'manifest.json').read_text())
    relative='processed/quality/train.jsonl';data=base/relative
    if digest(data)!=manifest['processed'][relative]['sha256']:raise ValueError('Frozen training-source hash changed')
    return {'reference/'+relative:data,'reference/manifest.json':base/'manifest.json'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-runtime-seconds',type=int,default=1800)
    parser.add_argument('--max-compute-usd',type=float,default=.80)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    smoke=verify_tests(args.smoke_report)
    checkpoints={}
    for name,origin in [('step0',STEP_ZERO_MAPPER),('step48',TRAINED_MAPPER)]:
        manifest=json.loads(aws('s3','cp',origin+'manifest.json','-','--only-show-errors'))
        checkpoints[name]=checkpoint_config(name,manifest)
    cfg=configuration(args.max_runtime_seconds,checkpoints)
    rate=price();estimate=cost_guard(rate['usd_per_hour'],args.max_runtime_seconds,args.max_compute_usd)
    files={**sources(),**inputs()}
    cfg['input_sha256']={name:digest(path)for name,path in files.items()if name.startswith('reference/')}
    job='sova-kv-state-diag-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    req=diagnostic_request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-state-diagnostic-')as tmp:
        tmp=Path(tmp);configuration_file=tmp/'config.json'
        configuration_file.write_text(json.dumps(cfg,indent=2)+'\n');files['cloud/recovery-config.json']=configuration_file
        bundle=tmp/'bundle.tar.gz';hashes=make_bundle(bundle,files,cfg)
        if any(hashes[name]!=smoke['file_sha256'][name]for name in sources()):
            raise ValueError('Source changed between tests and packaging')
        receipt={'status':'prepared','job_name':job,'region':REGION,'profile':PROFILE,'mode':'state_diagnostic',
            'config':cfg,'request':req,'price':rate,'max_worker_compute_usd':estimate,'runtime_compute_cap_usd':args.max_compute_usd,
            'cost_note':'Public-rate runtime estimate only; storage, transfer, provisioning/billing extras and tax excluded.',
            'training_enabled':False,'smoke_evidence':smoke,'file_sha256':hashes,'bundle_sha256':digest(bundle),
            'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/'}
        args.output.mkdir(parents=True,exist_ok=False)
        saved=args.output/'submission.json'
        saved.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong AWS account')
            request_file=tmp/'request.json';request_file.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                result=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(request_file)))
                receipt.update(status='submitted',job_arn=result['TrainingJobArn'])
            except BaseException as exc:
                receipt.update(status='submission_failed',error=str(exc));saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k]for k in('status','job_name','region','max_worker_compute_usd','live_results')},indent=2))


if __name__=='__main__':main()
