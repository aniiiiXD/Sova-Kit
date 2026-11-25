"""Submit one scoped recovery job with immutable test evidence and a cost guard."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import secrets
import subprocess
import tempfile

try:
    from .submit import ACCOUNT, BUCKET, IMAGE, MODELS, PROFILE, REGION, aws, price
    from .full_submit import make_bundle, digest, compute_budget
except ImportError:
    from submit import ACCOUNT, BUCKET, IMAGE, MODELS, PROFILE, REGION, aws, price
    from full_submit import make_bundle, digest, compute_budget

ROOT=Path(__file__).resolve().parents[1]
SAVED_MAPPER='s3://'+BUCKET+'/kv-transfer/sova-kv-full-20261118-132341-7daf/checkpoints/best-final/'
STEP_ZERO_MAPPER='s3://'+BUCKET+'/kv-transfer/sova-kv-repair-20261118-162905-150e/checkpoints/step-000000/'
BASE_MODULES=('__init__.py','data.py','mathops.py','hf_backend.py','quality.py','evidence.py','train_mapper.py')
MODE_MODULES={'probe':('boundary_probe.py',),'repair':('boundary_probe.py','qa_training.py','reference_eval.py','blocked_attention.py','activation_offload.py'),
              'precision':('boundary_probe.py','qa_training.py','reference_eval.py','blocked_attention.py')}
MODE_TESTS={'probe':('test_core.py','test_hf_optional.py','test_train_mapper.py','test_boundary_probe.py','test_recovery_cloud.py'),
            'repair':('test_core.py','test_hf_optional.py','test_train_mapper.py','test_boundary_probe.py','test_recovery_cloud.py','test_qa_training.py','test_reference_eval.py','test_reference_preparation.py','test_repair_worker.py','test_blocked_attention.py','test_training_memory_policy.py','test_activation_offload.py')}
MODE_TESTS['precision']=(*MODE_TESTS['repair'],'test_precision_worker.py')
MODE_MODULES['eval_resume']=(*MODE_MODULES['repair'],'eval_resume.py')
MODE_TESTS['eval_resume']=(*MODE_TESTS['repair'],'test_eval_resume.py')


def source_files(mode):
    if mode not in MODE_MODULES:raise ValueError('Unknown recovery mode')
    names=['kvverify/'+n for n in (*BASE_MODULES,*MODE_MODULES[mode])]
    names += ['cloud/'+n for n in ('submit.py','full_submit.py','recovery_submit.py','recovery_bootstrap.sh',mode+'_worker.py')]
    if mode in ('repair','precision'):names.append('cloud/probe_worker.py')
    if mode=='precision':names.append('cloud/repair_worker.py')
    if mode=='repair':names.append('cloud/precision_worker.py')
    if mode=='eval_resume':names+=['cloud/'+n for n in('probe_worker.py','precision_worker.py','repair_worker.py')]
    return {n:ROOT/n for n in names}


def tested_hashes(mode):
    files={**source_files(mode),**{'tests/'+n:ROOT/'tests'/n for n in MODE_TESTS[mode]}}
    return {n:digest(p) for n,p in sorted(files.items())}


def config(mode,runtime):
    limit={'probe':1800,'repair':3600,'precision':1200,'eval_resume':1800}[mode]
    if type(runtime)is not int or not 600<=runtime<=limit:raise ValueError('Runtime outside the scoped recovery limit')
    return {'schema_version':1,'mode':mode,'models':MODELS,'dtype':'float16','seed':19,
        'max_runtime_seconds':runtime,'worker_timeout_seconds':runtime-300,
        'mapper_origin':SAVED_MAPPER,'input_mapper_dir':'/opt/ml/input/data/mapper',
        'output_dir':'/opt/ml/model','probe_examples':16,'tails':[1,8,32],
        'max_prompt_tokens':8192,'train_steps':400,'train_max_seconds':1800,
        'training_answer_tokens':8,'quality_freegen_tokens':32,'benchmark_max_seconds':1800,
        'native_control_examples':16,'minimum_native_control_accuracy':.35,
        'require_fresh_native_oracle_gate':mode=='repair',
        'fresh_oracle_protocol':'oracle-probability-v2',
        'precision_arms':[{'name':'fp16_auto','dtype':'float16','backend':'auto'},
            {'name':'fp16_math','dtype':'float16','backend':'math'},
            {'name':'fp32_math','dtype':'float32','backend':'math'}],
        'training_enabled':mode=='repair',
        'train_checkpoint_every':50,'training_saved_tensors_cpu':False,
        'training_memory_preflight':False,'quality_max_seconds':600,'eval_max_prompt_tokens':8192,
        'note':'Explicit development diagnostic; no exact-vendor replication or positive result implied.'}


def request(job,cfg):
    prefix='kv-transfer/'+job
    entry='mkdir -p /opt/ml/kvverify && tar -xzf /opt/ml/input/data/code/bundle.tar.gz -C /opt/ml/kvverify && exec bash /opt/ml/kvverify/cloud/recovery_bootstrap.sh'
    req={'TrainingJobName':job,'RoleArn':f'arn:aws:iam::{ACCOUNT}:role/sova-sagemaker-exec',
        'AlgorithmSpecification':{'TrainingImage':IMAGE,'TrainingInputMode':'File','ContainerEntrypoint':['bash','-c',entry]},
        'ResourceConfig':{'InstanceType':'ml.g5.2xlarge','InstanceCount':1,'VolumeSizeInGB':50},
        'InputDataConfig':[{'ChannelName':'code','DataSource':{'S3DataSource':{'S3Uri':f's3://{BUCKET}/{prefix}/input/','S3DataType':'S3Prefix'}}},
                           {'ChannelName':'mapper','DataSource':{'S3DataSource':{'S3Uri':cfg['mapper_origin'],'S3DataType':'S3Prefix'}}}],
        'OutputDataConfig':{'S3OutputPath':f's3://{BUCKET}/{prefix}/output/'},
        'CheckpointConfig':{'S3Uri':f's3://{BUCKET}/{prefix}/checkpoints/','LocalPath':'/opt/ml/checkpoints'},
        'StoppingCondition':{'MaxRuntimeInSeconds':cfg['max_runtime_seconds']},
        'Environment':{'KV_BUCKET':BUCKET,'KV_PREFIX':prefix,'KV_MODE':cfg['mode'],
            'KV_CONFIG':'/opt/ml/kvverify/cloud/recovery-config.json','KV_WORKER_TIMEOUT':str(cfg['worker_timeout_seconds'])},
        'EnableManagedSpotTraining':False,
        'Tags':[{'Key':'Project','Value':'sova-kv-transfer'},{'Key':'Scope','Value':cfg['mode']+'-recovery'}]}
    if cfg['mode']=='precision':req['InputDataConfig']=req['InputDataConfig'][:1]
    return req


def inputs(mode):
    data={}
    if mode=='probe':
        p=ROOT/'datasets/prepared-v1/manifest.json';m=json.loads(p.read_text())
        row=m['files']['dev.jsonl'];d=p.parent/'dev.jsonl'
        if digest(d)!=row['sha256']:raise ValueError('Original development input drifted')
        data={'inputs/dev.jsonl':d,'inputs/manifest.json':p}
    else:
        p=ROOT/'datasets/reference-benchmarks-v1/manifest.json';m=json.loads(p.read_text())
        data['reference/manifest.json']=p
        for n in ('processed/quality/train.jsonl','processed/quality/dev-100.jsonl','processed/hellaswag/validation-1000.jsonl'):
            d=p.parent/n
            if digest(d)!=m['processed'][n]['sha256']:raise ValueError('Reference input drifted: '+n)
            data['reference/'+n]=d
        if mode=='precision':
            p=ROOT/'reports/sova-kv-repair-20261118-143720-3fef/collected/native-control.json'
            data['inputs/prior-native-control.json']=p
    return data


def verify_probe_summary(path, fresh_oracle_gate=False):
    evidence=json.loads(path.read_text())
    gate=evidence.get('oracle_gate',{})
    original_pass=(evidence.get('protocol')=='tail-boundary-v1'
        and gate.get('status')=='passed' and gate.get('matches_native')==48)
    # A revised, prospective worker gate may authorize training. The failed
    # historical raw-logit gate is retained verbatim, never relabeled passed.
    historical_complete=(evidence.get('protocol')=='tail-boundary-v1'
        and gate.get('status')=='failed' and gate.get('attempts')==48
        and gate.get('checked_or_failed')==48
        and all(evidence.get('arms',{}).get('oracle_suffix_'+str(t),{}).get('returned')==16
            and evidence['arms']['oracle_suffix_'+str(t)].get('errors')==0
            and evidence['arms']['oracle_suffix_'+str(t)].get('native_first_token_agreement_all_attempts')==1.
            for t in (1,8,32)))
    if not original_pass and not (fresh_oracle_gate and historical_complete):
        raise ValueError('All 48 native-cache oracle comparisons must pass before repair submission')
    return {'path':str(path.resolve()),'sha256':digest(path),'oracle_gate':gate,
        'historical_gate_passed':original_pass,
        'training_requires':'Fresh oracle-probability-v2 gate on unseen QuALITY mapper-validation questions inside this worker.'}


def continuation_inputs(receipt,cfg):
    """Freeze terminal parent execution evidence and its selected mapper."""
    parent=json.loads(receipt.read_text())
    if parent.get('region')!=REGION or parent.get('profile')!=PROFILE or parent.get('mode')!='repair':
        raise ValueError('Unexpected continuation parent scope')
    name=parent['job_name']
    if not name.startswith('sova-kv-repair-')or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-'for c in name):
        raise ValueError('Unexpected parent job name')
    description=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    if description['TrainingJobStatus']not in('Completed','Failed','Stopped'):
        raise ValueError('Parent must be terminal before freezing continuation evidence')
    for source in('kvverify/data.py','kvverify/hf_backend.py','kvverify/quality.py','kvverify/reference_eval.py',
                   'kvverify/train_mapper.py','kvverify/blocked_attention.py','kvverify/qa_training.py'):
        if parent['file_sha256'].get(source)!=digest(ROOT/source):
            raise ValueError('Continuation semantic source drift: '+source)
    prefix=f's3://{BUCKET}/kv-transfer/{name}/live/'
    destination=receipt.parent/'continuation-inputs';destination.mkdir(exist_ok=True)
    aws('s3','sync',prefix,str(destination),'--exclude','*','--include','config.json',
        '--include','quality/config.json','--include','quality/rows.jsonl',
        '--include','hellaswag/config.json','--include','hellaswag/rows.jsonl','--only-show-errors')
    worker=destination/'config.json'
    if json.loads(worker.read_text())!=parent['config']:raise ValueError('Exact parent worker config drifted')
    files={'inputs/parent/worker-config.json':worker}
    absent=[]
    for benchmark in('quality','hellaswag'):
        paths=[destination/benchmark/n for n in('config.json','rows.jsonl')]
        if not any(p.exists()for p in paths):absent.append(benchmark)
        elif not all(p.exists()for p in paths):raise ValueError('Incomplete parent evidence pair')
        else:
            for path in paths:files['inputs/parent/'+benchmark+'/'+path.name]=path
    origin=f's3://{BUCKET}/kv-transfer/{name}/checkpoints/best-final/'
    mapper=json.loads(aws('s3','cp',origin+'manifest.json','-','--only-show-errors'))
    cfg.update({k:parent['config'][k]for k in('models','dtype','attention_backend','query_block_size',
        'eval_max_prompt_tokens','quality_freegen_tokens','seed')})
    cfg.update(parent_job_name=name,parent_not_started_benchmarks=absent,mapper_origin=origin,
        mapper_weights_sha256=mapper['weights_sha256'],quality_max_seconds=900,hellaswag_max_seconds=900,
        training_enabled=False,continuation_only=True,parent_receipt_sha256=digest(receipt))
    return files


def verify_smoke(mode,path):
    report=json.loads(path.read_text())
    if (report.get('status')!='passed' or report.get('mode')!=mode or report.get('tests_run',0)<40
        or any(report.get(k,-1)!=0 for k in ('failures','errors','skipped'))
        or report.get('file_sha256')!=tested_hashes(mode)
        or report.get('sources_unchanged_during_tests')is not True):
        raise ValueError('Current scoped recovery tests and immutable hashes must pass')
    return {'path':str(path.resolve()),'sha256':digest(path),'tests_run':report['tests_run'],'file_sha256':report['file_sha256']}


def verify_precision_summary(path):
    evidence=json.loads(path.read_text());arm=evidence.get('arms',{}).get('fp32_math',{})
    rows=arm.get('rows',[]);settings=evidence.get('settings',{})
    if (evidence.get('status')!='completed' or arm.get('all_passed')is not True
        or arm.get('examples')!=16 or arm.get('passed')!=16 or arm.get('query_block_size')!=256
        or len(rows)!=16 or len({r.get('id')for r in rows})!=16
        or not all(r.get('status')=='ok'and r.get('oracle',{}).get('passed')is True for r in rows)
        or settings.get('matmul_tf32')is not False or settings.get('cudnn_tf32')is not False
        or settings.get('float32_matmul_precision')!='highest'):
        raise ValueError('Complete 16/16 FP32 math query-block-256 evidence required')
    return {'path':str(path.resolve()),'sha256':digest(path),'dtype':'float32',
        'attention_backend':'math','query_block_size':256,'controls_passed':16}


def verify_packaged_sources(mode,hashes,smoke):
    for name in source_files(mode):
        if hashes.get(name)!=smoke['file_sha256'].get(name):
            raise ValueError('Bundled source differs from tested bytes: '+name)


def recovery_budget(rate,seconds,prior,budget):
    """Keep the historical launcher cap intact; bound the explicit repair goal."""
    if budget==3.25:return compute_budget(rate,seconds,prior,budget)
    if (budget!=5.0 or type(seconds)is not int or not 600<=seconds<=3600
        or not all(math.isfinite(v)for v in(rate,prior,budget))or rate<=0 or not 0<=prior<=budget):
        raise ValueError('Finite bounded repair-goal pricing and runtime required')
    estimate=rate*seconds/3600;total=prior+estimate
    if not math.isfinite(total)or total>budget:raise RuntimeError('Repair-goal runtime estimate exceeds explicit guard')
    return estimate,total


def job_name(mode):
    if mode not in MODE_MODULES:raise ValueError('Unknown recovery mode')
    return 'sova-kv-'+mode.replace('_','-')+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=tuple(MODE_MODULES),required=True)
    parser.add_argument('--smoke-report',type=Path,required=True)
    parser.add_argument('--prior-compute-usd',type=float,required=True)
    parser.add_argument('--compute-guard-usd',type=float,choices=(3.25,5.0),default=3.25,help='Explicit cumulative runtime-compute cap; 5 is reserved for the user-requested persistent repair goal')
    parser.add_argument('--max-runtime-seconds',type=int)
    parser.add_argument('--probe-summary',type=Path,help='Required passed 16-document oracle evidence for repair')
    parser.add_argument('--fresh-oracle-gate',action='store_true',help='Preserve a complete failed historical raw-logit gate; require fresh probability-level native-cache controls before training')
    parser.add_argument('--query-block-size',type=int,help='Precision diagnostic only: bound math attention temporary query tiles; all keys/context retained')
    parser.add_argument('--fp32-only',action='store_true',help='Precision diagnostic only: rerun the FP32 math arm')
    parser.add_argument('--precision-summary',type=Path,help='Repair only: passed FP32 blocked-math controls; use exactly this execution path')
    parser.add_argument('--memory-safe-pilot',action='store_true',help='Repair only: 100 updates, bounded replay + CPU saved tensors + worst-case backward preflight; reserve evaluation time')
    parser.add_argument('--memory-safe-resume',action='store_true',help='Repair only: reuse validated untrained QuALITY step-zero initialization; 100-update pilot and selective pageable activation offload')
    parser.add_argument('--parent-receipt',type=Path,help='Evaluation continuation only: exact terminal parent receipt')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    runtime=args.max_runtime_seconds or {'probe':1800,'repair':3600,'precision':1200,'eval_resume':1800}[args.mode]
    cfg=config(args.mode,runtime)
    if args.query_block_size is not None or args.fp32_only:
        if args.mode!='precision':parser.error('Diagnostic precision switches require --mode precision')
        if args.query_block_size is not None and not 64<=args.query_block_size<=1024:
            parser.error('Query block must be between 64 and 1024')
        cfg['query_block_size']=args.query_block_size
        if args.fp32_only:cfg['precision_arms']=[r for r in cfg['precision_arms']if r['dtype']=='float32']
    precision=None
    if args.precision_summary:
        if args.mode!='repair':parser.error('--precision-summary requires --mode repair')
        precision=verify_precision_summary(args.precision_summary)
        cfg.update(dtype='float32',attention_backend='math',query_block_size=256)
    if args.memory_safe_pilot:
        if args.mode!='repair'or not args.precision_summary:parser.error('Memory-safe pilot requires FP32 repair proof')
        cfg.update(train_steps=100,train_checkpoint_every=100,train_max_seconds=1300,
            training_saved_tensors_cpu=True,training_memory_preflight=True,
            quality_freegen_tokens=16,quality_max_seconds=900,eval_max_prompt_tokens=9000)
    if args.memory_safe_resume:
        if not args.memory_safe_pilot:parser.error('Memory-safe resume requires --memory-safe-pilot')
        initial_text=aws('s3','cp',STEP_ZERO_MAPPER+'manifest.json','-','--only-show-errors')
        initial=json.loads(initial_text)
        if initial.get('trained_steps')!=0 or initial.get('selected_at_step')!=0 or initial.get('format_version')!=2:
            raise ValueError('Only the exact untrained step-zero snapshot may initialize this pilot')
        cfg.update(mapper_origin=STEP_ZERO_MAPPER,use_step_zero_initialization=True,
            initial_mapper_manifest_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            initial_mapper_manifest_hash_rule='canonical-json-sortkeys-compact-ascii',
            activation_offload_policy='selective_pageable',train_steps=100,train_checkpoint_every=100,train_max_seconds=1100)
    probe=None
    if args.mode=='repair':
        if args.probe_summary is None:parser.error('--probe-summary is required for repair')
        probe=verify_probe_summary(args.probe_summary,args.fresh_oracle_gate)
    continuation={}
    if args.mode=='eval_resume':
        if args.parent_receipt is None:parser.error('--parent-receipt is required for continuation')
        continuation=continuation_inputs(args.parent_receipt.resolve(strict=True),cfg)
    smoke=verify_smoke(args.mode,args.smoke_report)
    if args.compute_guard_usd==5.0 and not(args.memory_safe_resume or args.mode=='eval_resume'):
        parser.error('Expanded repair-goal guard requires explicit memory recovery or continuation')
    rate=price();estimate,total=recovery_budget(rate['usd_per_hour'],runtime,args.prior_compute_usd,args.compute_guard_usd)
    files={**source_files(args.mode),**inputs(args.mode),**continuation}
    if probe:files['inputs/prior-probe-summary.json']=args.probe_summary
    if precision:files['inputs/prior-precision-summary.json']=args.precision_summary
    cfg['input_sha256']={n:digest(p) for n,p in files.items() if n.startswith(('inputs/','reference/'))}
    job=job_name(args.mode)
    req=request(job,cfg)
    with tempfile.TemporaryDirectory(prefix='sova-recovery-')as temporary:
        temp=Path(temporary);bundle=temp/'bundle.tar.gz'
        # The shared bundler writes a full-config entry; include our separate
        # worker config explicitly and retain the exact generated archive hash.
        configuration=temp/'recovery-config.json';configuration.write_text(json.dumps(cfg,indent=2)+'\n')
        files['cloud/recovery-config.json']=configuration
        hashes=make_bundle(bundle,files,cfg)
        verify_packaged_sources(args.mode,hashes,smoke)
        receipt={'status':'prepared','job_name':job,'profile':PROFILE,'region':REGION,
            'mode':args.mode,'request':req,'config':cfg,'price':rate,'smoke_evidence':smoke,'probe_evidence':probe,'precision_evidence':precision,
            'prior_compute_usd':args.prior_compute_usd,'estimated_max_worker_compute_usd':estimate,
            'total_estimated_compute_at_runtime_limit_usd':total,'runtime_compute_guard_usd':args.compute_guard_usd,
            'cost_note':'Public-rate worker estimate; storage, transfer, billing overhead and tax are extra.',
            'bundle_sha256':digest(bundle),'file_sha256':hashes,
            'live_results':f's3://{BUCKET}/kv-transfer/{job}/live/',
            'checkpoints':f's3://{BUCKET}/kv-transfer/{job}/checkpoints/'}
        if args.submit:
            if json.loads(aws('sts','get-caller-identity'))['Account']!=ACCOUNT:raise ValueError('Wrong AWS account')
            # Refuse to launch against a missing selected artifact.
            if args.mode!='precision':
                objects=json.loads(aws('s3api','list-objects-v2','--bucket',BUCKET,'--prefix',cfg['mapper_origin'].split(BUCKET+'/',1)[1]))
                names={o['Key'].rsplit('/',1)[-1] for o in objects.get('Contents',[])}
                if not {'manifest.json','weights.safetensors'}<=names:raise ValueError('Selected input mapper artifact missing')
            report=ROOT/'reports'/job;report.mkdir(parents=True,exist_ok=False)
            saved=report/'submission.json';saved.write_text(json.dumps(receipt,indent=2)+'\n')
            path=temp/'request.json';path.write_text(json.dumps(req))
            try:
                aws('s3','cp',str(bundle),f's3://{BUCKET}/kv-transfer/{job}/input/bundle.tar.gz','--only-show-errors')
                result=json.loads(aws('sagemaker','create-training-job','--cli-input-json','file://'+str(path)))
                receipt.update(status='submitted',job_arn=result['TrainingJobArn'])
            except subprocess.CalledProcessError as exc:
                receipt.update(status='submission_failed',error=exc.stderr.strip());saved.write_text(json.dumps(receipt,indent=2)+'\n');raise
            saved.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
