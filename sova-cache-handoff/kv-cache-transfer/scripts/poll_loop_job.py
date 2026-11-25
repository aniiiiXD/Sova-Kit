"""Read-only managed-job progress; never submits or runs model work."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloud.submit import aws


def read_optional(uri):
    try:return json.loads(aws('s3','cp',uri,'-','--only-show-errors'))
    except subprocess.CalledProcessError as exc:
        if '404'in exc.stderr or 'Not Found'in exc.stderr or 'NoSuchKey'in exc.stderr:return None
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);args=p.parse_args()
    receipt=json.loads(args.receipt.read_text());name=receipt['job_name']
    state=json.loads(aws('sagemaker','describe-training-job','--training-job-name',name))
    prefix=receipt['live_results'];progress=read_optional(prefix+'progress.json');best=read_optional(prefix+'best-selection.json')
    result={'job':name,'state':state['TrainingJobStatus'],'phase':state['SecondaryStatus'],
        'runtime_seconds':state.get('TrainingTimeInSeconds'),'billable_seconds':state.get('BillableTimeInSeconds'),
        'failure':state.get('FailureReason'),'progress':progress,'best':best}
    (args.receipt.parent/'status-latest.json').write_text(json.dumps(result,indent=2)+'\n')
    compact={k:v for k,v in result.items()if k not in('progress','best')}
    if progress:compact['progress']={k:progress[k]for k in('phase','step','loss','document','total','layer','total_layers',
        'corrected_layers','mean_native_kl','correct','native_agreement','best_layer','sampled_states','candidate','tokens','selected',
        'unique_documents','total_supervised_tokens','elapsed_worker_seconds','peak_gpu_mib','host_peak_rss_mib')if k in progress}
    if best:compact['best']={k:best[k]for k in('step','mean_native_kl','correct','native_correct','native_agreement','documents')if k in best}
    print(json.dumps(compact,indent=2))
    if state['TrainingJobStatus']in('Failed','Stopped'):
        failure=read_optional(prefix+'worker-status.json')
        if failure:print(json.dumps(failure))


if __name__=='__main__':main()
