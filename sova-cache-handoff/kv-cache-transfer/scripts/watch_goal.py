"""Read-only, bounded live-status loop; never launch or retry cloud compute."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=45)
    parser.add_argument('--max-polls',type=int,default=90)
    args=parser.parse_args()
    if not 10<=args.interval<=55 or not 1<=args.max_polls<=120:parser.error('Bounded polling interval/count required')
    output=args.receipt.parent/'goal-status-history.jsonl'
    for index in range(args.max_polls):
        try:
            proc=subprocess.run([sys.executable,str(ROOT/'cloud/full_collect.py'),'--receipt',str(args.receipt),
                '--status-only','--progress'],check=True,capture_output=True,text=True,timeout=45)
            data=json.loads(proc.stdout);progress=data.get('live_progress',{})
            row={'checked_at':datetime.now(timezone.utc).isoformat(),'poll':index+1,
                **{k:data.get(k)for k in('job_name','status','secondary_status','failure_reason','billable_seconds','estimated_billed_compute_usd')},
                'progress':{k:v for k,v in progress.items()if k in('phase','step','completed_updates','loss','validation_loss',
                    'best_validation_loss','processed_items','planned_items','elapsed_seconds','elapsed_worker_seconds',
                    'host_peak_rss_mib','gpu_memory','status','full_prompt_tokens','gradients_finite')}}
        except(subprocess.SubprocessError,json.JSONDecodeError)as exc:
            row={'checked_at':datetime.now(timezone.utc).isoformat(),'poll':index+1,'monitor_error':str(exc)}
        with output.open('a')as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if row.get('status')in('Completed','Failed','Stopped'):return
        if index+1<args.max_polls:time.sleep(args.interval)


if __name__=='__main__':main()
