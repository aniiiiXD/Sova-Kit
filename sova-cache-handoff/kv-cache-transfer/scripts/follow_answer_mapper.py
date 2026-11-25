"""Follow the existing training job through independent audit and gated TTFT."""
import argparse
import asyncio
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.answer_train_loop import describe, optional_live
from kvverify.agent_graph import Controller, Node
from kvverify.evidence import percentile
from scripts.verify_answer_mapper_run import verify_run


def terminal(job):
    previous = None
    while True:
        state = describe(job)
        status = state['TrainingJobStatus']
        snapshot = (status, state['SecondaryStatus'])
        if snapshot != previous:
            print(json.dumps({'job':job, 'state':status, 'phase':snapshot[1]}), flush=True)
            previous = snapshot
        if status in ('Completed','Stopped','Failed'):
            return state
        time.sleep(30)


def verify_timing(receipt, state, output):
    live = receipt['live_results']
    raw = {}
    for filename in ('rows.json','parity.json','summary.json'):
        payload = optional_live(live+filename)
        if payload is None:
            raise ValueError('Missing timing evidence: '+filename)
        raw[filename] = payload
        (output/filename).write_text(json.dumps(payload,indent=2)+'\n')
    config = receipt['config']
    rows, parity, summary = raw['rows.json'], raw['parity.json'], raw['summary.json']
    from cloud.ttft_worker import timing_arms
    arms = timing_arms(config['native_prefix_tokens'])
    count = config['documents'] * config['repeats']
    if (summary['weight_sha256'] != config['mapper_weights_sha256']
            or summary['native_prefix_tokens'] != config['native_prefix_tokens']
            or len(rows) != count*len(arms) or len(parity) != 2*count):
        raise ValueError('Timing evidence differs from frozen execution')
    document_ids = {row['document_id'] for row in rows}
    pairs = {(row['document_id'],row['repeat'],row['arm']) for row in rows}
    expected = {(doc,rep,arm) for doc in document_ids for rep in range(config['repeats']) for arm in arms}
    if len(document_ids) != config['documents'] or pairs != expected:
        raise ValueError('Missing paired timing samples')
    parity_pairs = {(row['id'],row['repeat'],row['path']) for row in parity}
    expected_parity = {(row['id'],row['repeat'],path) for row in rows for path in ('cold','source_hit')}
    if parity_pairs != expected_parity:
        raise ValueError('Missing reference parity samples')
    timings = {}
    for arm in arms:
        samples = [row['ttft_ms'] for row in rows if row['arm'] == arm]
        timings[arm] = {'p50':median(samples), 'p95':percentile(samples, .95)}
    for row in rows:
        if row['arm'].startswith(('fast_','reference_')) and (
                row['target_prefill_tokens'] != config['native_prefix_tokens'] + row['suffix_tokens']
                or row['imported_translated_tokens'] != row['prefix_tokens'] - config['native_prefix_tokens']):
            raise ValueError('Unaccounted native prefill or translated tokens')
    exact = all(row['passed'] and row['option_agreement'] for row in parity)
    native, handoff = timings['native_cold'], timings['fast_source_hit']
    passes = exact and handoff['p50'] <= .8*native['p50'] and handoff['p95'] <= native['p95']
    return {'verdict':'timing_screen_passed' if passes else 'timing_screen_rejected',
            'job_name':receipt['job_name'],'billable_seconds':state['BillableTimeInSeconds'],
            'timings':timings,'parity_passed':exact,'model_only':True,
            'product_goal_achieved':False,'confirmation_opened':False}


def follow(run):
    run = run.resolve()
    training = json.loads((run/'training/submission.json').read_text())
    if training['status'] != 'submitted' or training['mode'] != 'answer_train':
        raise ValueError('Expected an existing submitted answer-mapper worker')
    root = run/'follow-through'
    timing = root/'timing'
    spec = {'training_job':training['job_name'], 'bundle_sha256':training['bundle_sha256'],
            'timing_gate':{'source_hit_p50_vs_native_cold_max_ratio':.8,
                           'source_hit_p95_vs_native_cold_max_ratio':1.0}}

    def training_end(_):
        state = terminal(training['job_name'])
        code = state['TrainingJobStatus']
        failure = state.get('FailureReason')
        if code == 'Failed':
            from cloud.submit import aws
            try:
                text = aws('s3','cp',training['live_results']+'failure.txt','-','--only-show-errors')
                (root/'training-failure.txt').write_text(text+'\n')
            except subprocess.CalledProcessError:
                pass
        return {'state':code, 'failure':failure, 'billable_seconds':state.get('BillableTimeInSeconds',0)}

    def audit(inputs):
        ended = inputs['training_end']
        if ended['state'] != 'Completed':
            return {'verdict':'training_'+ended['state'].lower(), **ended}
        return verify_run(run)

    def schedule(inputs):
        verified = inputs['audit']
        if verified['verdict'] != 'ready_for_timing':
            return {'verdict':verified['verdict'], 'timing_submitted':False}
        receipt = timing/'submission.json'
        if not receipt.is_file():
            if timing.exists():
                raise RuntimeError('Timing directory has no receipt; reconcile before submission')
            subprocess.run([sys.executable,'-m','cloud.ttft_submit','--training-run',str(run),
                            '--output',str(timing),'--submit'], cwd=ROOT, check=True)
        saved = json.loads(receipt.read_text())
        config = saved['config']
        if (config['verified_training_job'] != training['job_name']
                or config['mapper_weights_sha256'] != verified['weights_sha256']
                or config['native_prefix_tokens'] != verified['native_prefix_tokens']):
            raise ValueError('Existing timing receipt is for different evidence')
        if describe(saved['job_name'], absent_ok=True) is None:
            raise RuntimeError('Saved timing intent has no AWS job; reconcile before retrying')
        return {'verdict':'timing_submitted','timing_submitted':True,'receipt':saved}

    def verify(inputs):
        result = inputs['schedule']
        if not result['timing_submitted']:
            return {**result,'product_goal_achieved':False,'confirmation_opened':False}
        saved = result['receipt']
        state = terminal(saved['job_name'])
        if state['TrainingJobStatus'] != 'Completed':
            return {'verdict':'timing_worker_'+state['TrainingJobStatus'].lower(),
                    'job_name':saved['job_name'],'product_goal_achieved':False,'confirmation_opened':False}
        return verify_timing(saved, state, timing)

    graph = [Node('training_end',(),training_end,max_attempts=3),
             Node('audit',('training_end',),audit,max_attempts=3),
             Node('schedule',('audit',),schedule,max_attempts=3),
             Node('verify',('schedule',),verify,max_attempts=3)]
    result = asyncio.run(Controller(root/'graph.sqlite3',graph,spec).run())['verify']
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    follow(parser.parse_args().run)
