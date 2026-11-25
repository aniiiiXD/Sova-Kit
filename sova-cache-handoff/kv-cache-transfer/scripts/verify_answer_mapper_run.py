"""Independently check a completed answer-mapper graph run from item-level evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.answer_train_worker import development_gate, development_rows
from cloud.submit import aws
from kvverify.data import load_examples


def check_checkpoint(raw, expected, aggregate):
    if len(raw) != len(expected) or {row['id'] for row in raw} != set(expected):
        raise ValueError('Validation IDs differ from the predeclared panel')

    def answer(score, example):
        values = score['option_logits']
        if set(values) != set(example.choices) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Invalid logged option logits')
        choice = max(sorted(values), key=values.__getitem__)
        if choice != score['decision_answer'] or (choice == example.answer) != score['decision_correct']:
            raise ValueError('Logged correctness differs from score and gold label')
        return choice

    counts = dict.fromkeys(('correct','native_correct','source_correct','prefix_only_correct','native_agreement'),0)
    for row in raw:
        example = expected[row['id']]
        if row['document_id'] != example.document_id or not math.isfinite(row['kl']) or row['kl'] < 0:
            raise ValueError('Invalid document identity or logged KL')
        native = answer(row['native'], example)
        choice = answer(row['candidate'], example)
        counts['native_correct'] += native == example.answer
        counts['correct'] += choice == example.answer
        counts['source_correct'] += answer(row['source'], example) == example.answer
        counts['prefix_only_correct'] += answer(row['prefix_only'], example) == example.answer
        counts['native_agreement'] += native == choice
    result = {'step':aggregate['step'],'questions':len(raw),
              'documents':len({row['document_id'] for row in raw}), **counts,
              'mean_native_kl':sum(row['kl'] for row in raw)/len(raw)}
    result['passes_development_gate'] = development_gate(result)
    if result != aggregate:
        raise ValueError('Worker aggregate differs from raw rows')
    return result


def verify_run(run):
    run = Path(run)
    receipt = json.loads((run/'training/submission.json').read_text())
    state = json.loads(aws('sagemaker','describe-training-job','--training-job-name',receipt['job_name']))
    if state['TrainingJobStatus'] != 'Completed':
        raise ValueError(f"Expected completed worker, got {state['TrainingJobStatus']}")
    out = run/'collected'
    out.mkdir(exist_ok=True)
    source = ROOT/'datasets/loop-holdout-v1/quality.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt['config']['input_sha256']['inputs/development.jsonl']:
        raise ValueError('Development records changed')
    expected = {item.id:item for item in development_rows(load_examples(source,'eval'), receipt['config']['seed'])}

    def fetch(name):
        data = aws('s3','cp',receipt['live_results']+name,'-','--only-show-errors')
        path = out/name
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(data+'\n')
        return json.loads(data)

    summary = fetch('summary.json')
    fetch('protocol.json')
    preflight = fetch('memory-preflight.json')
    checkpoints = [check_checkpoint(fetch(f"validation/step-{row['step']:06d}.json"), expected, row)
                   for row in summary['validation_history']]
    if checkpoints[0]['step'] != 0 or checkpoints[-1]['step'] != summary['completed_updates']:
        raise ValueError('Missing baseline or terminal validation')
    selected = max(checkpoints, key=lambda row:(row['passes_development_gate'],row['correct'],
                                               row['native_agreement'],-row['mean_native_kl']))
    if selected != summary['selected']:
        raise ValueError('Selected checkpoint does not match frozen selection rule')
    manifest = fetch('best/manifest.json')
    if (manifest['weights_sha256'] != summary['weights_sha256']
            or manifest['selected_at_step'] != selected['step']
            or manifest['initial_weights_sha256'] != receipt['config']['initial_weights_sha256']
            or manifest['native_prefix_policy'] != str(receipt['config']['native_prefix_tokens'])):
        raise ValueError('Exported manifest does not match selected checkpoint')
    cost = state['BillableTimeInSeconds']*receipt['budget_ledger']['price']['usd_per_hour']/3600
    verified = {'selected':selected,'checkpoints':checkpoints,'memory_preflight':preflight,'compute_usd':cost,
                'completed_updates':summary['completed_updates'],'requested_updates':summary['requested_updates'],
                'confirmation_opened':False,'product_goal_achieved':False,
                'job_name': receipt['job_name'], 'weights_sha256': summary['weights_sha256'],
                'native_prefix_tokens': receipt['config']['native_prefix_tokens'],
                'verdict':'ready_for_timing' if selected['passes_development_gate'] else 'development_rejected'}
    (run/'verified.json').write_text(json.dumps(verified,indent=2)+'\n')
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run), indent=2))


if __name__ == '__main__':
    main()
