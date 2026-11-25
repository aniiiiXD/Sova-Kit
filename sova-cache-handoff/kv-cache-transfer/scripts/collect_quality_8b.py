"""Collect final SageMaker QuALITY scores and compare paired document results."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.submit import aws
from kvverify.evidence import paired_quality


LABELS = {'native': '8B native', 'source': '1.7B native',
          'translated': '8B with translated cache',
          'translated_native256': '8B with translated cache and 256 native tokens',
          'native256_only': '8B with first 256 document tokens only'}


def summarize(receipt_dir, receipt, state, results):
    staged = results['summary.json']
    rows = results['rows.json']
    timings = results['timings.json']
    dataset = ROOT / 'datasets/loop-holdout-v1/quality.jsonl'
    examples = {row['id']: row for row in map(json.loads, dataset.read_text().splitlines())}
    if (staged['benchmark'] != 'quality' or staged['status'] != 'completed' or
            len(rows) != 267 or len(timings) != 15 or {row['id'] for row in rows} != set(examples)):
        raise ValueError('Incomplete or mismatched frozen QuALITY run')
    if any((row['document_id'], row['gold']) !=
           (examples[row['id']]['document_id'], examples[row['id']]['answer']) for row in rows):
        raise ValueError('Scored row does not match the frozen document or gold answer')
    hard = [row for row in rows if examples[row['id']]['difficult']]
    n = len(rows)
    masks = {'native': [row['native']['decision_correct'] for row in rows]}
    paired = {}
    for name in LABELS.keys() - {'native'}:
        masks[name] = [row['variants'][name]['score']['decision_correct'] for row in rows]
        trace = []
        for row, reference, candidate in zip(rows, masks['native'], masks[name]):
            key = row['id']
            identity = {'id': key, 'document_id': row['document_id'], 'repeat': 0,
                        'context_hash': examples[key]['context_sha256_normalized'],
                        'prompt_hash': key, 'run_config_hash': receipt['bundle_sha256']}
            trace.extend([{**identity, 'arm': 'native_cold', 'correct': reference},
                          {**identity, 'arm': 'transfer_cold', 'correct': candidate}])
        paired[name] = paired_quality(trace, iterations=10000)
    if sum(masks['native']) != staged['native_correct']:
        raise ValueError('Native summary differs from raw scores')
    for name, item in staged['variants'].items():
        if sum(masks[name]) != item['correct']:
            raise ValueError('Variant summary differs from raw scores: ' + name)
    toracle = results['prompt-lengths.json']
    cold = median(item['native_cold_ms'] for item in timings)
    translated = median(item['source_prefill_ms'] + item['conversion_ms'] +
                        item['translated_query_ms'] for item in timings)
    source_cached = median(item['conversion_ms'] + item['translated_query_ms'] for item in timings)
    native_cached = median(item['native_cache_hit_query_ms'] for item in timings)
    compute = state['BillableTimeInSeconds'] * receipt['budget_ledger']['price']['usd_per_hour'] / 3600
    cumulative = compute + receipt['budget_ledger']['runtime_compute_usd']
    converted = {'scores': {}, 'paired_vs_8b': paired, 'native_cold_ms': cold,
                 'transfer_cold_ms': translated, 'source_cache_hit_ms': source_cached,
                 'native_cache_hit_ms': native_cached, 'job_compute_usd': compute,
                 'estimated_cumulative_compute_usd': cumulative}
    lines = ['# Qwen3 1.7B → 8B on QuALITY', '',
             f"Evaluated all {n} reserved questions over 15 articles ({min(toracle.values()):,}–{max(toracle.values()):,} prompt tokens). "
             f"The adapter was fitted on 32 separate official training articles. Models were frozen, Qwen thinking was disabled, and scores use the most likely option letter.",
             '', f'| Path | All {n} | Official hard subset ({len(hard)}) | Agreement with 8B |',
             '|---|---:|---:|---:|']
    for name, label in LABELS.items():
        score = sum(masks[name])
        if name == 'native':
            num_hard = sum(row['native']['decision_correct'] for row in hard)
            agreement = n
        else:
            num_hard = sum(row['variants'][name]['score']['decision_correct'] for row in hard)
            agreement = staged['variants'][name]['native_agreement']
        converted['scores'][name] = {'correct': score, 'hard_correct': num_hard,
                                    'native_agreement': agreement}
        lines.append(f'| {label} | {score}/{n} ({score/n:.1%}) | '
                     f'{num_hard}/{len(hard)} ({num_hard/len(hard):.1%}) | {agreement}/{n} |')
    lines.extend(['', 'The accuracy differences below are in percentage points relative to native 8B. '
                  'Intervals resample the 15 articles, keeping each article’s questions together.', ''])
    for name in ('source', 'translated', 'translated_native256'):
        item = paired[name]
        lo, hi = item['ci95']
        lines.append(f"- {LABELS[name]} {100*item['difference']:+.1f} pp "
                     f"(95% article-bootstrap interval {100*lo:+.1f} to {100*hi:+.1f} pp).")
    lines.extend(['', f'Median model-only TTFT was {cold:.0f} ms for native 8B and {translated:.0f} ms for cold transfer. '
                  f'Transfer from an existing 1.7B cache took {source_cached:.0f} ms and an existing native 8B cache took {native_cached:.0f} ms. '
                  'One timing sample was collected per article on a single L40S. Cold transfer includes source prefill, conversion, and the target query.',
                  '', f"Peak allocated GPU memory was {staged['peak_gpu_mib']/1024:.1f} GiB. "
                  f"The job completed in {state['BillableTimeInSeconds']} billable seconds at an estimated "
                  f"${compute:.2f} compute cost. Estimated project GPU compute through this run was ${cumulative:.2f}.",
                  '', 'The questions come from the official QuALITY development partition. This is a reserved slice '
                  'with this experiment’s scoring and mapper, so reported vendor scores are separate.',
                  '', '[Official QuALITY dataset](https://github.com/nyu-mll/quality) · '
                  '[Row scores](collected/rows.json) · [Timing samples](collected/timings.json)'])
    (receipt_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.run / 'submission.json').read_text())
    state = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', receipt['job_name']))
    if state['TrainingJobStatus'] != 'Completed':
        raise RuntimeError(f"Job {state['TrainingJobStatus']}: {state.get('FailureReason', state['SecondaryStatus'])}")
    out = args.run / 'collected'
    out.mkdir(exist_ok=True)
    results = {}
    for name in ('summary.json', 'rows.json', 'timings.json', 'prompt-lengths.json', 'environment.json'):
        data = aws('s3', 'cp', receipt['live_results'] + name, '-', '--only-show-errors')
        results[name] = json.loads(data)
        (out / name).write_text(data + '\n')
    report = summarize(args.run, receipt, state, results)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
