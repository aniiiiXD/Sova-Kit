"""Audit a completed development intervention; never launch compute."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.probe/'collected/summary.json').read_text())
    rows = json.loads((args.probe/'collected/rows.json').read_text())
    original = {r['id']: r for r in json.loads(args.reference.read_text())}
    if summary['status'] != 'completed' or len(rows) != 30 or len(original) != 30:
        raise ValueError('Complete thirty-article experiment required')
    if {r['id'] for r in rows} != set(original) or len({r['document_id'] for r in rows}) != 30:
        raise ValueError('Reference article selection differs')
    maximum_probability_delta = maximum_kl_delta = 0.
    for row in rows:
        prior = original[row['id']]
        current = row['variants']['translated_all']
        if row['native']['decision_answer'] != prior['native_score']['decision_answer']:
            raise ValueError('Native baseline decision changed')
        if current['score']['decision_answer'] != prior['score']['decision_answer']:
            raise ValueError('Unchanged checkpoint decision changed')
        maximum_probability_delta = max(maximum_probability_delta, max(
            abs(current['score']['option_probabilities'][k]-v)
            for k, v in prior['score']['option_probabilities'].items()))
        maximum_kl_delta = max(maximum_kl_delta, abs(
            current['distribution']['native_to_candidate_kl_nats']-
            prior['distribution']['native_to_candidate_kl_nats']))
    if maximum_probability_delta > 2e-5 or maximum_kl_delta > 1e-4:
        raise ValueError('Unchanged baseline numerical reproduction failed')
    audit = {'same_articles': 30, 'same_native_decisions': 30,
             'same_translated_decisions': 30,
             'max_option_probability_delta': maximum_probability_delta,
             'max_native_kl_delta': maximum_kl_delta,
             'passed': True}
    (args.probe/'reproduction-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    native = sum(r['native']['decision_correct'] for r in rows)
    lines = ['# Native-prefix intervention: development results', '',
             'Frozen adapter checkpoint 250; identical thirty validation articles; zero training updates.',
             'The unchanged arm reproduced all thirty prior decisions within the numerical tolerances recorded in reproduction-audit.json.', '',
             '| Arm | Correct /30 | Native agreement /30 | Mean native KL | Native prefix tokens (mean) |',
             '|---|---:|---:|---:|---:|']
    for arm, reported in summary['variants'].items():
        result = {'correct': sum(r['variants'][arm]['score']['decision_correct'] for r in rows),
                  'native_agreement': sum(r['variants'][arm]['native_agreement'] for r in rows),
                  'mean_native_kl': sum(r['variants'][arm]['distribution']['native_to_candidate_kl_nats'] for r in rows)/30,
                  'mean_native_prefix_tokens': sum(r['variants'][arm]['native_prefix_tokens'] for r in rows)/30}
        if any(abs(result[k]-reported[k]) > 1e-8 for k in result):
            raise ValueError('Summary does not match all raw article rows')
        lines.append(f'| {arm} | {result["correct"]} | {result["native_agreement"]} | {result["mean_native_kl"]:.6f} | {result["mean_native_prefix_tokens"]:.1f} |')
    lines += ['', f'Native target: {native}/30 correct. This development control uses no reserved acceptance records.', '',
              'Native-prefix arms perform partial target recomputation. Their cost must be included in any future timing comparison. This experiment makes no latency, throughput or generalization claim.', '']
    (args.probe/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'reproduction': audit, 'native_correct': native, 'variants': summary['variants']}, indent=2))


if __name__ == '__main__': main()
