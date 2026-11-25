"""Prospective larger evaluation, with decision and free-generation scores separate."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import time

from .data import Example, json_digest
from .evidence import ARMS, paired_quality, percentile, validate_trace
from .quality import generation_metrics, option_token_ids, score_option_logits


VARIANTS = ('source', 'native', 'ridge', 'neural')


def install_observer(runtime):
    cached = {}
    def observe(logits, example, answer, generated, eos, maximum):
        key = tuple(sorted(example.choices))
        if key not in cached:
            cached[key] = option_token_ids(runtime.tokenizer, example.choices)
        finish = 'eos' if generated[-1] in eos else 'length' if len(generated) == maximum else 'unknown'
        return {**score_option_logits(logits, cached[key], example.answer),
                **generation_metrics(answer, example.answer, example.choices, finish_reason=finish)}
    runtime.quality_observer = observe


def run_quality(runtime, examples, mappers, output, *, freegen_documents=61,
                freegen_max_new_tokens=16, seed=19, callback=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    install_observer(runtime)
    rng = random.Random(seed)
    config = {'scoring': 'prospective-v2', 'seed': seed, 'examples': json_digest([asdict(e) for e in examples]),
        'variants': {v: mappers[v].manifest if v in mappers else None for v in VARIANTS},
        'decision_primary': 'Argmax of first-token logits restricted to exact A/B/C tokens; no generation or correctness claim for other tasks',
        'free_generation': 'Separate unconstrained greedy generation; exact-letter and label-prefix diagnostic reported separately',
        'freegen_max_new_tokens': freegen_max_new_tokens,
        'prompt_token_limit': runtime.prompt_token_limit,
        'runtime': runtime.identities, 'old_pilot_scores_unchanged': True}
    config_hash = json_digest(config)
    (root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    # Warmup is explicit; never included in quality or latency summaries.
    eligible = next((e for e in examples if len(runtime.tokenize(e)) <= runtime.prompt_token_limit), None)
    if eligible is None:
        raise ValueError('No eligible evaluation prompt')
    with (root/'warmup.jsonl').open('w') as log:
        for variant in VARIANTS:
            if variant in mappers:
                runtime.mapper = mappers[variant]
            arm = 'source' if variant == 'source' else 'native_cold' if variant == 'native' else 'transfer_cold'
            row = runtime.answer(eligible, arm, 0, config_hash, max_new_tokens=1)
            validate_trace(row)
            log.write(json.dumps({**row, 'variant': variant})+'\n')
    rows = []
    bydoc = {}
    for example in examples:
        bydoc.setdefault(example.document_id, []).append(example)
    # One seeded question per document, selected before any model outputs.
    free_examples = [rng.choice(bydoc[d]) for d in sorted(bydoc)[:freegen_documents]]
    (root/'freegen-selection.json').write_text(json.dumps([e.id for e in free_examples], indent=2)+'\n')
    for phase, selected, budget in [('decision', examples, 1),
                                    ('free_generation', free_examples, freegen_max_new_tokens)]:
        with (root/(phase+'.jsonl')).open('w') as log:
            for index, example in enumerate(selected):
                order = list(VARIANTS)
                rng.shuffle(order)
                for variant in order:
                    if variant in mappers:
                        runtime.mapper = mappers[variant]
                    arm = 'source' if variant == 'source' else 'native_cold' if variant == 'native' else 'transfer_cold'
                    try:
                        row = runtime.answer(example, arm, 0, config_hash, max_new_tokens=budget)
                        validate_trace(row)
                    except Exception as exc:
                        row = {'id': example.id, 'document_id': example.document_id,
                            'context_hash': example.context_hash, 'prompt_hash': json_digest(runtime.tokenize(example)),
                            'run_config_hash': config_hash, 'repeat': 0, 'arm': arm, 'evidence_kind': 'hf_model',
                            'status': 'error', 'correct': False, 'fallback': False,
                            'error_type': type(exc).__name__, 'error': str(exc)}
                    row.update(variant=variant, phase=phase, gold_answer=example.answer)
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    log.flush()
                if (index+1) % 50 == 0 or index+1 == len(selected):
                    print(f'quality phase={phase} examples={index+1}/{len(selected)}', flush=True)
                    if callback:
                        callback({'phase': phase, 'completed': index+1, 'total': len(selected)})
    result = summarize_quality(rows)
    (root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def summarize_quality(rows):
    seen = set()
    grouped = {}
    for row in rows:
        if row.get('phase') not in ('decision', 'free_generation') or row.get('variant') not in VARIANTS:
            raise ValueError('Unknown quality phase or variant')
        if row.get('status') not in ('ok', 'error', 'over_limit') or row.get('repeat') != 0:
            raise ValueError('Quality requires one repeat-zero attempt with a known outcome per example/variant')
        if (not isinstance(row.get('gold_answer'), str) or len(row['gold_answer']) != 1
                or row['gold_answer'] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            raise ValueError('Quality rows must retain a gold option letter for post-generation scoring')
        key = (row['phase'], row['variant'], row['id'])
        if key in seen:
            raise ValueError('Repeated quality examples must not inflate the denominator')
        seen.add(key)
        grouped.setdefault(row['phase'], {}).setdefault(row['variant'], {})[row['id']] = row
    for phase, variants in grouped.items():
        expected = set(variants.get('native', {}))
        if not expected or set(variants) != set(VARIANTS) or any(set(v) != expected for v in variants.values()):
            raise ValueError('Every quality variant must retain the same example/error/over-limit denominator')
        for example_id in expected:
            reference = variants['native'][example_id]
            for variant in VARIANTS:
                if any(variants[variant][example_id][key] != reference[key] for key in
                       ('document_id', 'context_hash', 'prompt_hash', 'run_config_hash', 'gold_answer')):
                    raise ValueError('Quality variants have mismatched evidence or gold labels')
    def correct(row, field):
        return row['status'] == 'ok' and row.get(field) is True

    result = {'acceptance': 'not_established', 'cost_savings': 'not_measured',
              'score_definitions': 'Decision accuracy is restricted first-token choice scoring; free-generation strict accuracy remains separate.',
              'macro_definition': 'Unweighted average of per-class accuracy over observed gold classes; errors and over-limit attempts count as wrong.',
              'phases': {}}
    for phase in ('decision', 'free_generation'):
        result['phases'][phase] = {}
        for variant in VARIANTS:
            selected = [r for r in rows if r['phase'] == phase and r['variant'] == variant]
            n = len(selected)
            okay = [r for r in selected if r['status'] == 'ok']
            labels = sorted({r['gold_answer'] for r in selected})
            per_class = {}
            for label in labels:
                class_rows = [r for r in selected if r['gold_answer'] == label]
                count = len(class_rows)
                per_class[label] = {
                    'examples': count,
                    'returned': sum(r['status'] == 'ok' for r in class_rows),
                    'errors': sum(r['status'] == 'error' for r in class_rows),
                    'over_limit': sum(r['status'] == 'over_limit' for r in class_rows),
                    'decision_accuracy': sum(correct(r, 'decision_correct') for r in class_rows)/count,
                    'free_generation_strict_accuracy': sum(correct(r, 'free_generation_strict_correct') for r in class_rows)/count if phase == 'free_generation' else None,
                }
            value = {'examples': n, 'returned': len(okay), 'errors': sum(r['status'] == 'error' for r in selected),
                'over_limit': sum(r['status'] == 'over_limit' for r in selected),
                'decision_accuracy': sum(correct(r, 'decision_correct') for r in selected)/n if n else None,
                'free_generation_strict_accuracy': sum(correct(r, 'free_generation_strict_correct') for r in selected)/n if n and phase == 'free_generation' else None,
                'format_valid_count': sum(correct(r, 'free_generation_format_valid') for r in selected) if phase == 'free_generation' else None,
                'truncated_count': sum(correct(r, 'free_generation_truncated') for r in selected) if phase == 'free_generation' else None,
                'per_class': per_class,
                'label_distribution': {label: spec['examples'] for label, spec in per_class.items()},
                'majority_class_baseline_accuracy': max((spec['examples'] for spec in per_class.values()), default=0)/n if n else None,
                'macro_decision_accuracy': statistics.mean(spec['decision_accuracy'] for spec in per_class.values()) if per_class else None,
                'macro_free_generation_strict_accuracy': statistics.mean(spec['free_generation_strict_accuracy'] for spec in per_class.values()) if per_class and phase == 'free_generation' else None,
                'p50_ttft_ms': statistics.median([r['ttft_ms'] for r in okay]) if okay else None,
                'p95_ttft_ms': percentile([r['ttft_ms'] for r in okay], .95),
                'option_probability_mass_median': statistics.median([r['option_probability_mass'] for r in okay]) if okay else None}
            result['phases'][phase][variant] = value
    for variant in ('ridge', 'neural'):
        paired = []
        for row in rows:
            if row['phase'] != 'decision' or row['variant'] not in ('native', variant):
                continue
            paired.append({**row, 'arm': 'native_cold' if row['variant'] == 'native' else 'transfer_cold',
                           'correct': correct(row, 'decision_correct')})
        result['paired_decision_'+variant] = paired_quality(paired)
    return result


def run_latency(runtime, examples, output, *, example_count=24, repeats=5, seed=19, callback=None):
    from .runner import evaluate
    rng = random.Random(seed)
    # Frozen length-stratified sample, independent of correctness outcomes.
    eligible = sorted((e for e in examples if len(runtime.tokenize(e)) <= runtime.prompt_token_limit),
                      key=lambda e: (len(runtime.tokenize(e)), e.id))
    if not eligible:
        raise ValueError('No eligible latency examples')
    count = min(example_count, len(eligible))
    selected = [eligible[round(i*(len(eligible)-1)/max(1,count-1))] for i in range(count)]
    result = evaluate(runtime, selected, output, repeats=repeats, warmups=1, max_new_tokens=1, seed=seed)
    if callback:
        callback({'phase': 'latency', 'completed': count, 'repeats': repeats})
    return result


def run_long_latency(runtime, output, *, lengths=(8192,16384,32768), repeats=10,
                     max_seconds=900, callback=None):
    """Synthetic length controls are timing evidence only, never benchmark quality."""
    from .runner import evaluate
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    prior = runtime.prompt_token_limit
    started = time.monotonic()
    records = []
    try:
        runtime.prompt_token_limit = min(runtime.source.config.max_position_embeddings,
                                         runtime.target.config.max_position_embeddings)-1
        for length in lengths:
            if time.monotonic()-started >= max_seconds:
                records.append({'requested_tokens': length, 'status': 'not_run_time_budget'})
                continue
            filler = runtime.tokenizer('This is a background record without additional facts. ', add_special_tokens=False)['input_ids']
            count = max(1, (length-256)//len(filler))
            text = runtime.tokenizer.decode(filler*count, skip_special_tokens=True)
            example = Example(f'synthetic-timing-{length}', f'synthetic-timing-document-{length}',
                text+'\nThe access code is cobalt.', 'What is the access code?',
                {'A':'cobalt', 'B':'amber', 'C':'silver'}, 'A', 'eval')
            actual = len(runtime.tokenize(example))
            if actual+1 > runtime.prompt_token_limit:
                records.append({'requested_tokens': length, 'actual_tokens': actual, 'status': 'not_run_native_limit'})
                continue
            summary = evaluate(runtime, [example], root/f'tokens-{length}', repeats=repeats,
                               warmups=1, max_new_tokens=1, seed=19)
            records.append({'requested_tokens': length, 'actual_tokens': actual,
                            'status': 'completed', 'summary': summary,
                            'scope': 'Synthetic timing only; quality and generalization not established'})
            if callback:
                callback({'phase': 'long_latency', 'requested_tokens': length, 'actual_tokens': actual})
        (root/'lengths.json').write_text(json.dumps(records, indent=2)+'\n')
    finally:
        runtime.prompt_token_limit = prior
    return records
