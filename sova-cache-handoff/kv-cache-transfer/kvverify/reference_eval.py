"""Frozen QuALITY decision/generation and HellaSwag ending-likelihood tests.

No downloads, model loading or cloud submission. Call from an approved worker
with DenseRuntime and a validated mapper. QuALITY is our named custom protocol;
HellaSwag follows the pinned task's preprocessing and character normalization.
Neither is a claim to reproduce a private vendor's prompts or selected items.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

from .data import Example, check_heldout, json_digest
from .evidence import paired_quality
from .quality import generation_metrics, option_token_ids, score_option_logits

VARIANTS = ('source', 'native', 'mapper')
HARNESS_REVISION = 'd6de81643928d653435c431bae19945d41d32520'
QUALITY_PROTOCOL = 'sova-quality-doc-cache-native-query-choice-and-freegen-v1'
HELLASWAG_PROTOCOL = 'sova-hellaswag-zero-shot-ending-likelihood-char-norm-v1'
QUALITY_FILE = 'processed/quality/dev-100.jsonl'
TRAIN_FILE = 'processed/quality/train.jsonl'
HELLASWAG_FILE = 'processed/hellaswag/validation-1000.jsonl'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hellaswag_preprocess(text):
    """Pinned task/utils.py semantics; do not execute downloaded task source."""
    return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')


def validate_hellaswag_record(row):
    if row.get('official_split') != 'validation' or not isinstance(row.get('benchmark_id'), str):
        raise ValueError('Expected a frozen official HellaSwag validation record')
    if not isinstance(row.get('source_id'), str) or not row['source_id']:
        raise ValueError('HellaSwag source-group ID is required')
    choices = row.get('choices')
    if not isinstance(choices, list) or len(choices) != 4 or any(not isinstance(x, str) or not x for x in choices):
        raise ValueError('Exactly four nonempty processed endings are required')
    query = hellaswag_preprocess(row['activity_label']+': '+row['ctx_a']+' '+row['ctx_b'].capitalize())
    expected = [hellaswag_preprocess(x) for x in row['endings']]
    if row.get('query') != query or choices != expected:
        raise ValueError('Frozen HellaSwag preprocessing differs from pinned task')
    if type(row.get('gold_index')) is not int or not 0 <= row['gold_index'] < 4 or int(row['label']) != row['gold_index']:
        raise ValueError('Invalid ending index')
    if row.get('completion_char_lengths') != [len(x) for x in choices]:
        raise ValueError('acc_norm must use processed ending character lengths, excluding the delimiter')


def load_reference_data(directory, manifest_sha256, *, expected_quality=100, expected_hellaswag=1000):
    """Verify only the three frozen files the worker bundles, never raw/test data.

    Returns (QuALITY eval Example list, HellaSwag record list, original manifest).
    The train file is hash/count-checked for the worker's separate train-only
    partition, but official dev examples are never returned as training records.
    """
    root = Path(directory).resolve()
    manifest_path = root/'manifest.json'
    if _sha(manifest_path) != manifest_sha256:
        raise ValueError('Reference manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('official_test_accessed') is not False:
        raise ValueError('Unknown reference manifest or official test access')
    if manifest['upstream_revisions'].get('lm_evaluation_harness_github') != HARNESS_REVISION:
        raise ValueError('Unexpected HellaSwag protocol source revision')
    for relative in (QUALITY_FILE, TRAIN_FILE, HELLASWAG_FILE):
        path = root/relative
        spec = manifest['processed'][relative]
        if not path.is_file() or _sha(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('Frozen reference file mismatch: '+relative)
        with path.open() as stream:
            count = sum(bool(line.strip()) for line in stream)
        if count != spec['examples']:
            raise ValueError('Reference record count mismatch: '+relative)
    quality_rows = [json.loads(x) for x in (root/QUALITY_FILE).read_text().splitlines() if x.strip()]
    hellaswag = [json.loads(x) for x in (root/HELLASWAG_FILE).read_text().splitlines() if x.strip()]
    quality = []
    for row in quality_rows:
        if row.get('official_split') != 'dev' or row.get('split') != 'eval' or set(row['choices']) != set('ABCD'):
            raise ValueError('QuALITY reference requires official dev four-option eval records')
        quality.append(Example.parse(row))
    if (len(quality) != expected_quality or len({e.id for e in quality}) != len(quality)
            or len(hellaswag) != expected_hellaswag
            or len({e['benchmark_id'] for e in hellaswag}) != len(hellaswag)):
        raise ValueError('Frozen reference count or unique-ID protocol mismatch')
    for row in hellaswag:
        validate_hellaswag_record(row)
    return quality, hellaswag, manifest


def encode_pair(tokenizer, context, continuation):
    """Causal pair encoding matching pinned TemplateLM._encode_pair semantics.

    Move trailing context whitespace into the continuation, encode the joined
    string, then take continuation tokens after the encoded context length.
    Prefix instability is rejected rather than silently scoring a different
    token sequence. Leading delimiter space belongs to the scored continuation.
    """
    if not isinstance(context, str) or not context or not isinstance(continuation, str) or not continuation:
        raise ValueError('Nonempty raw context and continuation required')
    trimmed = context.rstrip()
    continuation = context[len(trimmed):]+continuation
    context = trimmed
    if not context:
        raise ValueError('Whitespace-only context requires an explicit prefix-token protocol')
    prefix = tokenizer(context, add_special_tokens=False)['input_ids']
    whole = tokenizer(context+continuation, add_special_tokens=False)['input_ids']
    if not prefix or whole[:len(prefix)] != prefix:
        raise ValueError('Context tokenization changes at ending boundary; no silent prefix repair')
    ending = whole[len(prefix):]
    if not ending:
        raise ValueError('Ending encodes to no continuation tokens')
    return prefix, ending


def score_endings(loglikelihoods, choices, gold):
    if (len(loglikelihoods) != 4 or len(choices) != 4 or type(gold) is not int or not 0 <= gold < 4
            or any(not isinstance(x, str) or not x for x in choices)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in loglikelihoods)):
        raise ValueError('Four finite ending scores and a valid index are required')
    lengths = [len(x) for x in choices]
    normalized = [score/length for score, length in zip(loglikelihoods, lengths)]
    answer = max(range(4), key=lambda i: loglikelihoods[i])
    answer_norm = max(range(4), key=lambda i: normalized[i])
    return {'choice_loglikelihoods': list(loglikelihoods), 'choice_character_lengths': lengths,
            'choice_normalized_loglikelihoods': normalized, 'prediction': answer,
            'prediction_norm': answer_norm, 'acc': answer == gold, 'acc_norm': answer_norm == gold,
            'normalization': 'Python len(processed ending string), not tokens or bytes; delimiter excluded'}


def _validate_mapper(runtime, mapper, examples):
    for key, value in [('source_sha256', runtime.identities['source']['sha256']),
                       ('target_sha256', runtime.identities['target']['sha256']),
                       ('tokenizer_sha256', runtime.tokenizer_sha256)]:
        if mapper.manifest.get(key) != value:
            raise ValueError('Reference mapper identity mismatch: '+key)
    check_heldout(mapper.manifest, examples)


def _map_cache(runtime, mapper, source_cache, expected_length):
    from .hf_backend import legacy
    cache = mapper.apply(source_cache, runtime.source, runtime.target)
    key_policy=mapper.manifest.get('key_projection','none')
    if key_policy=='target_rms_ellipsoid':
        from .key_geometry import project_target_cache
        cache=project_target_cache(cache,runtime.target)
    elif key_policy!='none':raise ValueError('Unsupported frozen key projection policy')
    rows = legacy(cache)
    heads, dim, layers = runtime.target_geometry
    expected = (1, heads, expected_length, dim)
    if (cache.get_seq_length() != expected_length or len(rows) != layers
            or any(k.shape != expected or v.shape != expected for k, v in rows)):
        raise ValueError('Mapped reference cache geometry/import length failed')
    return cache


def _prepare(runtime, mapper, variant, full_ids, prefix_length):
    """Cold execution through next-token selection; includes mapping/import."""
    if variant not in VARIANTS or not 1 <= prefix_length < len(full_ids):
        raise ValueError('Invalid reference variant or transfer boundary')
    for device in {next(m.parameters()).device for m in (runtime.source, runtime.target)}:
        if device.type == 'cuda':
            runtime.torch.cuda.reset_peak_memory_stats(device)
    runtime.sync()
    start = time.perf_counter()
    source_ms = map_ms = prefix_ms = 0.
    native_prefix_tokens=0;fallback=False
    if variant=='mapper':
        policy=mapper.manifest.get('native_prefix_policy','none')
        if policy not in('none','256'):raise ValueError('Unsupported explicit reference prefix policy')
        if policy=='256':
            fallback=prefix_length<=256
            native_prefix_tokens=0 if fallback else 256
    if variant == 'mapper'and not fallback:
        source, source_ms = runtime.timed(lambda: runtime.forward(runtime.source, full_ids[:prefix_length]).past_key_values)
        with runtime.torch.inference_mode():
            cache, map_ms = runtime.timed(lambda: _map_cache(runtime, mapper, source, prefix_length))
        if native_prefix_tokens:
            from .native_prefix import replace_prefix
            def native_initial():
                initial=runtime.forward(runtime.target,full_ids[:native_prefix_tokens]).past_key_values
                return replace_prefix(cache,initial,native_prefix_tokens)
            cache,prefix_ms=runtime.timed(native_initial)
        suffix, model = full_ids[prefix_length:], runtime.target
        del source
    else:
        suffix, cache = full_ids, None
        model = runtime.source if variant == 'source' else runtime.target
    def first():
        out = runtime.forward(model, suffix, cache)
        return out.past_key_values, out.logits[0, -1], int(out.logits[0, -1].argmax().item())
    (cache, logits, token), native_ms = runtime.timed(first)
    runtime.sync()
    elapsed = (time.perf_counter()-start)*1000
    return cache, logits, token, {
        'cold_first_token_ms': elapsed, 'source_prefill_ms': source_ms, 'map_ms': map_ms,
        'native_suffix_ms': native_ms, 'native_prefix_recompute_ms':prefix_ms,
        'imported_tokens':prefix_length-native_prefix_tokens if variant=='mapper'and not fallback else 0,
        'native_prefix_tokens':native_prefix_tokens,'native_prefix_fraction':native_prefix_tokens/prefix_length,
        'source_prefill_tokens':prefix_length if variant=='mapper'and not fallback else len(full_ids)if variant=='source'else 0,
        'target_prefill_tokens':len(suffix)+native_prefix_tokens if variant!='source'else 0,
        'cache_import_verified':variant=='mapper'and not fallback,'map_includes_move_and_import':variant=='mapper'and not fallback,
        'transfer_used':variant=='mapper'and not fallback,'fallback':fallback,
        'fallback_reason':'Prefix at most256tokens: execute full native target; not evidence of cache transfer'if fallback else None,
        'measurement': 'Synchronized batch-one HF execution; not HTTP latency or production throughput',
    }


def _all_logits(runtime, model, ids, cache):
    torch = runtime.torch
    device = next(model.parameters()).device
    offset = cache.get_seq_length()
    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids], device=device), past_key_values=cache,
            use_cache=True, attention_mask=torch.ones((1, offset+len(ids)), dtype=torch.long, device=device),
            position_ids=torch.arange(offset, offset+len(ids), device=device)[None, :],
            cache_position=torch.arange(offset, offset+len(ids), device=device),
            logits_to_keep=0, return_dict=True).logits[0]


def _ending_ll(runtime, model, cache, first_logits, ending):
    """Every continuation token is scored once; the context is never scored."""
    torch = runtime.torch
    with torch.inference_mode():
        first = torch.log_softmax(first_logits.float(), dim=-1)[ending[0]]
        if len(ending) == 1:
            return float(first.item())
        branch = runtime.clone_cache(cache)
        logits = _all_logits(runtime, model, ending[:-1], branch).float()
        targets = torch.tensor(ending[1:], dtype=torch.long, device=logits.device)
        later = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).sum()
        value = float((first+later).item())
        if not math.isfinite(value):
            raise ValueError('Nonfinite ending likelihood')
        return value


def _new_run(output, protocol, config):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    frozen = {'protocol': protocol, **config, 'acceptance': 'not_established'}
    frozen['run_config_hash'] = json_digest(frozen)
    (root/'config.json').write_text(json.dumps(frozen, indent=2)+'\n')
    return root, frozen


def _summary(rows, metrics):
    result = {'acceptance': 'not_established', 'variants': {}, 'paired': {},
              'denominator_policy': 'All frozen selected items; errors, over-limit and budget-not-run count as incorrect',
              'metric_note': 'QuALITY uses a custom prompt/decision protocol; HellaSwag is zero-shot ending likelihood; no private-vendor replication claim'}
    seen, ids = set(), {}
    for row in rows:
        key = row['variant'], row['id']
        if key in seen:
            raise ValueError('Duplicate reference result')
        seen.add(key)
        ids.setdefault(row['variant'], set()).add(row['id'])
    if set(ids) != set(VARIANTS) or any(ids[v] != ids['native'] for v in VARIANTS):
        raise ValueError('Reference variants have unequal denominators')
    native_by_id={r['id']:r for r in rows if r['variant']=='native'}
    for variant in VARIANTS:
        selected = [r for r in rows if r['variant'] == variant]
        n = len(selected)
        stats = {'planned_items': n, 'completed_items': sum(r['status'] == 'ok' for r in selected),
                 'errors': sum(r['status'] == 'error' for r in selected),
                 'over_limit': sum(r['status'] == 'over_limit' for r in selected),
                 'not_run_time_budget': sum(r['status'] == 'not_run_time_budget' for r in selected),
                 'documents_or_source_groups': len({r['document_id'] for r in selected})}
        for metric in metrics:
            stats[metric] = sum(r['status'] == 'ok' and r.get(metric) is True for r in selected)/n if n else None
        for field in('decision_answer','prediction','prediction_norm'):
            if any(field in r for r in native_by_id.values()):
                stats['native_agreement_'+field]=sum(r['status']=='ok'and native_by_id[r['id']]['status']=='ok'
                    and field in r and field in native_by_id[r['id']]and r[field]==native_by_id[r['id']][field]
                    for r in selected)/n if n else None
        stats['completed_transfer_items']=sum(r['status']=='ok'and r.get('transfer_used',False)for r in selected)
        stats['completed_native_fallback_items']=sum(r['status']=='ok'and r.get('fallback',False)for r in selected)
        if stats['completed_native_fallback_items']:
            stats['fallback_note']='Overall accuracy includes explicit native fallback; fallback rows do not establish transfer quality or acceleration.'
        result['variants'][variant] = stats
    for metric in metrics:
        paired = [{**row, 'arm': 'native_cold' if row['variant'] == 'native' else 'transfer_cold',
                   'correct': row['status'] == 'ok' and row.get(metric) is True}
                  for row in rows if row['variant'] in ('native', 'mapper')]
        result['paired'][metric] = paired_quality(paired)
    result['fully_completed'] = all(x['completed_items'] == x['planned_items'] for x in result['variants'].values())
    return result


def run_quality_reference(runtime, examples, mapper, output, *, freegen_max_new_tokens=16,
                          max_seconds=1800, callback=None, seed=19):
    from .qa_training import split_document_prompt
    if not examples or freegen_max_new_tokens < 1 or max_seconds <= 0:
        raise ValueError('Nonempty QuALITY evaluation and positive budgets required')
    if any(e.split != 'eval' or set(e.choices) != set('ABCD') for e in examples):
        raise ValueError('QuALITY requires eval records with four options')
    _validate_mapper(runtime, mapper, examples)
    limit = min(getattr(runtime, 'prompt_token_limit', 8192),
                runtime.source.config.max_position_embeddings-freegen_max_new_tokens,
                runtime.target.config.max_position_embeddings-freegen_max_new_tokens)
    root, config = _new_run(output, QUALITY_PROTOCOL, {'seed': seed, 'runtime': runtime.identities,
        'mapper': mapper.manifest, 'example_hash': json_digest([asdict(e) for e in examples]),
        'prompt_tokens_limit': limit, 'freegen_max_new_tokens': freegen_max_new_tokens,
        'max_seconds': max_seconds, 'protocol_note': 'Custom QuALITY decision+unconstrained generation; not exact vendor prompt/scorer reproduction'})
    deadline, rows = time.monotonic()+max_seconds, []
    rng = random.Random(seed)
    with (root/'rows.jsonl').open('w') as log:
        for index, example in enumerate(examples):
            base = {'id': example.id, 'document_id': example.document_id, 'context_hash': example.context_hash,
                'prompt_hash': 'unavailable', 'run_config_hash': config['run_config_hash'], 'repeat': 0,
                'gold_answer': example.answer, 'protocol': QUALITY_PROTOCOL, 'fallback': False}
            try:
                full, prefix, suffix = split_document_prompt(runtime, example)
                base.update(prompt_hash=json_digest(full), prompt_tokens=len(full),
                            document_prefix_tokens=len(prefix), native_query_tokens=len(suffix))
                tokens = option_token_ids(runtime.tokenizer, example.choices)
                token_error = None
            except Exception as exc:
                token_error = exc
            order = list(VARIANTS)
            rng.shuffle(order)
            for variant in order:
                # An assignment evaluates its RHS before releasing the old LHS.
                # Drop previous variant caches AND the last decode output first;
                # otherwise that output can retain a second full GPU KV cache.
                cache = logits = answer = None
                row = {**base, 'variant': variant}
                if time.monotonic() >= deadline:
                    row.update(status='not_run_time_budget')
                elif token_error is not None:
                    row.update(status='error', error=str(token_error), error_type=type(token_error).__name__)
                elif len(full) > limit:
                    row.update(status='over_limit')
                else:
                    try:
                        started = time.perf_counter()
                        cache, logits, first, timing = _prepare(runtime, mapper, variant, full, len(prefix))
                        decision = score_option_logits(logits, tokens, example.answer)
                        generated = [first]
                        model = runtime.source if variant == 'source' else runtime.target
                        eos = model.config.eos_token_id
                        eos = set(eos if isinstance(eos, list) else [eos])
                        for _ in range(freegen_max_new_tokens-1):
                            if generated[-1] in eos:
                                break
                            answer = runtime.forward(model, [generated[-1]], cache)
                            generated.append(int(answer.logits[0,-1].argmax().item()))
                            cache = answer.past_key_values
                        runtime.sync()
                        text = runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        finish = 'eos' if generated[-1] in eos else 'length'
                        row.update({**timing, **decision,
                            **generation_metrics(text, example.answer, example.choices, finish),
                            'status': 'ok', 'answer': text, 'generated_token_ids': generated,
                            'elapsed_ms': (time.perf_counter()-started)*1000})
                    except Exception as exc:
                        row.update(status='error', error=str(exc), error_type=type(exc).__name__)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                # Rows contain Python scalars/lists only, never tensor owners.
                cache = logits = answer = None
            if callback and ((index+1)%10 == 0 or index+1 == len(examples)):
                callback({'phase': 'reference_quality', 'processed_items': index+1, 'planned_items': len(examples)})
    summary = _summary(rows, ('decision_correct', 'free_generation_strict_correct', 'free_generation_format_valid'))
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def run_hellaswag_reference(runtime, records, mapper, output, *, max_seconds=1800, callback=None, seed=19):
    if not records or max_seconds <= 0:
        raise ValueError('Nonempty HellaSwag records and a positive budget required')
    registry = [Example(r['benchmark_id'], r['source_id'], r['query'], 'Ending likelihood',
                {chr(65+i): x for i,x in enumerate(r['choices'])}, chr(65+r['gold_index']), 'eval') for r in records]
    _validate_mapper(runtime, mapper, registry)
    limit = min(getattr(runtime, 'prompt_token_limit', 8192), runtime.source.config.max_position_embeddings,
                runtime.target.config.max_position_embeddings)
    root, config = _new_run(output, HELLASWAG_PROTOCOL, {'seed': seed, 'runtime': runtime.identities,
        'mapper': mapper.manifest, 'records_hash': json_digest(records), 'max_tokens': limit,
        'max_seconds': max_seconds, 'harness_revision': HARNESS_REVISION, 'target_delimiter': ' ',
        'prompt': 'Zero-shot raw completion, no chat template or option letters',
        'acc_norm': 'sum ending token log probabilities / len(processed ending string in characters)'})
    deadline, rows = time.monotonic()+max_seconds, []
    rng = random.Random(seed)
    with (root/'rows.jsonl').open('w') as log:
        for index, record in enumerate(records):
            base = {'id': record['benchmark_id'], 'document_id': record['source_id'],
                'context_hash': hashlib.sha256(record['query'].encode()).hexdigest(),
                'prompt_hash': 'unavailable', 'run_config_hash': config['run_config_hash'], 'repeat': 0,
                'gold_index': record['gold_index'], 'protocol': HELLASWAG_PROTOCOL, 'fallback': False}
            try:
                validate_hellaswag_record(record)
                pairs = [encode_pair(runtime.tokenizer, record['query'], ' '+ending) for ending in record['choices']]
                prefix = pairs[0][0]
                if any(p != prefix for p,_ in pairs):
                    raise ValueError('Candidate endings do not share identical context token IDs')
                if hasattr(runtime, 'source_tokenizer'):
                    other = [encode_pair(runtime.source_tokenizer, record['query'], ' '+ending) for ending in record['choices']]
                    if other != pairs:
                        raise ValueError('Source and target continuation token IDs differ')
                base.update(prompt_hash=json_digest(pairs), context_tokens=len(prefix),
                            ending_token_counts=[len(e) for _,e in pairs],
                            candidate_total_tokens=[len(p)+len(e) for p,e in pairs])
                token_error = None
            except Exception as exc:
                token_error = exc
            order = list(VARIANTS)
            rng.shuffle(order)
            for variant in order:
                # Prefix caches are per item/variant; only candidate endings
                # within this attempt share one. Release it before a new RHS.
                cache = logits = None
                row = {**base, 'variant': variant}
                if time.monotonic() >= deadline:
                    row.update(status='not_run_time_budget')
                elif token_error is not None:
                    row.update(status='error', error=str(token_error), error_type=type(token_error).__name__)
                elif max(base['candidate_total_tokens']) > limit:
                    row.update(status='over_limit')
                else:
                    try:
                        started = time.perf_counter()
                        cache, logits, _, timing = _prepare(runtime, mapper, variant, prefix, len(prefix)-1)
                        model = runtime.source if variant == 'source' else runtime.target
                        scores = [_ending_ll(runtime, model, cache, logits, ending) for _,ending in pairs]
                        runtime.sync()
                        row.update(status='ok', **timing,
                            **score_endings(scores, record['choices'], record['gold_index']),
                            elapsed_ms=(time.perf_counter()-started)*1000,
                            score_scope='Full conditional likelihood of all ending tokens; shared context cache cloned for each candidate')
                    except Exception as exc:
                        row.update(status='error', error=str(exc), error_type=type(exc).__name__)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                cache = logits = None
            if callback and ((index+1)%25 == 0 or index+1 == len(records)):
                callback({'phase': 'reference_hellaswag', 'processed_items': index+1, 'planned_items': len(records)})
    summary = _summary(rows, ('acc', 'acc_norm'))
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
