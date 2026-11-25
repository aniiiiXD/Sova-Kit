"""Prospective, bounded handoff-tail diagnostic; never changes old run scores.

Call run_probe from an explicitly approved worker with already-loaded models
and a verified mapper. No model download, cloud submission or CLI execution is
provided here. All scoring and serialization happen outside model timers.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .data import check_heldout, json_digest
from .hf_backend import legacy
from .quality import option_token_ids, score_option_logits

PROTOCOL = 'tail-boundary-v1'
PROBABILITY_EQUIVALENCE_PROTOCOL = 'oracle-probability-v2'


def _clip_divergence_roundoff(value):
    """Only tiny negative FP64 summation roundoff may become mathematical zero."""
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError('Nonfinite or materially negative probability divergence')
    return max(0., value), value < 0


def probability_equivalence(torch, reference, candidate):
    """Separate prospective probability-level native-cache equivalence check.

    This utility never changes the original v1 oracle gate or historical rows.
    It copies detached vectors to CPU float64 and must be called outside model
    timing. There is no epsilon smoothing, probability truncation, renormalized
    option-only distribution, or raw-logit allclose gate. All vocabulary entries
    contribute. New acceptance claims require fresh controls under this declared
    protocol; a post-hoc audit alone cannot turn the old run into a pass.
    """
    def vector(value):
        result = value.detach() if hasattr(value, 'detach') else torch.as_tensor(value, device='cpu', dtype=torch.float64)
        result = result.to(device='cpu', dtype=torch.float64).contiguous()
        if result.ndim != 1 or result.numel() == 0 or not torch.isfinite(result).all():
            raise ValueError('Finite nonempty one-dimensional vocabulary vectors required')
        return result
    left, right = vector(reference), vector(candidate)
    if left.shape != right.shape:
        raise ValueError('Reference/candidate vocabulary shapes must match')
    logp = torch.log_softmax(left, dim=-1)
    logq = torch.log_softmax(right, dim=-1)
    p, q = logp.exp(), logq.exp()
    tv = float(.5*(p-q).abs().sum())
    maximum = float((p-q).abs().max())
    forward_raw = float((p*(logp-logq)).sum())
    reverse_raw = float((q*(logq-logp)).sum())
    forward, clipped_forward = _clip_divergence_roundoff(forward_raw)
    reverse, clipped_reverse = _clip_divergence_roundoff(reverse_raw)
    if not math.isfinite(tv) or not math.isfinite(maximum):
        raise ValueError('Nonfinite probability difference')
    first_equal = int(left.argmax()) == int(right.argmax())
    thresholds = {'max_total_variation': 1e-3, 'max_forward_kl_nats': 1e-4,
                  'max_reverse_kl_nats': 1e-4, 'first_token_must_match': True,
                  'negative_kl_roundoff_tolerance': 1e-12}
    passed = first_equal and tv <= thresholds['max_total_variation'] and forward <= thresholds['max_forward_kl_nats'] and reverse <= thresholds['max_reverse_kl_nats']
    return {'protocol': PROBABILITY_EQUIVALENCE_PROTOCOL, 'passed': bool(passed),
            'first_token_equal': first_equal, 'total_variation': tv,
            'max_probability_difference': maximum,
            'native_to_candidate_kl_nats': forward, 'candidate_to_native_kl_nats': reverse,
            'raw_native_to_candidate_kl_nats': forward_raw,
            'raw_candidate_to_native_kl_nats': reverse_raw,
            'forward_kl_roundoff_clipped': clipped_forward,
            'reverse_kl_roundoff_clipped': clipped_reverse,
            'vocabulary_size': left.numel(), 'probability_precision': 'float64_cpu',
            'thresholds': thresholds, 'raw_logit_allclose_used': False,
            'scope': 'Numerical native-cache equivalence; not mapper quality acceptance or retrospective v1 gate replacement'}


def select_examples(examples, max_examples=16, seed=19):
    """Select documents, then questions, before inspecting tokens or outputs."""
    if type(max_examples) is not int or not 1 <= max_examples <= 16:
        raise ValueError('The diagnostic is bounded to 1-16 distinct documents')
    if type(seed) is not int:
        raise ValueError('Integer selection seed required')
    bydoc, contexts, ids = {}, {}, set()
    for example in sorted(examples, key=lambda e: e.id):
        if example.split != 'eval' or example.id in ids:
            raise ValueError('Probe inputs must be unique eval records')
        ids.add(example.id)
        if example.document_id in bydoc and bydoc[example.document_id][0].context_hash != example.context_hash:
            raise ValueError('One document ID cannot refer to different contexts')
        if example.context_hash in contexts and contexts[example.context_hash] != example.document_id:
            raise ValueError('Repeated contexts must share a document ID')
        contexts[example.context_hash] = example.document_id
        bydoc.setdefault(example.document_id, []).append(example)
    if not bydoc:
        raise ValueError('Nonempty diagnostic input required')
    rng = random.Random(seed)
    documents = rng.sample(sorted(bydoc), min(max_examples, len(bydoc)))
    return [rng.choice(bydoc[document]) for document in documents]


def validate_mapper(runtime, examples):
    mapper = runtime.mapper
    if mapper is None:
        raise ValueError('An explicitly loaded, verified mapper is required')
    for field, value in (('source_sha256', runtime.identities['source']['sha256']),
                         ('target_sha256', runtime.identities['target']['sha256']),
                         ('tokenizer_sha256', runtime.tokenizer_sha256)):
        if mapper.manifest.get(field) != value:
            raise ValueError('Mapper identity mismatch: '+field)
    if runtime.tokenizer_fingerprint(runtime.tokenizer) != runtime.tokenizer_sha256:
        raise ValueError('Runtime tokenizer changed after its identity was recorded')
    check_heldout(mapper.manifest, examples)


def validate_row(row):
    if row.get('protocol') != PROTOCOL or row.get('fallback') is not False:
        raise ValueError('Unknown diagnostic protocol or hidden fallback')
    if row.get('status') not in ('ok', 'error', 'over_limit'):
        raise ValueError('Unknown diagnostic outcome')
    if row['status'] != 'ok':
        return
    for name in ('ttft_ms', 'source_prefill_ms', 'native_prefix_prefill_ms', 'map_ms', 'target_first_ms'):
        if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]) or row[name] < 0:
            raise ValueError('Invalid diagnostic timing: '+name)
    if row['arm'] == 'native':
        if row['imported_tokens'] != 0 or row['target_prefill_tokens'] != row['prompt_tokens']:
            raise ValueError('Native accounting changed')
    elif (row['target_prefill_tokens'] != row['tail_tokens']
          or row['imported_tokens'] != row['prompt_tokens']-row['tail_tokens']
          or row['imported_tokens'] < 1 or row.get('cache_import_verified') is not True):
        raise ValueError('Boundary import/recomputation accounting failed')
    oracle = row['arm'].startswith('oracle_suffix_')
    if row['semantic_transfer'] != row['arm'].startswith('transfer_suffix_'):
        raise ValueError('Native oracle must not be described as semantic transfer')
    if oracle and (row['source_prefill_tokens'] != 0 or row['target_total_prefill_tokens'] != row['prompt_tokens']):
        raise ValueError('Oracle must compute every prompt token on the native target')
    if row['ttft_ms']+.01 < sum(row[k] for k in ('source_prefill_ms', 'native_prefix_prefill_ms', 'map_ms', 'target_first_ms')):
        raise ValueError('Cold TTFT omits part of the diagnostic path')


def _timed_first(runtime, ids, cache=None):
    def forward():
        result = runtime.forward(runtime.target, ids, cache)
        logits = result.logits[0, -1]
        return logits, int(logits.argmax().item())
    return runtime.timed(forward)


def _attempt(runtime, ids, tail, base):
    torch = runtime.torch
    if tail is not None and tail >= len(ids):
        raise ValueError('Requested target suffix leaves no source prefix')
    oracle = base['arm'].startswith('oracle_suffix_')
    base['model_execution_started'] = True
    devices = {next(m.parameters()).device for m in (runtime.source, runtime.target)}
    for device in devices:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
    runtime.sync()
    started = time.perf_counter()
    source_ms = map_ms = native_prefix_ms = 0.
    if tail is None:
        cache, suffix, imported = None, ids, 0
    else:
        if oracle:
            cache, native_prefix_ms = runtime.timed(lambda: runtime.forward(runtime.target, ids[:-tail]).past_key_values)
        else:
            source, source_ms = runtime.timed(lambda: runtime.forward(runtime.source, ids[:-tail]).past_key_values)
            with torch.inference_mode():
                cache, map_ms = runtime.timed(lambda: runtime.mapper.apply(source, runtime.source, runtime.target))
            del source
        imported = cache.get_seq_length()
        heads, dim, layers = runtime.target_geometry
        rows = legacy(cache)
        expected = (1, heads, len(ids)-tail, dim)
        if imported != len(ids)-tail or len(rows) != layers or any(k.shape != expected or v.shape != expected for k, v in rows):
            raise ValueError('Mapped cache length/layer/shape differs from this boundary')
        suffix = ids[-tail:]
    (logits, first), target_ms = _timed_first(runtime, suffix, cache)
    runtime.sync()
    elapsed = (time.perf_counter()-started)*1000
    memory = {str(device): {'allocated_mib': torch.cuda.max_memory_allocated(device)/2**20,
                            'reserved_mib': torch.cuda.max_memory_reserved(device)/2**20}
              for device in devices if device.type == 'cuda'}
    # Copy/check/logit diagnostics only after synchronized first-token timing.
    vector = logits.detach().float().cpu().contiguous()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError('Nonfinite or malformed full-vocabulary first logits')
    row = {**base, 'status': 'ok', 'first_token_id': first,
        'first_logits_sha256': hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
        'ttft_ms': elapsed, 'source_prefill_ms': source_ms, 'map_ms': map_ms,
        'native_prefix_prefill_ms': native_prefix_ms,
        'target_first_ms': target_ms, 'imported_tokens': imported,
        'target_prefill_tokens': len(suffix), 'cache_import_verified': tail is not None,
        'source_prefill_tokens': len(ids)-tail if tail is not None and not oracle else 0,
        'target_total_prefill_tokens': len(ids) if tail is None or oracle else tail,
        'semantic_transfer': tail is not None and not oracle,
        'cache_origin': 'native_target_prefix' if oracle else 'mapped_source_prefix' if tail is not None else 'none',
        'map_includes_move_and_import': tail is not None and not oracle, 'gpu_peak_memory': memory,
        'measurement': 'Synchronized in-process cold execution through first-token selection; not HTTP or first-visible-text latency'}
    validate_row(row)
    return row, vector


def compare_logits(torch, reference, candidate):
    from torch.nn import functional as F
    if reference.shape != candidate.shape:
        raise ValueError('Native and transferred vocabulary shapes differ')
    left = F.log_softmax(reference.float(), dim=-1)
    right = F.log_softmax(candidate.float(), dim=-1)
    divergence = float(F.kl_div(right, left, reduction='sum', log_target=True))
    if not math.isfinite(divergence) or divergence < -1e-5:
        raise ValueError('Invalid full-vocabulary native-to-candidate KL')
    return {'native_first_token_agreement': int(reference.argmax()) == int(candidate.argmax()),
            'native_to_candidate_kl_nats': max(0., divergence),
            'kl_roundoff_clipped': divergence < 0,
            'native_max_logit_abs_difference': float((reference-candidate).abs().max()),
            'native_first_token_probability': float(right[int(reference.argmax())].exp())}


def summarize(rows):
    arms = {}
    for arm in sorted({row['arm'] for row in rows}):
        group = [row for row in rows if row['arm'] == arm]
        okay = [row for row in group if row['status'] == 'ok']
        paired = [row for row in okay if row.get('native_reference_available')]
        median = lambda name: statistics.median(row[name] for row in okay) if okay else None
        arms[arm] = {'attempts': len(group), 'returned': len(okay),
            'errors': sum(row['status'] == 'error' for row in group),
            'blocked_by_oracle_gate': sum(row.get('blocked_by_oracle_gate') is True for row in group),
            'over_limit': sum(row['status'] == 'over_limit' for row in group),
            'decision_accuracy_all_attempts': sum(row.get('decision_correct') is True for row in group)/len(group),
            'native_references_available': len(paired),
            'native_first_token_agreement_all_attempts': sum(row.get('native_first_token_agreement') is True for row in group)/len(group),
            'median_native_kl_nats_on_pairs': statistics.median(row['native_to_candidate_kl_nats'] for row in paired) if paired else None,
            'median_option_probability_mass_on_success': median('option_probability_mass'),
            'median_ttft_ms_on_success': median('ttft_ms'),
            'median_source_prefill_ms_on_success': median('source_prefill_ms'),
            'median_native_prefix_prefill_ms_on_success': median('native_prefix_prefill_ms'),
            'median_map_ms_on_success': median('map_ms'),
            'median_target_first_ms_on_success': median('target_first_ms')}
    oracles = [row for row in rows if row['arm'].startswith('oracle_suffix_')]
    applicable = [row for row in oracles if row['status'] != 'over_limit']
    passed = bool(applicable) and all(row.get('oracle_matches_native') is True for row in applicable)
    return {'protocol': PROTOCOL, 'status': 'diagnostic_only_not_acceptance', 'acceptance': 'not_established',
        'arms': arms, 'oracle_gate': {'status': 'passed' if passed else 'failed' if applicable else 'not_applicable',
            'attempts': len(oracles), 'checked_or_failed': len(applicable),
            'matches_native': sum(row.get('oracle_matches_native') is True for row in oracles),
            'semantic_transfer': False,
            'policy': 'All eligible native-cache suffix controls must match before mapped arms execute for that example.'},
        'limitations': ['At most 16 distinct documents; no statistical noninferiority conclusion.',
            'Native runs first, then native-cache oracles, then mapped arms; tail order is seeded/shuffled; no timed repetitions or performance warmup.',
            'Decision scoring restricts first-token choices; no free-generation answer quality is established.',
            'Successful-only timing/KL summaries condition on available pairs; all attempt/error counts remain visible.']}


def run_probe(runtime, examples, output, tails=(1, 8, 32), max_examples=16, seed=19, callback=None,
              oracle_rtol=.02, oracle_atol=.1, oracle_max_kl_nats=.01):
    """Save a new, standalone handoff-boundary ledger using an existing mapper."""
    examples = list(examples)
    if (not tails or len(set(tails)) != len(tails)
            or any(type(tail) is not int or not 1 <= tail <= 256 for tail in tails)):
        raise ValueError('Distinct positive suffix lengths up to 256 required')
    if any(not math.isfinite(value) or value < 0 for value in (oracle_rtol, oracle_atol, oracle_max_kl_nats)):
        raise ValueError('Finite nonnegative oracle tolerances required')
    selected = select_examples(examples, max_examples, seed)
    validate_mapper(runtime, examples)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'logits').mkdir()
    prepared = []
    for example in selected:
        try:
            ids = runtime.tokenize(example)
            prepared.append({'example': example, 'ids': ids, 'error': None,
                             'prompt_hash': json_digest(ids)})
        except Exception as exc:
            prepared.append({'example': example, 'ids': None,
                'error': {'error_type': type(exc).__name__, 'error': str(exc)},
                'prompt_hash': json_digest({'unavailable_prompt': asdict(example)})})
    config = {'protocol': PROTOCOL, 'selected_ids': [e.id for e in selected],
        'selection': 'Seeded document sample then seeded question per document before tokenization/output; no length/correctness filtering',
        'seed': seed, 'tails': list(tails), 'max_examples': max_examples,
        'oracle_thresholds': {'rtol': oracle_rtol, 'atol': oracle_atol, 'max_kl_nats': oracle_max_kl_nats,
                              'first_token_must_match': True},
        'oracle_definition': 'Fresh true target prefix cache followed by the target suffix; no source model or mapper is used.',
        'model_identities': runtime.identities, 'tokenizer_sha256': runtime.tokenizer_sha256,
        'mapper_manifest': runtime.mapper.manifest,
        'inputs_sha256': json_digest([asdict(e) for e in selected]),
        'prompt_hashes': {p['example'].id: p['prompt_hash'] for p in prepared},
        'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        'measurement': 'All models already resident; tokenization/metrics/logit export outside timing. Fresh caches for every arm.',
        'native_order': 'Full native reference, then true-target-cache suffix controls, then mapped suffix arms.',
        'original_protocol_preserved': True}
    config_hash = json_digest(config)
    (root/'config.json').write_text(json.dumps({**config, 'config_sha256': config_hash}, indent=2)+'\n')
    with (root/'inputs.jsonl').open('w') as stream:
        for item in prepared:
            stream.write(json.dumps({'record': asdict(item['example']), 'token_ids': item['ids'],
                'prompt_hash': item['prompt_hash'], 'tokenization_error': item['error']})+'\n')
    from safetensors.torch import save_file
    rows, rng = [], random.Random(seed)
    native_cap = min(runtime.source.config.max_position_embeddings, runtime.target.config.max_position_embeddings)
    prompt_cap = getattr(runtime, 'prompt_token_limit', float('inf'))
    with (root/'traces.jsonl').open('w') as log:
        for index, item in enumerate(prepared):
            example, ids, reference = item['example'], item['ids'], None
            order = list(tails)
            rng.shuffle(order)
            native_decision = None
            oracle_matches = {}
            attempts = [('native', None), *[('oracle', tail) for tail in order], *[('transfer', tail) for tail in order]]
            for kind, tail in attempts:
                arm = 'native' if tail is None else kind+'_suffix_'+str(tail)
                row = {'protocol': PROTOCOL, 'arm': arm, 'tail_tokens': tail,
                    'id': example.id, 'document_id': example.document_id, 'context_hash': example.context_hash,
                    'prompt_hash': item['prompt_hash'], 'run_config_hash': config_hash,
                    'prompt_tokens': len(ids) if ids is not None else None,
                    'target_tail_token_ids': ids[-tail:] if ids is not None and tail is not None else None,
                    'source_prefix_token_hash': json_digest(ids[:-tail]) if ids is not None and kind == 'transfer' else None,
                    'native_prefix_token_hash': json_digest(ids[:-tail]) if ids is not None and kind == 'oracle' else None,
                    'target_start_position': len(ids)-tail if ids is not None and tail is not None else 0,
                    'source_checkpoint_sha256': runtime.identities['source']['sha256'] if kind == 'transfer' else None,
                    'target_checkpoint_sha256': runtime.identities['target']['sha256'],
                    'gold_answer': example.answer, 'fallback': False,
                    'native_reference_available': False, 'semantic_transfer': kind == 'transfer',
                    'model_execution_started': False}
                try:
                    if item['error']:
                        row.update(status='error', **item['error'])
                    elif len(ids) < 2 or len(ids) > prompt_cap or len(ids)+1 > native_cap:
                        row.update(status='over_limit', decision_correct=False)
                    elif kind == 'transfer' and not all(oracle_matches.get(suffix) is True for suffix in tails):
                        row.update(status='error', decision_correct=False, blocked_by_oracle_gate=True,
                            error_type='OracleGateError', error='Native-cache suffix controls did not all match the native full-prompt reference')
                    elif kind == 'oracle' and reference is None:
                        row.update(status='error', decision_correct=False, oracle_matches_native=False,
                            error_type='NativeReferenceError', error='Native reference failed; oracle comparison cannot be established')
                    else:
                        row, vector = _attempt(runtime, ids, tail, row)
                        row.update(score_option_logits(vector, option_token_ids(runtime.tokenizer, example.choices), example.answer))
                        name = f'logits/{index:03d}-{arm}.safetensors'
                        save_file({'first_logits': vector}, str(root/name))
                        row.update(first_logits_file=name,
                            first_logits_file_sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
                        if tail is None:
                            reference, native_decision = vector, row['decision_answer']
                        if reference is not None:
                            row.update(native_reference_available=True,
                                native_decision_agreement=row['decision_answer'] == native_decision,
                                **compare_logits(runtime.torch, reference, vector))
                        if kind == 'oracle':
                            close = bool(runtime.torch.allclose(reference, vector, rtol=oracle_rtol, atol=oracle_atol))
                            row.update(oracle_logits_allclose=close,
                                oracle_matches_native=close and row['native_first_token_agreement']
                                    and row['native_to_candidate_kl_nats'] <= oracle_max_kl_nats)
                except Exception as exc:
                    row.update(status='error', decision_correct=False,
                               error_type=type(exc).__name__, error=str(exc), fallback=False)
                if kind == 'oracle':
                    oracle_matches[tail] = row.get('oracle_matches_native') is True
                validate_row(row)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
            if callback:
                callback({'protocol': PROTOCOL, 'completed_examples': index+1, 'total_examples': len(prepared)})
    result = summarize(rows)
    (root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
