"""Fit a dense 1.7B-to-8B cache map and evaluate held-out documents."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.probe_worker import OUT, notice, upload, write


def aligned_layers(source_count, target_count):
    if source_count < 2 or target_count < 2:
        raise ValueError('At least two layers per model are required')
    return [round(index * (source_count - 1) / (target_count - 1))
            for index in range(target_count)]


def calibration_examples(examples, count, seed):
    unique = {}
    for example in sorted(examples, key=lambda row: row.id):
        unique.setdefault(example.document_id, example)
    if len(unique) < count:
        raise ValueError('Too few distinct calibration documents')
    return random.Random(seed).sample(list(unique.values()), count)


def main():
    import torch
    from huggingface_hub import snapshot_download
    from kvverify.affine_cache import CrossHeadMoments
    from kvverify.data import assert_disjoint, load_examples
    from kvverify.hf_backend import DenseRuntime, legacy
    from kvverify.native_prefix import replace_prefix
    from kvverify.qa_training import split_document_prompt
    from kvverify.quality import option_token_ids, score_option_logits
    from kvverify.state_diagnostic import distribution_metrics

    config = json.loads(Path(os.environ['KV_CONFIG']).read_text())
    if config['mode'] != 'harvey_pilot' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Bigger-pair evaluation requires a BF16-capable CUDA GPU')
    for path, expected in config['input_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input changed: ' + path)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark = config['benchmark']
    examples = load_examples(ROOT / 'inputs/evaluation.jsonl', 'eval')
    calibrations = calibration_examples(load_examples(ROOT / 'inputs/quality-train.jsonl', 'calibration'),
                                        config['calibration_documents'], config['seed'])
    assert_disjoint(calibrations, examples)
    write('config.json', config)

    paths = []
    for model in config['models']:
        notice({'phase': 'downloading_model', 'candidate': model['repo']})
        paths.append(snapshot_download(model['repo'], revision=model['revision'], token=False,
                     allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json'], max_workers=4))
    notice({'phase': 'loading_models'})
    runtime = DenseRuntime.from_local(*paths, dtype='bfloat16')
    for model in (runtime.source, runtime.target):
        model.requires_grad_(False)
    source_layers = aligned_layers(runtime.source_geometry[2], runtime.target_geometry[2])
    if (runtime.source_geometry, runtime.target_geometry) != ((8, 128, 28), (8, 128, 36)):
        raise ValueError('Unexpected model-pair geometry')
    write('environment.json', {'gpu': torch.cuda.get_device_name(0),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'torch': str(torch.__version__), 'identities': runtime.identities,
        'source_layers': source_layers, 'attention': 'Torch SDPA automatic backend'})

    grouped = defaultdict(list)
    for example in examples:
        full, prefix, suffix = split_document_prompt(runtime, example)
        if len(full) > config['max_prompt_tokens'] or len(prefix) <= 256:
            raise ValueError(f'Prompt outside the frozen limits: {example.id} ({len(full)} tokens)')
        grouped[example.document_id].append((example, full, prefix, suffix))
    if len(grouped) != config['expected_documents'] or len(examples) != config['expected_questions']:
        raise ValueError('Unexpected frozen evaluation denominator')
    for rows in grouped.values():
        if any(item[2] != rows[0][2] for item in rows):
            raise ValueError('Queries for the same document produced different prefix tokens')
    write('prompt-lengths.json', {example.id: len(full) for rows in grouped.values()
                                 for example, full, _, _ in rows})

    moments = CrossHeadMoments(runtime.target_geometry, source_layers, 'cuda:0',
                               source_geometry=runtime.source_geometry)
    sampler = random.Random(config['seed'])
    with torch.inference_mode():
        for index, example in enumerate(calibrations, 1):
            _, prefix, _ = split_document_prompt(runtime, example)
            prefix = prefix[:config['calibration_prefix_tokens']]
            if len(prefix) <= 256:
                raise ValueError('Calibration prefix too short')
            indices = sorted(sampler.sample(range(256, len(prefix)), min(512, len(prefix) - 256)))
            source = runtime.forward(runtime.source, prefix).past_key_values
            native = runtime.forward(runtime.target, prefix).past_key_values
            moments.update(source, native, runtime.source, runtime.target, token_indices=indices)
            del source, native
            notice({'phase': 'calibration', 'document': index, 'total': len(calibrations),
                    'sampled_states': moments.n})
        mapper = moments.fit(config['ridge_alpha'], True, {
            'source_sha256': runtime.identities['source']['sha256'],
            'target_sha256': runtime.identities['target']['sha256'],
            'tokenizer_sha256': runtime.tokenizer_sha256,
            'calibration_ids': [row.id for row in calibrations],
            'calibration_documents': [row.document_id for row in calibrations],
            'calibration_context_hashes': [row.context_hash for row in calibrations],
            'calibration_prefix_limit': config['calibration_prefix_tokens']}).to('cuda:0')
        del moments
        torch.cuda.empty_cache()

        def query(model, suffix, cache):
            # DynamicCache appends by replacing entries rather than mutating the
            # prefix tensors. Recreate its container for each independent query.
            copy = runtime.DynamicCache.from_legacy_cache(legacy(cache))
            logits = runtime.forward(model, suffix, copy).logits[0, -1]
            return logits.detach().float()

        rows, timings = [], []
        for index, document in enumerate(grouped.values(), 1):
            notice({'phase': 'evaluating_documents', 'document': index, 'total': len(grouped)})
            example, full, prefix, suffix = document[0]
            native_full, native_ms = runtime.timed(
                lambda: runtime.forward(runtime.target, full).logits[0, -1].float())
            native_cache = runtime.forward(runtime.target, prefix).past_key_values
            source_cache, source_ms = runtime.timed(
                lambda: runtime.forward(runtime.source, prefix).past_key_values)
            mapped, map_ms = runtime.timed(lambda: mapper.apply(source_cache, runtime.source, runtime.target))
            header, header_ms = runtime.timed(
                lambda: runtime.forward(runtime.target, prefix[:256]).past_key_values)
            mapped_header, replacement_ms = runtime.timed(lambda: replace_prefix(mapped, header, 256))
            native_split, split_query_ms = runtime.timed(lambda: query(runtime.target, suffix, native_cache))
            guard = distribution_metrics(native_full, native_split)
            candidates = {'source': (runtime.source, source_cache),
                          'translated': (runtime.target, mapped),
                          'translated_native256': (runtime.target, mapped_header),
                          'native256_only': (runtime.target, header)}
            preview = {name: runtime.timed(lambda: query(model, suffix, cache))
                       for name, (model, cache) in candidates.items()}
            timings.append({'document_id': example.document_id, 'prefix_tokens': len(prefix),
                'native_cold_ms': native_ms, 'native_cache_hit_query_ms': split_query_ms,
                'source_prefill_ms': source_ms, 'conversion_ms': map_ms,
                'translated_query_ms': preview['translated'][1],
                'native256_prefill_ms': header_ms, 'native256_replacement_ms': replacement_ms,
                'translated_native256_query_ms': preview['translated_native256'][1],
                'native_full_vs_split': guard})
            for i, (example, full, _, suffix) in enumerate(document):
                native = native_full if i == 0 else runtime.forward(runtime.target, full).logits[0, -1].float()
                logits = {name: preview[name][0] if i == 0 else query(model, suffix, cache)
                          for name, (model, cache) in candidates.items()}
                tokens = option_token_ids(runtime.tokenizer, example.choices)
                native_score = score_option_logits(native, tokens, example.answer)
                variants = {}
                for name, scores in logits.items():
                    score = score_option_logits(scores, tokens, example.answer)
                    variants[name] = {'score': score, 'agrees_with_native':
                        score['decision_answer'] == native_score['decision_answer'],
                        'kl': distribution_metrics(native, scores)['native_to_candidate_kl_nats']}
                rows.append({'id': example.id, 'document_id': example.document_id,
                             'gold': example.answer, 'native': native_score, 'variants': variants})
            write('rows.json', rows)
            write('timings.json', timings)
            del candidates, preview, native_cache, source_cache, mapped, mapped_header, header

        summaries = {}
        for name in rows[0]['variants']:
            results = [row['variants'][name] for row in rows]
            summaries[name] = {'correct': sum(item['score']['decision_correct'] for item in results),
                'native_agreement': sum(item['agrees_with_native'] for item in results),
                'mean_native_kl': sum(item['kl'] for item in results) / len(results)}
        artifact = mapper.save(OUT / 'ridge')
        summary = {'status': 'completed', 'benchmark': benchmark, 'documents': len(grouped), 'questions': len(rows),
            'native_correct': sum(row['native']['decision_correct'] for row in rows),
            'variants': summaries, 'sampled_states': artifact['calibration_tokens'],
            'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
            'native_split_checks': [{'kl': item['native_full_vs_split']['native_to_candidate_kl_nats'],
                                      'first_token_equal': item['native_full_vs_split']['first_token_equal']}
                                     for item in timings],
            'scope': ('Reserved official QuALITY questions; constrained letter scoring with Qwen thinking disabled.'
                      if benchmark == 'quality' else
                      'Text-only draft-fact diagnostic derived from Harvey LAB; no official agent score.'),
            'timing_scope': 'Single model-only timing sample per document.'}
        write('summary.json', summary)
        write('worker-status.json', summary)
        notice({'phase': 'completed', 'correct': summary['native_correct'], 'document': len(grouped),
                'total': len(grouped), 'peak_gpu_mib': summary['peak_gpu_mib']})


if __name__ == '__main__':
    def interrupt(*_):
        raise TimeoutError('SageMaker worker deadline')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        write('worker-status.json', {'status': 'failed'})
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
