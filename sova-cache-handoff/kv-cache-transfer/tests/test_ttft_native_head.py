"""The hybrid TTFT clock must include native-head reconstruction."""
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Tiny CPU fixture')
class TimingHeadTests(unittest.TestCase):
    def test_hybrid_recomputes_native_head_inside_every_transfer_timer(self):
        import torch
        from transformers import DynamicCache

        from cloud.ttft_worker import bench
        from kvverify.hf_backend import legacy

        prefix, suffix = [1] * 300, [2] * 20
        calls = []
        timed_calls = []
        runtime = SimpleNamespace(source='source', target='target', tokenizer=None,
                                  torch=torch, DynamicCache=DynamicCache)
        example = SimpleNamespace(id='example', document_id='doc', choices={'A':'a', 'B':'b'})

        def forward(model, ids, cache=None):
            offset = cache.get_seq_length() if cache else 0
            calls.append((model, len(ids), offset))
            k = torch.zeros(1, 1, offset + len(ids), 2)
            return SimpleNamespace(past_key_values=DynamicCache.from_legacy_cache(((k, k.clone()),)),
                                   logits=torch.tensor([[[2., 1., 0.]]]))

        def timed(run):
            first = len(calls)
            result = run()
            timed_calls.append(calls[first:])
            return result, 1.0

        def copy_cache(cache, *_args, **_kwargs):
            return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in legacy(cache)))

        runtime.forward, runtime.timed = forward, timed
        mapper = SimpleNamespace(apply=copy_cache)
        with (patch('kvverify.qa_training.split_document_prompt', return_value=(prefix+suffix,prefix,suffix)),
              patch('kvverify.fast_affine.map_affine_fast', side_effect=lambda _, cache, *a, **k: copy_cache(cache)),
              patch('kvverify.quality.option_token_ids', return_value={'A':0,'B':1}),
              patch.object(torch.cuda, 'reset_peak_memory_stats'),
              patch.object(torch.cuda, 'max_memory_allocated', return_value=0)):
            rows, checks, _, _ = bench(runtime, mapper, example, 1, random.Random(0),
                                        fast_block_tokens=256, diagnose_cache_tensors=False,
                                        native_prefix_tokens=256)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row['passed'] for row in checks))
        self.assertEqual(timed_calls.count([('source', 300, 0), ('target', 256, 0), ('target', 20, 300)]), 4)
        self.assertEqual(timed_calls.count([('target', 256, 0), ('target', 20, 300)]), 4)
        self.assertEqual(timed_calls.count([('target', 64, 256)]), 2)
        by_arm = {row['arm']:row for row in rows}
        self.assertEqual(by_arm['fast_source_hit']['target_prefill_tokens'], 276)
        self.assertEqual(by_arm['fast_source_hit']['imported_translated_tokens'], 44)
        self.assertEqual(by_arm['native_256_hit']['target_prefill_tokens'], 64)
