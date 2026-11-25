import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("KVVERIFY_TORCH_SMOKE") == "1", "Explicit tiny CPU model test")
class GeneratedContinuationTests(unittest.TestCase):
    def setUp(self):
        from test_affine_cache import AffineCacheTests

        self.fixture = AffineCacheTests()
        self.fixture.setUp()
        self.runtime = self.fixture.r
        self.torch = self.fixture.t

    def test_same_document_uses_correct_causal_answer_positions(self):
        from kvverify.affine_cache import AffineCacheMapper
        from kvverify.continuation_pilot import MultiDonorMapper, Trace, answer_token_loss
        from kvverify.train_mapper import model_forward

        runtime, torch = self.runtime, self.torch
        runtime.source = runtime.target
        manifest = {"head_mixing": True, "source_geometry": runtime.source_geometry,
                    "target_geometry": runtime.target_geometry, "source_layers": [0, 1]}
        base = AffineCacheMapper(manifest)
        with torch.no_grad():
            base.weight.copy_(torch.eye(16).expand_as(base.weight))
            base.bias.zero_()
        mapper = MultiDonorMapper(base, rank=4)
        prefix, question, answer = [1, 4, 7, 3, 2, 1], [4, 7, 8], [9, 10, 11, 47]
        with torch.no_grad():
            teacher = torch.stack([model_forward(runtime.target, prefix+question+answer[:i]).logits[0, -1]
                                   for i in range(len(answer))]).log_softmax(-1).half()
        trace = Trace(tuple(answer), "", True, teacher)
        loss = answer_token_loss(runtime, mapper, prefix, question, trace, native_head=2)
        self.assertLess(abs(float(loss.detach())), 2e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(mapper.up.grad).all())
        self.assertTrue(all(p.grad is None for p in runtime.target.parameters()))

    def test_trained_donor_receives_gradient_and_checkpoint_round_trips(self):
        from kvverify.continuation_pilot import MultiDonorMapper, Trace, answer_token_loss
        from kvverify.train_mapper import model_forward

        runtime, torch = self.runtime, self.torch
        base = self.fixture.fit()
        base.manifest["weights_sha256"] = "test-affine"
        mapper = MultiDonorMapper(base, rank=4)
        prefix, question, answer = [1, 4, 7, 3, 2, 1], [4, 7, 8], [9, 10, 11, 47]
        with torch.no_grad():
            mapper.up.fill_(.004)
            teacher = torch.stack([model_forward(runtime.target, prefix+question+answer[:i]).logits[0, -1]
                                   for i in range(len(answer))]).log_softmax(-1)
        loss = answer_token_loss(runtime, mapper, prefix, question, Trace(tuple(answer), "", True, teacher), native_head=2)
        loss.backward()
        self.assertGreater(float(mapper.down.grad[:, 1].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"residual"
            mapper.save(path, {"steps": 1})
            loaded = MultiDonorMapper.load(path, base)
            self.assertEqual(loaded.groups, mapper.groups)
            for actual, expected in zip(loaded.residual_parameters(), mapper.residual_parameters()):
                torch.testing.assert_close(actual, expected)

    def test_fast_mapper_preserves_trained_keys_and_values_across_blocks(self):
        from transformers import DynamicCache

        from kvverify.continuation_pilot import MultiDonorMapper
        from kvverify.fast_affine import map_multi_donor_fast
        from kvverify.hf_backend import legacy

        runtime, torch = self.runtime, self.torch
        base = self.fixture.fit()
        base.manifest["source_geometry"] = [2, 8, 3]
        mapper = MultiDonorMapper(base, rank=4).eval()
        source, _ = self.fixture.synthetic(37)
        extra, _ = self.fixture.synthetic(37)
        rows = (*legacy(source), legacy(extra)[0])
        runtime.target.model.rotary_emb.inv_freq.mul_(0.73)
        with torch.inference_mode():
            mapper.up.normal_(std=0.04)
            for dtype in (torch.float32, torch.bfloat16):
                runtime.source.to(dtype)
                runtime.target.to(dtype)
                cached = DynamicCache.from_legacy_cache(tuple((k.to(dtype), v.to(dtype)) for k, v in rows))
                original = tuple((k.clone(), v.clone()) for k, v in legacy(cached))
                expected = legacy(mapper.apply(cached, runtime.source, runtime.target, block_tokens=11))
                for block_tokens in (7, 64):
                    actual = legacy(map_multi_donor_fast(mapper, cached, runtime.source, runtime.target,
                                                         block_tokens=block_tokens))
                    for pair, reference in zip(actual, expected):
                        for tensor, reference_tensor in zip(pair, reference):
                            self.assertEqual(tensor.dtype, dtype)
                            self.assertTrue(tensor.is_contiguous())
                            torch.testing.assert_close(tensor, reference_tensor, rtol=2e-5, atol=2e-6)
                for pair, reference in zip(legacy(cached), original):
                    for tensor, reference_tensor in zip(pair, reference):
                        torch.testing.assert_close(tensor, reference_tensor, rtol=0, atol=0)

    def test_shared_cache_scores_both_questions_with_one_mapping(self):
        from unittest.mock import patch

        from kvverify.continuation_pilot import MultiDonorMapper
        from kvverify.evidence_replay import EvidenceReplay
        from kvverify.fast_affine import map_multi_donor_fast
        from kvverify.shared_cache_replay import choose_option, shared_option_logits

        runtime, torch = self.runtime, self.torch
        mapper = MultiDonorMapper(self.fixture.fit(), rank=4).eval()
        prefix = [4, 10, 11, 15] * 72
        first_suffix, second_suffix = [16, 17], [7, 19, 20, 22]
        evidence = EvidenceReplay([], len(prefix), second_suffix, ())

        def select(output):
            return output.logits[0, -1, :4].clone()

        with torch.inference_mode():
            mapper.up.normal_(std=0.04)
            source = runtime.forward(runtime.source, prefix).past_key_values
            unshared = [shared_option_logits(runtime, mapper, source, prefix, suffix, select)[0]
                        for suffix in (first_suffix, second_suffix)]
            with patch("kvverify.fast_affine.map_multi_donor_fast", wraps=map_multi_donor_fast) as mapping:
                shared = shared_option_logits(runtime, mapper, source, prefix, first_suffix, select, evidence)
            self.assertEqual(mapping.call_count, 1)
            self.assertEqual(source.get_seq_length(), len(prefix))
            for actual, expected in zip(shared, unshared):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(choose_option({"A": 1001., "B": 1000.}, {"A": 1., "B": 5.}), "B")

    def test_teacher_trace_records_each_generated_token_including_eos(self):
        from kvverify.continuation_pilot import decode

        torch = self.torch
        emitted = iter((4, 5, 6, 1))
        calls = []

        def forward(_model, ids, cache):
            calls.append(ids)
            logits = torch.zeros(1, 1, 10)
            logits[0, 0, next(emitted)] = 10.
            return SimpleNamespace(logits=logits, past_key_values=object())

        runtime = SimpleNamespace(forward=forward, tokenizer=SimpleNamespace(
            decode=lambda tokens, skip_special_tokens: " ".join(str(t) for t in tokens if t != 1)))
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 2]))
        trace = decode(runtime, model, [7, 8], store_distributions=True)
        self.assertEqual(trace.tokens, (4, 5, 6, 1))
        self.assertEqual(trace.text, "4 5 6")
        self.assertTrue(trace.ended)
        self.assertEqual(calls, [[7, 8], [4], [5], [6]])
        self.assertEqual(trace.log_probs.argmax(-1).tolist(), list(trace.tokens))
