"""Explicit opt-in: tiny random CPU Qwen3, no network/checkpoint downloads."""
import os
from pathlib import Path
import tempfile
import unittest

ENABLED = os.environ.get("KVVERIFY_TORCH_SMOKE") == "1"


class TinyTokenizer:
    special_tokens_map = {}
    def get_vocab(self):
        return {str(i): i for i in range(48)}
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 45 + 1 for c in text]}
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)
    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


@unittest.skipUnless(ENABLED, "Set KVVERIFY_TORCH_SMOKE=1 for the separately scoped tiny CPU model tests")
class TinyHFTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=1024, rope_theta=10000., attention_dropout=0.,
                            use_sliding_window=False, eos_token_id=47)
        model = Qwen3ForCausalLM(config).float().eval()
        self.runtime = DenseRuntime(model, model, TinyTokenizer())
        self.ids = [1, 5, 7, 12, 2, 9, 3, 14]

    def test_identity_cache_preserves_real_qwen_logits(self):
        r = self.runtime
        prefix = r.forward(r.source, self.ids[:-1]).past_key_values
        expected = r.forward(r.target, self.ids[-1:], r.clone_cache(prefix)).logits
        with self.torch.inference_mode():
            transferred = r.identity_mapper().apply(prefix, r.source, r.target)
        self.assertEqual(transferred.get_seq_length(), len(self.ids)-1)
        actual = r.forward(r.target, self.ids[-1:], transferred).logits
        self.torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    def test_zero_cache_changes_logits(self):
        from transformers import DynamicCache
        from kvverify.hf_backend import legacy
        r = self.runtime
        prefix = r.forward(r.source, self.ids[:-1]).past_key_values
        expected = r.forward(r.target, self.ids[-1:], r.clone_cache(prefix)).logits
        zero = DynamicCache.from_legacy_cache(tuple((self.torch.zeros_like(k), self.torch.zeros_like(v)) for k, v in legacy(prefix)))
        actual = r.forward(r.target, self.ids[-1:], zero).logits
        self.assertGreater((actual-expected).abs().max().item(), 1e-5)

    def test_all_five_real_adapter_arms_emit_valid_traces(self):
        from kvverify.evidence import ARMS, validate_trace
        from kvverify.smoke import example
        r = self.runtime
        r.mapper = r.identity_mapper()
        for arm in ARMS:
            row = r.answer(example(), arm, 0, "tiny-random-test-config", max_new_tokens=2)
            validate_trace(row)
            self.assertEqual(row["status"], "ok")
            if arm.startswith("transfer"):
                self.assertEqual(row["target_prefill_tokens"], 1)
                self.assertEqual(row["imported_tokens"], row["prompt_tokens"]-1)

    def test_tiny_calibration_save_load_and_tamper_detection(self):
        from kvverify.hf_backend import RidgeMapper
        from kvverify.smoke import example
        r = self.runtime
        examples = [example("cal-1", "doc-1", "independent alpha", "calibration"),
                    example("cal-2", "doc-2", "independent beta", "calibration")]
        mapper = r.fit(examples, prefix_tokens=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mapper"
            mapper.save(path)
            loaded = RidgeMapper.load(path, "random-smoke-source", "random-smoke-target", r.tokenizer_sha256)
            prefix = r.forward(r.source, self.ids[:-1]).past_key_values
            with self.torch.inference_mode():
                mapped = loaded.apply(prefix, r.source, r.target)
            self.assertEqual(mapped.get_seq_length(), len(self.ids)-1)
            weights = path/"weights.safetensors"
            weights.write_bytes(weights.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                RidgeMapper.load(path, "random-smoke-source", "random-smoke-target", r.tokenizer_sha256)

    def test_checkpoint_mismatch_fails_before_loading_weights(self):
        import json
        from kvverify.hf_backend import RidgeMapper
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"manifest.json").write_text(json.dumps({"format_version": 1, "source_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                RidgeMapper.load(path, "expected-source", "expected-target", "tokenizer")
