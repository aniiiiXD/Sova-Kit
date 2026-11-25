"""Opt-in tiny CPU checks; no model download, CUDA execution or cloud jobs."""
from contextlib import nullcontext
import os
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1',
                     'Explicit tiny Torch smoke opt-in required')
class SelectiveActivationOffloadTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_detached_transposed_and_sliced_frozen_weight_storage_is_kept(self):
        from kvverify.activation_offload import _SavedTensorPolicy
        torch = self.torch
        model = torch.nn.Linear(7, 5)
        model.requires_grad_(False)
        policy = _SavedTensorPolicy(torch, [model], offload_device_types=('cpu',))
        for tensor in (model.weight, model.weight.detach().T, model.weight[:, 1:4]):
            self.assertTrue(policy.is_frozen_weight_storage(tensor))
            record = policy.pack(tensor)
            self.assertEqual(record[0], 'resident')
            self.assertEqual(record[3].untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
            torch.testing.assert_close(policy.unpack(record), tensor)
        self.assertEqual(policy.stats['copied_bytes'], 0)
        self.assertEqual(policy.stats['kept_weight_tensors'], 3)
        self.assertFalse(policy.is_frozen_weight_storage(model.weight.detach().clone()))

    def test_trainable_parameter_storage_is_not_misclassified_as_frozen(self):
        from kvverify.activation_offload import _SavedTensorPolicy
        model = self.torch.nn.Linear(3, 2)
        policy = _SavedTensorPolicy(self.torch, [model], offload_device_types=('cpu',))
        self.assertFalse(policy.is_frozen_weight_storage(model.weight.T))
        self.assertEqual(policy.pack(model.weight.T)[0], 'cpu')

    def test_cpu_default_noop_preserves_storage_values_dtype_and_no_grad_wrapper(self):
        from kvverify.activation_offload import _SavedTensorPolicy
        torch = self.torch
        value = torch.randn(4, 3, dtype=torch.float64, requires_grad=True).T
        policy = _SavedTensorPolicy(torch, [])
        record = policy.pack(value)
        restored = policy.unpack(record)
        self.assertEqual(restored.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())
        self.assertFalse(restored.requires_grad)
        self.assertEqual(restored.dtype, value.dtype)
        torch.testing.assert_close(restored, value, rtol=0, atol=0)
        self.assertTrue(value.requires_grad)
        self.assertEqual(policy.stats['copied_bytes'], 0)

    def test_emulated_pageable_activation_copy_and_restore_preserves_values_and_counters(self):
        from kvverify.activation_offload import _SavedTensorPolicy
        torch = self.torch
        value = torch.randn(3, 5, dtype=torch.float64, requires_grad=True).T
        policy = _SavedTensorPolicy(torch, [], offload_device_types=('cpu',))
        record = policy.pack(value)
        restored = policy.unpack(record)
        self.assertEqual(record[0], 'cpu')
        self.assertNotEqual(restored.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())
        torch.testing.assert_close(restored, value, rtol=0, atol=0)
        size = value.numel() * value.element_size()
        self.assertEqual(policy.stats['copied_bytes'], size)
        self.assertEqual(policy.stats['restored_bytes'], size)
        self.assertFalse(record[3].is_pinned())

    def test_emulated_offload_preserves_differentiable_loss_and_gradients(self):
        from kvverify.activation_offload import _SavedTensorPolicy
        torch = self.torch
        base = torch.nn.Linear(4, 3).requires_grad_(False)
        mapper = torch.nn.Linear(2, 4)
        inputs = torch.randn(5, 2)
        reference = None
        for offload in (False, True):
            mapper.zero_grad(set_to_none=True)
            policy = _SavedTensorPolicy(torch, [base], offload_device_types=('cpu',))
            with torch.autograd.graph.saved_tensors_hooks(policy.pack, policy.unpack) if offload else nullcontext():
                loss = base(mapper(inputs)).square().mean()
            loss.backward()
            result = (loss.detach().clone(), {n:p.grad.detach().clone() for n,p in mapper.named_parameters()})
            if reference is None: reference = result
            else:
                torch.testing.assert_close(result[0], reference[0], rtol=0, atol=0)
                for name in reference[1]:
                    torch.testing.assert_close(result[1][name], reference[1][name], rtol=0, atol=0)
                self.assertGreater(policy.stats['kept_weight_bytes'], 0)
                self.assertGreater(policy.stats['copied_bytes'], 0)
            self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_tiny_qwen_loss_and_all_mapper_gradients_match(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from kvverify.activation_offload import selective_saved_tensors_cpu
        from kvverify.hf_backend import DenseRuntime
        from kvverify.qa_training import qa_training_loss, split_document_prompt
        from kvverify.train_mapper import NeuralMapper
        from test_qa_training import OffsetTokenizer, fixture
        torch = self.torch
        def model():
            config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=1024, rope_theta=10000.,
                attention_dropout=0., use_sliding_window=False, eos_token_id=47)
            return Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
        runtime = DenseRuntime(model(), model(), OffsetTokenizer())
        example = fixture()
        mapper = NeuralMapper.from_ridge(runtime.fit([example], prefix_tokens=16), runtime.target, rank=3)
        with torch.no_grad(): mapper.up.normal_(std=.02)
        ids, prefix, suffix = split_document_prompt(runtime, example)
        reference = None
        for offload in (False, True):
            mapper.zero_grad(set_to_none=True)
            with selective_saved_tensors_cpu([runtime.source, runtime.target]) if offload else nullcontext():
                loss, _ = qa_training_loss(runtime, mapper, ids, prefix, suffix,
                    answer_tokens=2, checkpoint_mapper=True)
            loss.backward()
            result = (loss.detach().clone(), {n:p.grad.detach().clone() for n,p in mapper.named_parameters() if p.grad is not None})
            if reference is None: reference = result
            else:
                torch.testing.assert_close(result[0], reference[0], rtol=0, atol=0)
                self.assertEqual(set(reference[1]), set(result[1]))
                for name in reference[1]:
                    torch.testing.assert_close(result[1][name], reference[1][name], rtol=1e-6, atol=1e-7)
            self.assertTrue(all(p.grad is None for m in (runtime.source,runtime.target) for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
