"""Opt-in tiny random CPU tests of learned cache training, never pretrained runs."""
import json
import os
from pathlib import Path
import tempfile
import unittest


class FakeGradScaler:
    """CPU implementation of GradScaler's public skip/backoff contract."""
    def __init__(self, scale=1024., enabled=True):
        self.value, self.enabled = scale, enabled
        self.found_inf = False
        self.skipped, self.updated = 0, 0
    def is_enabled(self):
        return self.enabled
    def get_scale(self):
        return self.value
    def scale(self, loss):
        return loss*self.value
    def unscale_(self, optimizer):
        import torch
        gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        for gradient in gradients:
            gradient.div_(self.value)
        self.found_inf = any(not torch.isfinite(gradient).all() for gradient in gradients)
    def step(self, optimizer):
        if self.found_inf:
            self.skipped += 1
        else:
            self.updated += 1
            optimizer.step()
    def update(self, new_scale=None):
        if new_scale is not None:
            self.value = float(new_scale)
        elif self.found_inf:
            self.value /= 2
        self.found_inf = False


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Explicit tiny Torch smoke opt-in required')
class NeuralMapperTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        from test_hf_optional import TinyTokenizer
        torch.set_num_threads(1)
        torch.manual_seed(43)
        def model(layers):
            config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=48,
                num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=1024, rope_theta=10000.,
                attention_dropout=0., use_sliding_window=False, eos_token_id=47)
            return Qwen3ForCausalLM(config).float().eval()
        self.runtime = DenseRuntime(model(2), model(3), TinyTokenizer())
        self.torch = torch

    def examples(self):
        from kvverify.smoke import example
        return ([example('train-a', 'train-doc-a', 'independent first training context', 'calibration'),
                 example('train-b', 'train-doc-b', 'different second training context', 'calibration')],
                [example('val-a', 'validation-doc', 'held out mapper selection context', 'calibration')])

    def test_tail_kl_backpropagates_to_mapper_with_frozen_lms(self):
        from kvverify.train_mapper import NeuralMapper, training_loss
        train, _ = self.examples()
        r = self.runtime
        ridge = r.fit(train, prefix_tokens=16)
        mapper = NeuralMapper.from_ridge(ridge, r.target, rank=3)
        for model in (r.source, r.target):
            for parameter in model.parameters(): parameter.requires_grad_(False)
        # Disable KV reconstruction to prove the frozen target's prediction
        # loss, by itself, reaches mapper parameters through imported caches.
        loss, metrics = training_loss(r, mapper, [1, 5, 3, 7, 8, 2, 9, 4], tail_tokens=2, kv_weight=0.)
        loss.backward()
        self.assertGreater(float(mapper.affine_weight.grad.abs().sum()), 0)
        self.assertGreater(float(mapper.up.grad.abs().sum()), 0)
        self.assertGreaterEqual(metrics['kl_loss'], -1e-6)
        self.assertTrue(all(p.grad is None for model in (r.source, r.target) for p in model.parameters()))

    def test_training_serialization_identity_and_tamper_checks(self):
        from kvverify.train_mapper import NeuralMapper, train_mapper
        train, validation = self.examples()
        r = self.runtime
        before = {id(p): p.detach().clone() for model in (r.source, r.target) for p in model.parameters()}
        notices = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'training'
            mapper = train_mapper(r, train, validation, output, steps=3, prefix_tokens=16,
                                  tail_tokens=2, rank=3, checkpoint_every=2, callback=notices.append)
            summary = json.loads((output/'training-summary.json').read_text())
            self.assertEqual(summary['completed_steps'], 3)
            self.assertTrue(summary['base_models_frozen'])
            self.assertEqual(summary['training_documents'], 2)
            self.assertGreaterEqual(len(notices), 3)
            updates = [json.loads(line) for line in (output/'training-log.jsonl').read_text().splitlines()
                       if 'grad_norm' in json.loads(line)]
            self.assertEqual(len(updates), 3)
            self.assertTrue(all(row['gradients_finite'] for row in updates))
            self.assertTrue(all(row['loss_scale'] == 1. for row in updates))  # CPU scaler disabled.
            for model in (r.source, r.target):
                for parameter in model.parameters(): self.torch.testing.assert_close(parameter, before[id(parameter)], rtol=0, atol=0)
            loaded = NeuralMapper.load(output/'best', 'random-smoke-source', 'random-smoke-target', r.tokenizer_sha256)
            prefix = r.forward(r.source, [1, 4, 6, 8, 9]).past_key_values
            mapped = loaded.apply(prefix, r.source, r.target)
            self.assertEqual(mapped.get_seq_length(), 5)
            loss, _ = __import__('kvverify.train_mapper', fromlist=['training_loss']).training_loss(r, loaded, [1, 2, 3, 4, 5], tail_tokens=2)
            loss.backward()  # Inference-mode source tensors must not poison training.
            weights = output/'best/weights.safetensors'
            weights.write_bytes(weights.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                NeuralMapper.load(output/'best', 'random-smoke-source', 'random-smoke-target', r.tokenizer_sha256)

    def test_training_refuses_eval_or_leaking_validation(self):
        from dataclasses import replace
        from kvverify.train_mapper import train_mapper
        train, validation = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'calibration'):
                train_mapper(self.runtime, train, [replace(validation[0], split='eval')], Path(directory)/'bad')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                train_mapper(self.runtime, train, train, Path(directory)/'leak')

    def test_half_cache_overflow_is_rejected_after_cast(self):
        from kvverify.train_mapper import NeuralMapper
        train, _ = self.examples()
        r = self.runtime
        mapper = NeuralMapper.from_ridge(r.fit(train, prefix_tokens=8), r.target, rank=2)
        prefix = r.forward(r.source, [1, 4, 7, 2]).past_key_values
        r.target.half()
        with self.torch.no_grad():
            mapper.affine_weight[1].fill_(1e20)  # Finite FP32, overflows target FP16.
        with self.assertRaisesRegex(ValueError, 'Nonfinite mapped cache'):
            mapper.apply(prefix, r.source, r.target)

    def test_best_registry_includes_validation_documents_and_unique_train_docs(self):
        from dataclasses import replace
        from kvverify.train_mapper import train_mapper
        from kvverify.data import check_heldout
        train, validation = self.examples()
        train.append(replace(train[0], id='another-question-same-document'))
        with tempfile.TemporaryDirectory() as directory:
            mapper = train_mapper(self.runtime, train, validation, Path(directory)/'training',
                                  steps=1, prefix_tokens=8, tail_tokens=2, rank=2, checkpoint_every=1)
            self.assertEqual(len(mapper.manifest['training_documents']), 2)
            with self.assertRaisesRegex(ValueError, 'leaks'):
                check_heldout(mapper.manifest, validation)

    def test_overflow_replays_same_loss_and_updates_optimizer_exactly_once(self):
        from kvverify.train_mapper import bounded_optimizer_update
        torch = self.torch
        mapper = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): mapper.weight.zero_()
        optimizer = torch.optim.AdamW(mapper.parameters(), lr=.1, weight_decay=.1)
        scaler = FakeGradScaler()
        calls, overflow = [], []
        def loss_fn():
            calls.append(88)  # Captured document/window identity is unchanged.
            loss = (mapper.weight-1).square().sum()
            if len(calls) <= 2:
                loss.register_hook(lambda gradient: gradient*float('inf'))
            return loss, {'loss': float(loss.detach()), 'window': 88}
        def notice(event):
            self.assertEqual(float(mapper.weight.detach()), 0.)
            self.assertFalse(optimizer.state)  # No Adam state or weight decay on skips.
            overflow.append(event)
        metrics = bounded_optimizer_update(loss_fn, mapper, optimizer, scaler, on_overflow=notice)
        self.assertEqual(calls, [88, 88, 88])
        self.assertEqual((scaler.skipped, scaler.updated), (2, 1))
        self.assertEqual(metrics['optimizer_attempts'], 3)
        self.assertEqual(metrics['overflow_attempts'], 2)
        self.assertEqual(metrics['loss_scale'], 256.)
        self.assertEqual([event['loss_scale_after'] for event in overflow], [512., 256.])
        self.assertGreater(float(mapper.weight.detach()), 0.)
        self.assertTrue(all(event['optimizer_updated'] is False for event in overflow))

    def test_persistent_overflow_stops_at_retry_limit_without_parameter_update(self):
        from kvverify.train_mapper import bounded_optimizer_update, MapperGradientOverflow
        torch = self.torch
        mapper = torch.nn.Linear(1, 1, bias=False)
        before = mapper.weight.detach().clone()
        optimizer = torch.optim.AdamW(mapper.parameters(), lr=.1)
        scaler, notices = FakeGradScaler(), []
        def loss_fn():
            loss = mapper.weight.square().sum()
            loss.register_hook(lambda gradient: gradient*float('inf'))
            return loss, {'loss': float(loss.detach())}
        with self.assertRaises(MapperGradientOverflow) as raised:
            bounded_optimizer_update(loss_fn, mapper, optimizer, scaler, max_attempts=3, on_overflow=notices.append)
        self.assertEqual(raised.exception.details['reason'], 'nonfinite_gradients_retry_limit')
        self.assertEqual((scaler.skipped, scaler.updated), (3, 0))
        self.assertEqual(len(notices), 3)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(mapper.weight, before, rtol=0, atol=0)

    def test_overflow_at_minimum_scale_fails_without_subunit_scale(self):
        from kvverify.train_mapper import bounded_optimizer_update, MapperGradientOverflow
        torch = self.torch
        mapper = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(mapper.parameters(), lr=.1)
        scaler = FakeGradScaler(scale=1.)
        def loss_fn():
            loss = mapper.weight.square().sum()
            loss.register_hook(lambda gradient: gradient*float('inf'))
            return loss, {'loss': float(loss.detach())}
        with self.assertRaises(MapperGradientOverflow):
            bounded_optimizer_update(loss_fn, mapper, optimizer, scaler)
        self.assertEqual(scaler.get_scale(), 1.)
        self.assertEqual((scaler.skipped, scaler.updated), (1, 0))

    def test_finite_gradient_norm_failure_never_calls_scaler_step(self):
        from unittest.mock import patch
        from kvverify.train_mapper import bounded_optimizer_update, MapperGradientOverflow
        torch = self.torch
        mapper = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(mapper.parameters(), lr=.1)
        scaler = FakeGradScaler()
        def loss_fn():
            loss = mapper.weight.square().sum()
            return loss, {'loss': float(loss.detach())}
        with patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(float('inf'))):
            with self.assertRaises(MapperGradientOverflow) as raised:
                bounded_optimizer_update(loss_fn, mapper, optimizer, scaler)
        self.assertEqual(raised.exception.details['reason'], 'nonfinite_norm_with_finite_gradients')
        self.assertEqual((scaler.skipped, scaler.updated), (0, 0))
        self.assertFalse(optimizer.state)

    def test_retry_checks_deadline_before_computing_loss(self):
        from kvverify.train_mapper import bounded_optimizer_update
        torch = self.torch
        mapper = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(mapper.parameters(), lr=.1)
        def loss_fn():
            self.fail('Expired run must not evaluate another forward/backward')
        with self.assertRaises(TimeoutError):
            bounded_optimizer_update(loss_fn, mapper, optimizer, FakeGradScaler(), deadline=0.)


if __name__ == '__main__':
    unittest.main()
