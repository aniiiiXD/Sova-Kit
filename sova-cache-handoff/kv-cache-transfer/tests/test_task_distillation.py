"""Causal alignment and actual mapper-only gradients on offline tiny Qwen."""
import os
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Tiny explicit Torch tests')
class QuestionReadoutTests(unittest.TestCase):
    def setUp(self):
        from test_continuation_training import ContinuationLossTests
        ContinuationLossTests.setUp(self)

    def loss(self, count=0):
        from kvverify.task_distillation import question_readout_loss
        return question_readout_loss(self.r, self.m, list(range(1, 11)),
                                     list(range(1, 7)), list(range(7, 11)),
                                     native_prefix_tokens=count)

    def test_identity_full_and_partial_native_handoffs_match_the_native_readout(self):
        for count in (0, 2):
            with self.subTest(native_prefix_tokens=count):
                loss, metrics = self.loss(count)
                self.assertLess(abs(float(loss)), 1e-6)
                self.assertEqual(metrics['supervised_tokens'], 1)
                self.assertEqual(metrics['mapped_prefix_tokens'], 6-count)
                self.assertEqual(metrics['native_suffix_tokens'], 4)
                self.assertFalse(metrics['gold_qa_used'])

    def test_perturbed_mapper_has_finite_nonzero_gradients_and_frozen_bases(self):
        with self.t.no_grad(): self.m.bias.add_(.3)
        loss, _ = self.loss(2)
        loss.backward()
        self.assertGreater(float(loss), 0.)
        self.assertTrue(all(p.grad is not None and self.t.isfinite(p.grad).all()
                            for p in self.m.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.m.parameters()), 0.)
        self.assertTrue(all(p.grad is None for p in self.r.target.parameters()))

    def test_invalid_boundary_and_full_native_bypass_are_rejected(self):
        from kvverify.task_distillation import question_readout_loss
        with self.assertRaisesRegex(ValueError, 'handoff'):
            question_readout_loss(self.r, self.m, [1, 2, 9], [1, 2], [3])
        with self.assertRaisesRegex(ValueError, 'leave translated'):
            self.loss(6)


if __name__ == '__main__': unittest.main()
