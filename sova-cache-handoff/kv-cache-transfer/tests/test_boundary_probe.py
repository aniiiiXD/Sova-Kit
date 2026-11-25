"""Small offline tail-boundary controls; no pretrained weights or GPU work."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from kvverify.boundary_probe import PROTOCOL, select_examples, run_probe
from kvverify.data import Example


def records(documents=20, questions=3):
    return [Example(f'e-{doc:02d}-{question}', f'doc-{doc:02d}', f'Independent context number {doc}.',
                    f'Question number {question}?', {'A': 'alpha', 'B': 'beta', 'C': 'gamma'}, 'A', 'eval')
            for doc in range(documents) for question in range(questions)]


class BoundarySelectionTests(unittest.TestCase):
    def test_sixteen_distinct_documents_selected_before_any_outputs(self):
        examples = records()
        selected = select_examples(examples)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({example.document_id for example in selected}), 16)
        self.assertEqual([e.id for e in selected], [e.id for e in select_examples(list(reversed(examples)))])

    def test_empty_duplicate_or_training_inputs_rejected(self):
        example = records(1, 1)[0]
        for examples in ([], [example, example], [replace(example, split='calibration')]):
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                select_examples(examples)

    def test_conflicting_document_contexts_rejected(self):
        first, second = records(1, 2)
        with self.assertRaisesRegex(ValueError, 'different contexts'):
            select_examples([first, replace(second, context='Different document')])


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Explicit tiny CPU smoke opt-in')
class ProbabilityEquivalenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_identical_full_vocabulary_distribution_passes(self):
        from kvverify.boundary_probe import probability_equivalence
        logits = self.torch.tensor([3., -1., 0., .1], dtype=self.torch.float64)
        result = probability_equivalence(self.torch, logits, logits.clone())
        self.assertTrue(result['passed'])
        self.assertEqual(result['total_variation'], 0)
        self.assertEqual(result['native_to_candidate_kl_nats'], 0)
        self.assertEqual(result['candidate_to_native_kl_nats'], 0)
        self.assertFalse(result['raw_logit_allclose_used'])

    def test_additive_logit_shift_is_probability_invariant(self):
        from kvverify.boundary_probe import probability_equivalence
        logits = self.torch.tensor([2., -.5, .25, -4.], dtype=self.torch.float64)
        shifted = logits+100.
        self.assertFalse(self.torch.allclose(logits, shifted, rtol=.02, atol=.1))
        result = probability_equivalence(self.torch, logits, shifted)
        self.assertTrue(result['passed'])
        self.assertLess(result['total_variation'], 1e-12)

    def test_same_argmax_but_meaningfully_changed_probabilities_fails(self):
        from kvverify.boundary_probe import probability_equivalence
        result = probability_equivalence(self.torch, [4., 1., 0.], [.2, .1, 0.])
        self.assertTrue(result['first_token_equal'])
        self.assertGreater(result['total_variation'], 1e-3)
        self.assertFalse(result['passed'])

    def test_changed_first_token_fails(self):
        from kvverify.boundary_probe import probability_equivalence
        result = probability_equivalence(self.torch, [1., 0.], [0., 1.])
        self.assertFalse(result['first_token_equal'])
        self.assertFalse(result['passed'])

    def test_low_probability_raw_logit_roundoff_is_not_a_distribution_failure(self):
        from kvverify.boundary_probe import probability_equivalence
        left = self.torch.tensor([40., 0., -1.], dtype=self.torch.float64)
        right = self.torch.tensor([40., .15, -1.], dtype=self.torch.float64)
        self.assertFalse(self.torch.allclose(left, right, rtol=.02, atol=.1))
        result = probability_equivalence(self.torch, left, right)
        self.assertTrue(result['passed'])
        self.assertLess(result['max_probability_difference'], 1e-12)

    def test_malformed_nonfinite_or_mismatched_vectors_are_rejected(self):
        from kvverify.boundary_probe import probability_equivalence
        for left, right in (([], []), ([[1., 2.]], [[1., 2.]]), ([1., 2.], [1.]),
                             ([float('nan'), 0.], [1., 0.]), ([1., 0.], [float('inf'), 0.])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                probability_equivalence(self.torch, left, right)

    def test_only_bounded_negative_kl_roundoff_can_be_clipped(self):
        from kvverify.boundary_probe import _clip_divergence_roundoff
        self.assertEqual(_clip_divergence_roundoff(-1e-12), (0., True))
        self.assertEqual(_clip_divergence_roundoff(0.), (0., False))
        self.assertEqual(_clip_divergence_roundoff(.1), (.1, False))
        for value in (-1.0001e-12, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _clip_divergence_roundoff(value)

    def test_input_tensors_and_autograd_state_are_unchanged(self):
        from kvverify.boundary_probe import probability_equivalence
        left = self.torch.tensor([2., 1., 0.], requires_grad=True)
        before = left.detach().clone()
        result = probability_equivalence(self.torch, left, left+1.)
        self.assertTrue(result['passed'])
        self.torch.testing.assert_close(left, before, rtol=0, atol=0)
        self.assertTrue(left.requires_grad)
        self.assertIsNone(left.grad)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Explicit tiny CPU smoke opt-in')
class BoundaryRuntimeTests(unittest.TestCase):
    def setUp(self):
        from test_hf_optional import TinyHFTests, TinyTokenizer
        class Letters(TinyTokenizer):
            def __call__(self, text, **kwargs):
                if text in ('A', 'B', 'C'):
                    return {'input_ids': [{'A': 1, 'B': 2, 'C': 3}[text]]}
                return super().__call__(text, **kwargs)
            def decode(self, ids, **kwargs):
                return ''.join({1: 'A', 2: 'B', 3: 'C'}.get(i, 'x') for i in ids)
        fixture = TinyHFTests()
        fixture.setUp()
        self.torch = fixture.torch
        self.runtime = fixture.runtime
        self.runtime.tokenizer = Letters()
        self.runtime.tokenizer_sha256 = self.runtime.tokenizer_fingerprint(self.runtime.tokenizer)
        self.runtime.mapper = self.runtime.identity_mapper()
        self.runtime.mapper.manifest.update(
            source_sha256=self.runtime.identities['source']['sha256'],
            target_sha256=self.runtime.identities['target']['sha256'],
            tokenizer_sha256=self.runtime.tokenizer_sha256,
            calibration_ids=[], calibration_documents=[], calibration_context_hashes=[])
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)/'probe'
        self.addCleanup(self.temporary.cleanup)

    def traces(self):
        return [json.loads(line) for line in (self.output/'traces.jsonl').read_text().splitlines()]

    def test_identity_cache_suffix_one_eight_thirtytwo_matches_native(self):
        from safetensors.torch import load_file
        result = run_probe(self.runtime, records(2, 2), self.output)
        rows = self.traces()
        self.assertEqual(len(rows), 14)
        self.assertEqual(set(result['arms']), {'native', 'transfer_suffix_1', 'transfer_suffix_8', 'transfer_suffix_32',
                                             'oracle_suffix_1', 'oracle_suffix_8', 'oracle_suffix_32'})
        self.assertEqual(result['oracle_gate']['status'], 'passed')
        for row in rows:
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['protocol'], PROTOCOL)
            self.assertTrue(row['native_first_token_agreement'])
            self.assertLess(row['native_to_candidate_kl_nats'], 1e-5)
            self.assertLess(row['native_max_logit_abs_difference'], 1e-5)
            self.assertEqual(row['gpu_peak_memory'], {})
            self.assertTrue((self.output/row['first_logits_file']).is_file())
            logits = load_file(self.output/row['first_logits_file'])['first_logits']
            self.assertEqual(logits.numel(), 48)
            if row['arm'] != 'native':
                self.assertEqual(row['target_prefill_tokens'], row['tail_tokens'])
                self.assertEqual(len(row['target_tail_token_ids']), row['tail_tokens'])
                self.assertEqual(row['imported_tokens'], row['prompt_tokens']-row['tail_tokens'])
                self.assertEqual(row['target_start_position'], row['imported_tokens'])
            if row['arm'].startswith('oracle_'):
                self.assertTrue(row['oracle_matches_native'])
                self.assertFalse(row['semantic_transfer'])
                self.assertEqual(row['cache_origin'], 'native_target_prefix')
                self.assertEqual(row['source_prefill_tokens'], 0)
                self.assertEqual(row['source_prefill_ms'], 0)
                self.assertEqual(row['map_ms'], 0)
                self.assertEqual(row['target_total_prefill_tokens'], row['prompt_tokens'])
        config = json.loads((self.output/'config.json').read_text())
        self.assertTrue(config['original_protocol_preserved'])
        self.assertEqual(len(config['selected_ids']), 2)

    def test_over_limit_records_are_retained_for_every_boundary(self):
        self.runtime.prompt_token_limit = 3
        result = run_probe(self.runtime, records(2, 1), self.output)
        self.assertEqual(len(self.traces()), 14)
        self.assertEqual(result['oracle_gate']['status'], 'not_applicable')
        for summary in result['arms'].values():
            self.assertEqual(summary['attempts'], 2)
            self.assertEqual(summary['over_limit'], 2)
            self.assertEqual(summary['returned'], 0)

    def test_wrong_import_geometry_is_an_error_without_native_fallback(self):
        from kvverify.hf_backend import legacy
        mapper = self.runtime.mapper
        runtime = self.runtime
        class Broken:
            manifest = mapper.manifest
            def apply(self, cache, source, target):
                rows = legacy(cache)
                return runtime.DynamicCache.from_legacy_cache(rows[:-1])
        runtime.mapper = Broken()
        result = run_probe(runtime, records(1, 1), self.output)
        self.assertEqual(result['arms']['native']['returned'], 1)
        for row in self.traces():
            if row['arm'].startswith('transfer_'):
                self.assertEqual(row['status'], 'error')
                self.assertFalse(row['fallback'])
                self.assertIn('boundary', row['error'])
        self.assertEqual(result['oracle_gate']['status'], 'passed')

    def test_mapper_identity_or_training_overlap_rejected_before_execution(self):
        example = records(1, 1)[0]
        self.runtime.mapper.manifest['source_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            run_probe(self.runtime, [example], self.output)
        self.assertFalse(self.output.exists())
        self.runtime.mapper.manifest['source_sha256'] = self.runtime.identities['source']['sha256']
        self.runtime.mapper.manifest['calibration_documents'] = [example.document_id]
        with self.assertRaisesRegex(ValueError, 'leaks'):
            run_probe(self.runtime, [example], self.output)

    def test_prompt_exactly_at_declared_cap_is_eligible(self):
        example = records(1, 1)[0]
        self.runtime.prompt_token_limit = len(self.runtime.tokenize(example))
        result = run_probe(self.runtime, [example], self.output, tails=(1,))
        self.assertEqual(result['arms']['native']['returned'], 1)
        self.assertEqual(result['arms']['transfer_suffix_1']['returned'], 1)

    def test_reports_cannot_be_overwritten(self):
        run_probe(self.runtime, records(1, 1), self.output, tails=(1,))
        with self.assertRaises(FileExistsError):
            run_probe(self.runtime, records(1, 1), self.output, tails=(1,))

    def test_native_oracle_at_longer_positions_matches_without_semantic_transfer(self):
        example = replace(records(1, 1)[0], context='An extended native prefix control. '*16)
        self.assertGreater(len(self.runtime.tokenize(example)), 512)
        result = run_probe(self.runtime, [example], self.output, oracle_rtol=1e-4, oracle_atol=1e-5)
        self.assertEqual(result['oracle_gate']['status'], 'passed')
        for row in self.traces():
            if row['arm'].startswith('oracle_'):
                self.assertGreater(row['target_start_position'], 480)
                self.assertLess(row['native_max_logit_abs_difference'], 1e-5)
                self.assertTrue(row['native_first_token_agreement'])
                self.assertIsNone(row['source_checkpoint_sha256'])

    def test_failed_oracle_blocks_all_mapped_execution_and_fails_gate(self):
        from unittest.mock import patch
        from kvverify.boundary_probe import _attempt
        def fail_oracle(runtime, ids, tail, base):
            if base['arm'] == 'oracle_suffix_8':
                raise RuntimeError('Injected native cache failure')
            return _attempt(runtime, ids, tail, base)
        with patch('kvverify.boundary_probe._attempt', side_effect=fail_oracle):
            result = run_probe(self.runtime, records(1, 1), self.output)
        self.assertEqual(result['oracle_gate']['status'], 'failed')
        self.assertEqual(result['acceptance'], 'not_established')
        self.assertEqual(len(self.traces()), 7)
        for row in self.traces():
            if row['arm'].startswith('transfer_'):
                self.assertEqual(row['status'], 'error')
                self.assertTrue(row['blocked_by_oracle_gate'])
                self.assertFalse(row['model_execution_started'])
                self.assertNotIn('first_logits_file', row)


if __name__ == '__main__':
    unittest.main()
