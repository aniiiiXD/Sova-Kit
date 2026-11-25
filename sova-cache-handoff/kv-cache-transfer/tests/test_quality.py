import math
import json
from pathlib import Path
import tempfile
import unittest

from kvverify.quality import generation_metrics, option_token_ids, score_option_logits
from cloud.prepare_data import prepare


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(text)-65]}

    def decode(self, ids, skip_special_tokens=False):
        return chr(ids[0]+65)


class QualityTests(unittest.TestCase):
    def test_native_label_description_is_not_silently_strict_correct(self):
        result = generation_metrics('A. Entailment', 'A', {'A': 'yes', 'B': 'no'}, 'length')
        self.assertFalse(result['free_generation_strict_correct'])
        self.assertTrue(result['free_generation_invalid_format'])
        self.assertTrue(result['free_generation_prefix_diagnostic_correct'])
        self.assertTrue(result['free_generation_truncated'])

    def test_prose_word_is_not_a_label_prefix(self):
        result = generation_metrics('Apple is useful', 'A', ['A', 'B'])
        self.assertIsNone(result['free_generation_prefix_diagnostic'])

    def test_exact_letter_and_finish_reason_are_separate(self):
        result = generation_metrics(' A\n', 'A', ['A', 'B'], 'length')
        self.assertTrue(result['free_generation_strict_correct'])
        self.assertTrue(result['free_generation_truncated'])

    def test_constrained_decision_exposes_low_unconditional_option_mass(self):
        result = score_option_logits([1., 0., 10.], {'A': 0, 'B': 1}, 'A')
        self.assertTrue(result['decision_correct'])
        self.assertFalse(result['full_vocab_top_token_is_option'])
        self.assertLess(result['option_probability_mass'], .001)
        self.assertAlmostEqual(sum(result['option_probabilities'].values()), 1.)
        self.assertAlmostEqual(math.exp(result['option_log_probabilities']['A']),
                               result['option_probabilities']['A'])

    def test_ties_are_deterministic_sorted_labels(self):
        result = score_option_logits([0., 0.], {'B': 1, 'A': 0}, 'B')
        self.assertEqual(result['decision_answer'], 'A')
        self.assertFalse(result['decision_correct'])

    def test_duplicate_or_out_of_range_ids_rejected(self):
        for ids in ({'A': 0, 'B': 0}, {'A': 0, 'B': 3}, {'A': -1, 'B': 0}):
            with self.assertRaises(ValueError):
                score_option_logits([0., 0.], ids, 'A')

    def test_nonfinite_distribution_rejected(self):
        with self.assertRaises(ValueError):
            score_option_logits([0., float('nan')], {'A': 0, 'B': 1}, 'A')

    def test_option_tokens_checked(self):
        self.assertEqual(option_token_ids(Tokenizer(), ['A', 'B']), {'A': 0, 'B': 1})
        class SplitTokenizer(Tokenizer):
            def __call__(self, *args, **kwargs):
                return {'input_ids': [0, 1]}
        with self.assertRaises(ValueError):
            option_token_ids(SplitTokenizer(), ['A', 'B'])


class PreparedDataTests(unittest.TestCase):
    def fixture(self, root):
        for directory in ('documents', 'decisions'):
            (root/directory).mkdir()
        for name in ('README.md', 'LICENSE', 'TERMS'):
            (root/name).write_text('Synthetic fixture, no real legal document')
        for split, ids in [('train_clean', ['1', '2', '3']), ('dev', ['4', '5'])]:
            official = 'train' if split == 'train_clean' else 'dev'
            documents = [{'document_id': i, 'text': 'Unique document '+i, 'official_split': official} for i in ids]
            decisions = [{'document_id': i, 'decision_id': official+':'+i, 'gold_label': 'Entailment',
                          'statement': 'Statement', 'official_split': official} for i in ids]
            for kind, rows in [('documents', documents), ('decisions', decisions)]:
                (root/kind/(split+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        # This is deliberately invalid. Any accidental test parsing would fail.
        (root/'documents/test.jsonl').write_text('DO NOT READ')
        (root/'decisions/test.jsonl').write_text('DO NOT READ')

    def test_document_split_hashes_and_test_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            source.mkdir()
            self.fixture(source)
            output = root/'prepared'
            result = prepare(source, output, 2, 1, 19)
            self.assertEqual(result['splits']['train']['documents'], 2)
            self.assertEqual(result['splits']['mapper_validation']['documents'], 1)
            self.assertEqual(result['splits']['dev']['examples'], 2)
            self.assertEqual(result['splits']['train']['record_split'], 'calibration')
            self.assertEqual(result['splits']['dev']['record_split'], 'eval')
            self.assertFalse(any('test' in name for name in result['source_files']))
            self.assertFalse(any('test' in name for name in result['files']))
            self.assertEqual((source/'documents/test.jsonl').read_text(), 'DO NOT READ')
            self.assertEqual(result['files']['train.jsonl']['sha256'], result['splits']['train']['sha256'])
            with self.assertRaises(ValueError):
                prepare(source, output, 2, 1, 19)

    def test_train_dev_context_leakage_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            source.mkdir()
            self.fixture(source)
            dev = source/'documents/dev.jsonl'
            rows = [json.loads(x) for x in dev.read_text().splitlines()]
            rows[0]['text'] = 'Unique document 1'
            dev.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):
                prepare(source, root/'prepared', 2, 1, 19)
            self.assertFalse((root/'prepared').exists())


if __name__ == '__main__':
    unittest.main()
