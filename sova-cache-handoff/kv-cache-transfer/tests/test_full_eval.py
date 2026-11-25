import unittest

from kvverify.full_eval import VARIANTS, summarize_quality


def fixture(phase='decision'):
    rows = []
    for variant in VARIANTS:
        for i, (document, gold) in enumerate([('doc-1', 'A'), ('doc-1', 'A'), ('doc-2', 'B')]):
            rows.append({'id': str(i), 'document_id': document, 'context_hash': document,
                'prompt_hash': 'prompt-'+str(i), 'run_config_hash': 'config', 'repeat': 0,
                'arm': 'native_cold' if variant == 'native' else 'transfer_cold',
                'phase': phase, 'variant': variant, 'status': 'ok', 'gold_answer': gold,
                'decision_correct': True, 'free_generation_strict_correct': False,
                'free_generation_format_valid': False, 'free_generation_truncated': True,
                'option_probability_mass': .3, 'ttft_ms': 5.})
    return rows


class FullEvalTests(unittest.TestCase):
    def test_errors_and_over_limit_remain_in_all_denominators(self):
        rows = fixture()
        for row in rows:
            if row['id'] == '1':
                row['status'] = 'error'
            elif row['id'] == '2':
                row['status'] = 'over_limit'
        # Deliberately stale true correctness must not turn an error into success.
        result = summarize_quality(rows)
        native = result['phases']['decision']['native']
        self.assertEqual(native['examples'], 3)
        self.assertEqual(native['returned'], 1)
        self.assertEqual(native['errors'], 1)
        self.assertEqual(native['over_limit'], 1)
        self.assertAlmostEqual(native['decision_accuracy'], 1/3)
        self.assertEqual(native['per_class']['A']['decision_accuracy'], .5)
        self.assertEqual(native['per_class']['B']['decision_accuracy'], 0.)
        self.assertEqual(native['macro_decision_accuracy'], .25)
        self.assertAlmostEqual(native['majority_class_baseline_accuracy'], 2/3)

    def test_one_token_decisions_are_not_free_generation_compliance(self):
        result = summarize_quality(fixture())['phases']['decision']['native']
        self.assertEqual(result['decision_accuracy'], 1.)
        self.assertIsNone(result['free_generation_strict_accuracy'])
        self.assertIsNone(result['format_valid_count'])

    def test_free_generation_failure_is_separate_from_correct_decision(self):
        result = summarize_quality(fixture('free_generation'))['phases']['free_generation']['native']
        self.assertEqual(result['decision_accuracy'], 1.)
        self.assertEqual(result['free_generation_strict_accuracy'], 0.)
        self.assertEqual(result['truncated_count'], 3)
        self.assertEqual(result['format_valid_count'], 0)

    def test_paired_document_counts_do_not_multiply_hypotheses(self):
        rows = fixture()
        for row in rows:
            if row['variant'] == 'neural' and row['id'] == '0':
                row['decision_correct'] = False
        paired = summarize_quality(rows)['paired_decision_neural']
        self.assertEqual(paired['documents'], 2)
        self.assertEqual(paired['examples'], 3)
        self.assertAlmostEqual(paired['difference'], -1/3)

    def test_duplicate_or_repeated_quality_attempts_rejected(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            summarize_quality(rows+[dict(rows[0])])
        rows[0]['repeat'] = 1
        with self.assertRaises(ValueError):
            summarize_quality(rows)

    def test_missing_error_row_or_wrong_gold_rejected(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            summarize_quality(rows[:-1])
        rows[0]['gold_answer'] = 'B'
        with self.assertRaises(ValueError):
            summarize_quality(rows)
        for invalid in (None, 1, '', 'AA'):
            rows = fixture()
            rows[0]['gold_answer'] = invalid
            with self.assertRaises(ValueError):
                summarize_quality(rows)


if __name__ == '__main__':
    unittest.main()
