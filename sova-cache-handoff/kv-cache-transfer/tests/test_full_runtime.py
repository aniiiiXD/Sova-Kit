"""Tiny offline integration of prospective metrics with real cache execution."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE') == '1', 'Explicit tiny CPU smoke opt-in')
class FullRuntimeTests(unittest.TestCase):
    def setUp(self):
        import test_hf_optional as fixture
        self.fixture = fixture.TinyHFTests()
        self.fixture.setUp()
        class Letters(fixture.TinyTokenizer):
            def __call__(self, text, **kwargs):
                if text in ('A','B','C'):
                    return {'input_ids':[{'A':1,'B':2,'C':3}[text]]}
                return super().__call__(text,**kwargs)
            def decode(self, ids, **kwargs):
                return ''.join({1:'A',2:'B',3:'C'}.get(i,'x') for i in ids)
        self.runtime = self.fixture.runtime
        self.runtime.tokenizer = Letters()
        self.runtime.prompt_token_limit = 1024
        self.runtime.mapper = self.runtime.identity_mapper()

    def test_observer_metrics_do_not_replace_trace_score(self):
        from kvverify.full_eval import install_observer
        from kvverify.smoke import example
        from kvverify.evidence import validate_trace
        install_observer(self.runtime)
        row = self.runtime.answer(example(), 'transfer_cold',0,'observer-test',2)
        validate_trace(row)
        self.assertIn(row['decision_answer'],('A','B'))
        self.assertAlmostEqual(sum(row['option_probabilities'].values()),1.)
        self.assertTrue(0 <= row['option_probability_mass'] <= 1)
        self.assertEqual(row['correct'],row['answer']==example().answer)
        self.assertIn('free_generation_truncated',row)

    def test_observer_cannot_change_evidence(self):
        from kvverify.smoke import example
        self.runtime.quality_observer = lambda *args:{'correct':True}
        with self.assertRaisesRegex(ValueError,'overwrite'):
            self.runtime.answer(example(),'native_cold',0,'forbidden-observer',1)

    def test_prompt_cap_retains_over_limit_outcome(self):
        from kvverify.smoke import example
        self.runtime.prompt_token_limit = 3
        row = self.runtime.answer(example(),'native_cold',0,'cap-test',1)
        self.assertEqual(row['status'],'over_limit')
        self.assertFalse(row['correct'])

    def test_four_variant_end_to_end_quality_report(self):
        from kvverify.full_eval import run_quality
        from kvverify.smoke import example
        rows = [example('qa-1','doc-1','independent alpha'),
                example('qa-2','doc-2','different beta')]
        with tempfile.TemporaryDirectory() as temporary:
            result = run_quality(self.runtime,rows,
                {'ridge':self.runtime.mapper,'neural':self.runtime.mapper},Path(temporary)/'quality',
                freegen_documents=2,freegen_max_new_tokens=2)
            self.assertEqual(result['paired_decision_neural']['documents'],2)
            for variant in ('source','native','ridge','neural'):
                self.assertEqual(result['phases']['decision'][variant]['examples'],2)
                self.assertIsNone(result['phases']['decision'][variant]['free_generation_strict_accuracy'])
                self.assertEqual(result['phases']['free_generation'][variant]['examples'],2)


if __name__ == '__main__':
    unittest.main()
