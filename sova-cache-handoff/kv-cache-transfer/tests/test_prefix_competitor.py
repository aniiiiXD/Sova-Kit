"""The cheap competitor genuinely omits context and preserves every outcome."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class PrefixCompetitorTests(unittest.TestCase):
    def test_exact_short_input_uses_native_target_without_mapper(self):
        from kvverify.prefix_competitor import run_prefix_competitor
        from test_qa_training import fixture
        calls=[];target=object()
        class Logits:
            def __getitem__(self,key):return self
            def detach(self):return self
            def float(self):return self
            def cpu(self):return self
        runtime=types.SimpleNamespace(target=target,tokenizer=object(),
            forward=lambda model,ids:(calls.append((model,list(ids)))or types.SimpleNamespace(logits=Logits())))
        with tempfile.TemporaryDirectory()as td,patch('kvverify.qa_training.split_document_prompt',return_value=([1,2,3,4,5],[1,2,3],[4,5])),patch('kvverify.quality.option_token_ids',return_value={}),patch('kvverify.quality.score_option_logits',return_value={'decision_answer':'A','decision_correct':True}):
            result=run_prefix_competitor(runtime,[fixture()],Path(td)/'result',prefix_tokens=2,native_answers={'doc-1:q1':'A'})
        self.assertEqual(calls,[(target,[1,2,4,5])]);self.assertEqual(result['decision_correct'],1.)
        self.assertFalse(result['full_context_read'])

    def test_time_budget_retains_every_question_in_the_denominator(self):
        from kvverify.prefix_competitor import run_prefix_competitor
        from test_qa_training import fixture
        with tempfile.TemporaryDirectory()as td,patch('kvverify.prefix_competitor.time.monotonic',side_effect=[0.,10.,10.]):
            result=run_prefix_competitor(None,[fixture(),fixture('q2')],Path(td)/'result',max_seconds=1)
        self.assertEqual(result['planned_items'],2);self.assertEqual(result['not_run_time_budget'],2)
        self.assertEqual(result['decision_correct'],0.)
