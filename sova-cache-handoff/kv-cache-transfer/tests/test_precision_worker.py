"""Frozen-input safety plus a real tiny-Qwen numerical control, offline."""
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cloud import precision_worker as worker
from cloud import recovery_submit as submit


class PrecisionSelectionTests(unittest.TestCase):
    def records(self):
        return [SimpleNamespace(id=f'q-{i}',document_id=f'article-{i%150}',
            context_hash=f'context-{i%150}',split='calibration',answer='A')for i in range(2523)]

    def prior(self,records):
        from cloud.repair_worker import split_train_articles
        _,val=split_train_articles(records)
        return {'rows':[{'id':e.id,'document_id':e.document_id,'gold_answer':e.answer}for e in val[:16]]}

    def test_exact_ids_and_order_preserved_even_when_input_order_changes(self):
        data=self.records();prior=self.prior(data)
        result=worker.frozen_examples(list(reversed(data)),prior)
        self.assertEqual([e.id for e in result],[r['id']for r in prior['rows']])

    def test_duplicate_partial_or_identity_changed_inputs_fail(self):
        for corrupt in ('duplicate','partial','document','answer','unknown'):
            data=self.records();prior=self.prior(data)
            if corrupt=='duplicate':prior['rows'][0]=prior['rows'][1]
            if corrupt=='partial':prior['rows'].pop()
            if corrupt=='document':prior['rows'][0]['document_id']='different'
            if corrupt=='answer':prior['rows'][0]['gold_answer']='B'
            if corrupt=='unknown':prior['rows'][0]['id']='outside-validation'
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):worker.frozen_examples(data,prior)

    def test_precision_has_no_mapper_channel_or_training_and_bounded_time(self):
        cfg=submit.config('precision',1200)
        req=submit.request('test-precision',cfg)
        self.assertFalse(cfg['training_enabled'])
        self.assertEqual([r['ChannelName']for r in req['InputDataConfig']],['code'])
        self.assertEqual(req['StoppingCondition']['MaxRuntimeInSeconds'],1200)
        self.assertIn('cloud/precision_worker.py',submit.source_files('precision'))
        self.assertIn('cloud/repair_worker.py',submit.source_files('precision'))
        with self.assertRaises(ValueError):submit.config('precision',1201)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny Torch smoke opt-in')
class PrecisionRuntimeTests(unittest.TestCase):
    def setUp(self):
        from test_hf_optional import TinyHFTests
        from test_qa_training import OffsetTokenizer,fixture
        helper=TinyHFTests();helper.setUp()
        self.runtime=helper.runtime;self.runtime.tokenizer=OffsetTokenizer()
        self.runtime.prompt_token_limit=1024;self.example=fixture()

    def test_real_qwen_segmented_logits_match_full_prompt(self):
        row,full,segmented=worker.check_example(self.runtime,self.example)
        self.assertTrue(row['oracle']['passed'])
        self.assertEqual(row['prompt_tokens'],row['prefix_tokens']+row['native_suffix_tokens'])
        self.runtime.torch.testing.assert_close(full,segmented,rtol=1e-4,atol=1e-6)

    def test_wrong_cache_position_cannot_pass_on_same_logits(self):
        original=self.runtime.forward
        def wrong(model,ids,cache=None):
            out=original(model,ids,cache)
            if cache is None:out.past_key_values.get_seq_length=lambda:0
            return out
        with patch.object(self.runtime,'forward',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'incorrect length'):
                worker.check_example(self.runtime,self.example)

    def test_over_limit_fails_without_silently_truncating(self):
        self.runtime.prompt_token_limit=3
        with self.assertRaisesRegex(ValueError,'unchanged prompt limit'):
            worker.check_example(self.runtime,self.example)


if __name__=='__main__':unittest.main()
