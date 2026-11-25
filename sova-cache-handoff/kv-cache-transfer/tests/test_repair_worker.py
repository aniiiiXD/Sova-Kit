from dataclasses import replace
from types import SimpleNamespace
import unittest
import os
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from cloud.repair_worker import split_train_articles
from cloud import repair_worker


class RepairPartitionTests(unittest.TestCase):
    def records(self):
        return [SimpleNamespace(id=f'q-{i}',document_id=f'article-{i%150}',
            context_hash=f'exact-context-{i%150}',split='calibration')for i in range(2523)]

    def test_partition_preserves_every_question_and_article_isolation(self):
        original=self.records()
        train,val=split_train_articles(original)
        self.assertEqual(len(train)+len(val),2523)
        self.assertEqual(len({e.document_id for e in train}),120)
        self.assertEqual(len({e.document_id for e in val}),30)
        self.assertFalse({e.document_id for e in train}&{e.document_id for e in val})
        self.assertEqual({e.id for e in train+val},{e.id for e in original})

    def test_partition_does_not_depend_on_input_question_order(self):
        train,val=split_train_articles(self.records())
        other_train,other_val=split_train_articles(list(reversed(self.records())))
        self.assertEqual({e.id for e in train},{e.id for e in other_train})
        self.assertEqual({e.id for e in val},{e.id for e in other_val})

    def test_evaluation_records_and_incomplete_input_cannot_become_training(self):
        rows=self.records();rows[0].split='eval'
        with self.assertRaises(ValueError):split_train_articles(rows)
        with self.assertRaises(ValueError):split_train_articles(self.records()[:-1])


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny CPU smoke opt-in')
class FreshOracleControlTests(unittest.TestCase):
    def test_bases_are_frozen_before_offload_storage_classification(self):
        import torch
        from kvverify.activation_offload import _SavedTensorPolicy
        source=torch.nn.Linear(3,4);target=torch.nn.Linear(3,4);mapper=torch.nn.Linear(3,4)
        runtime=SimpleNamespace(source=source,target=target,mapper=mapper)
        for parameter in source.parameters():parameter.grad=torch.ones_like(parameter)
        repair_worker.freeze_base_models(runtime)
        policy=_SavedTensorPolicy(torch,[source,target])
        self.assertTrue(policy.is_frozen_weight_storage(target.weight.t()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for m in(source,target)for p in m.parameters()))
        self.assertTrue(all(p.requires_grad for p in mapper.parameters()))

    def runtime(self,corrupt=False):
        import torch
        class Tokenizer:
            def __call__(self,text,**kwargs):return {'input_ids':[0 if text=='A'else 1]}
            def decode(self,ids,**kwargs):return 'A'if ids==[0]else'B'
        class Runtime:
            tokenizer=Tokenizer();prompt_token_limit=20
            def __init__(self):self.torch=torch;self.calls=[]
            def tokenize(self,e):return [1,2,3,4]
            def forward(self,model,ids,cache=None):
                self.calls.append((list(ids),cache))
                logits=[1.,0.]if corrupt and cache else[4.,-4.]
                return SimpleNamespace(logits=torch.tensor([[logits]]),past_key_values='native-cache')
        r=Runtime();r.target=object();return r

    def control(self,runtime):
        e=SimpleNamespace(id='q',document_id='doc',answer='A',choices={'A':'one','B':'two'})
        with patch('kvverify.qa_training.select_mapper_validation',return_value=([e],'fixed')),\
             patch('kvverify.qa_training.split_document_prompt',return_value=([1,2,3,4],[1,2],[3,4])),\
             patch.dict(repair_worker.CONFIG,{'minimum_native_control_accuracy':.35}):
            return repair_worker.native_control(runtime,[e],count=1)

    def test_real_operational_suffix_uses_native_prefix_and_passes(self):
        runtime=self.runtime();control=self.control(runtime)
        self.assertTrue(control['oracle_gate_passed'])
        self.assertEqual(runtime.calls,[([1,2,3,4],None),([1,2],None),([3,4],'native-cache')])

    def test_corrupted_cache_distribution_fails_even_with_correct_answer(self):
        control=self.control(self.runtime(corrupt=True))
        self.assertEqual(control['decision_accuracy'],1.)
        self.assertFalse(control['oracle_gate_passed'])

    def test_verified_attention_path_covers_entire_worker_and_restores_patch(self):
        import torch
        from torch.nn import functional as F
        original=F.scaled_dot_product_attention
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'config.json'
            path.write_text(json.dumps({'dtype':'float32','attention_backend':'math','query_block_size':256}))
            def configured_main():
                self.assertIsNot(F.scaled_dot_product_attention,original)
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                return 'worker-completed'
            with patch.dict(os.environ,{'KV_CONFIG':str(path)}),patch.object(repair_worker,'main',side_effect=configured_main):
                self.assertEqual(repair_worker.run_configured(),'worker-completed')
            self.assertIs(F.scaled_dot_product_attention,original)
            path.write_text(json.dumps({'dtype':'float16','attention_backend':'math','query_block_size':256}))
            with patch.dict(os.environ,{'KV_CONFIG':str(path)}),patch.object(repair_worker,'main')as main:
                with self.assertRaisesRegex(ValueError,'Unverified'):repair_worker.run_configured()
                main.assert_not_called()


if __name__=='__main__':unittest.main()
