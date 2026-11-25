"""Synthetic boundaries and opt-in tiny random-model QA-distillation checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from kvverify.data import Example
from kvverify.qa_training import DocumentQuestionSampler,select_mapper_validation,split_document_prompt


class OffsetTokenizer:
    is_fast=True
    special_tokens_map={}
    def get_vocab(self):return {str(i):i for i in range(48)}
    def __call__(self,text,**kwargs):
        result={'input_ids':[ord(c)%45+1 for c in text]}
        if kwargs.get('return_offsets_mapping'):
            result['offset_mapping']=[(i,i+1) for i in range(len(text))]
        return result
    def apply_chat_template(self,messages,**kwargs):return '<u>'+messages[0]['content']+'</u><a>'
    def decode(self,ids,**kwargs):return ' '.join(str(i) for i in ids)


class MergedOffsetTokenizer(OffsetTokenizer):
    def __call__(self,text,**kwargs):
        offsets=[(i,min(i+2,len(text))) for i in range(0,len(text),2)]
        result={'input_ids':[sum(ord(c) for c in text[a:b])%45+1 for a,b in offsets]}
        if kwargs.get('return_offsets_mapping'):result['offset_mapping']=offsets
        return result


def fixture(identifier='doc-1:q1',document='doc-1',context='one complete article context',question='Which choice?'):
    return Example(identifier,document,context,question,{'A':'alpha','B':'beta'},'A','calibration')


class BoundaryRuntime:
    def __init__(self,tokenizer=None):self.tokenizer=tokenizer or OffsetTokenizer()
    def tokenize(self,example):
        text=self.tokenizer.apply_chat_template([{'role':'user','content':example.prompt()}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer(text,add_special_tokens=False)['input_ids']


class QABoundarySamplingTests(unittest.TestCase):
    def test_exact_document_boundary_preserves_full_prompt(self):
        r=BoundaryRuntime()
        e=fixture()
        ids,prefix,suffix=split_document_prompt(r,e)
        self.assertEqual(ids,r.tokenize(e))
        self.assertEqual(ids,prefix+suffix)
        self.assertEqual(len(prefix),len('<u>Document:\n'+e.context))
        self.assertGreater(len(suffix),len(e.question))

    def test_straddling_bpe_token_is_retained_in_native_suffix(self):
        r=BoundaryRuntime(MergedOffsetTokenizer())
        e=fixture(context='ab')
        ids,prefix,suffix=split_document_prompt(r,e)
        boundary=len('<u>Document:\n'+e.context)
        self.assertEqual(boundary%2,1)
        self.assertEqual(len(prefix),boundary//2)
        self.assertEqual(ids,prefix+suffix)
        separate_prefix=r.tokenizer('<u>Document:\n'+e.context)['input_ids']
        self.assertNotEqual(prefix,separate_prefix)

    def test_boundary_refuses_missing_offsets_and_token_mismatch(self):
        r=BoundaryRuntime()
        r.tokenizer.is_fast=False
        with self.assertRaisesRegex(ValueError,'fast-tokenizer'):
            split_document_prompt(r,fixture())
        r=BoundaryRuntime()
        r.tokenize=lambda e:[99]
        with self.assertRaisesRegex(ValueError,'token IDs differ'):
            split_document_prompt(r,fixture())

    def test_document_uniform_sampler_cycles_all_questions_before_reuse(self):
        examples=[fixture(f'{d}:q{q}',f'doc-{d}',f'article-{d}',f'question-{q}')
                  for d in range(2) for q in range(3)]
        sampler=DocumentQuestionSampler(examples,seed=19)
        selected=[sampler.next() for _ in range(6)]
        self.assertEqual(len({e.id for e in selected}),6)
        repeated=DocumentQuestionSampler(list(reversed(examples)),seed=19)
        self.assertEqual([e.id for e in selected],[repeated.next().id for _ in range(6)])
        for offset in (0,2,4):self.assertEqual(len({e.document_id for e in selected[offset:offset+2]}),2)

    def test_validation_covers_shared_hypotheses_and_generic_articles(self):
        repeated=[fixture(f'{d}:q{q}',f'doc-{d}',f'article-{d}',f'hypothesis-{q}')
                  for d in range(3) for q in range(17)]
        selected,policy=select_mapper_validation(repeated)
        self.assertEqual(len(selected),17)
        self.assertEqual(len({e.question for e in selected}),17)
        self.assertEqual(len({e.document_id for e in selected}),3)
        unique=[replace(e,question=e.id+' unique wording') for e in repeated]
        selected,policy=select_mapper_validation(unique)
        self.assertEqual(len(selected),3)
        self.assertIn('article',policy)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny Torch smoke opt-in required')
class TinyQATrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        torch.set_num_threads(1)
        torch.manual_seed(91)
        def model():
            c=Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=1024,
                rope_theta=10000.,attention_dropout=0.,use_sliding_window=False,eos_token_id=47)
            return Qwen3ForCausalLM(c).float().eval()
        self.torch=torch
        self.runtime=DenseRuntime(model(),model(),OffsetTokenizer())
        for m in (self.runtime.source,self.runtime.target):
            for p in m.parameters():p.requires_grad_(False)
        self.train=[fixture(f'train-{d}:q{q}',f'train-doc-{d}',f'unique full train article {d}',f'hypothesis-{q}')
                    for d in range(2) for q in range(3)]
        self.val=[fixture(f'val-{d}:q{q}',f'val-doc-{d}',f'separate validation article {d}',f'hypothesis-{q}')
                  for d in range(2) for q in range(3)]

    def mapper(self):
        from kvverify.train_mapper import NeuralMapper
        r=self.runtime
        unique=[self.train[0],self.train[3]]
        return NeuralMapper.from_ridge(r.fit(unique,prefix_tokens=16),r.target,rank=3)

    def test_native_identity_handoff_matches_teacher_answer_positions(self):
        from kvverify.hf_backend import DenseRuntime
        from kvverify.qa_training import qa_training_loss
        r=DenseRuntime(self.runtime.target,self.runtime.target,OffsetTokenizer())
        ids,prefix,suffix=split_document_prompt(r,self.train[0])
        loss,metrics=qa_training_loss(r,r.identity_mapper(),ids,prefix,suffix,
            answer_tokens=3,checkpoint_mapper=False)
        self.assertLess(abs(metrics['kl_loss']),1e-5)
        self.assertEqual(metrics['native_query_tokens'],len(suffix))
        self.assertEqual(metrics['teacher_answer_tokens'],len(metrics['teacher_token_ids']))
        self.assertFalse(metrics['gold_answer_used'])
        self.assertGreater(metrics['ce_loss'],0.)

    def test_answer_objective_backpropagates_without_gold_or_base_gradients(self):
        from kvverify.qa_training import qa_training_loss
        r=self.runtime
        mapper=self.mapper()
        ids,prefix,suffix=split_document_prompt(r,self.train[0])
        loss,metrics=qa_training_loss(r,mapper,ids,prefix,suffix,answer_tokens=2,
            kv_weight=0.,checkpoint_mapper=True)
        loss.backward()
        self.assertGreater(float(mapper.up.grad.abs().sum()),0.)
        self.assertGreater(float(mapper.affine_weight.grad.abs().sum()),0.)
        self.assertEqual(metrics['kv_loss'],0.)
        self.assertFalse(metrics['gold_answer_used'])
        self.assertTrue(all(p.grad is None for m in (r.source,r.target) for p in m.parameters()))

    def test_checkpointed_mapper_has_matching_values_and_parameter_gradients(self):
        from kvverify.qa_training import checkpointed_mapper_cache
        from kvverify.hf_backend import legacy
        r=self.runtime
        mapper=self.mapper()
        source=r.forward(r.source,[1,3,5,7,9]).past_key_values
        caches=[]
        gradients=[]
        for enabled in (False,True):
            mapper.zero_grad(set_to_none=True)
            cache=checkpointed_mapper_cache(mapper,source,r.source,r.target,enabled)
            rows=legacy(cache)
            caches.append([(k.detach().clone(),v.detach().clone()) for k,v in rows])
            loss=sum(k.square().mean()+v.square().mean() for k,v in rows)
            loss.backward()
            gradients.append({n:p.grad.detach().clone() for n,p in mapper.named_parameters() if p.grad is not None})
        for left,right in zip(caches[0],caches[1]):
            for a,b in zip(left,right):self.torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        self.assertEqual(set(gradients[0]),set(gradients[1]))
        for name in gradients[0]:
            self.torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=1e-5,atol=1e-6)

    def test_layer_token_checkpoints_match_full_function_with_absolute_rope_offsets(self):
        from kvverify.hf_backend import legacy
        r=self.runtime
        mapper=self.mapper()
        with self.torch.no_grad():
            mapper.up.normal_(std=.03)
            mapper.up_bias.normal_(std=.02)
        source=r.forward(r.source,[i%44+1 for i in range(23)]).past_key_values
        original=mapper._map_layer_token_block
        observed=[]
        def record(layer,offset,needed,tensors,*models):
            observed.append((layer,offset,[t.shape[-2] for t in tensors],len(needed)))
            return original(layer,offset,needed,tensors,*models)
        mapper._map_layer_token_block=record
        values=[]
        gradients=[]
        for bounded in (False,True):
            mapper.zero_grad(set_to_none=True)
            cache=(mapper.apply_checkpointed(source,r.source,r.target,token_block_size=7)
                   if bounded else mapper.apply(source,r.source,r.target))
            rows=legacy(cache)
            values.append([(k.detach().clone(),v.detach().clone()) for k,v in rows])
            loss=sum(k.square().mean()+v.square().mean() for k,v in rows)
            loss.backward()
            gradients.append({n:p.grad.detach().clone() for n,p in mapper.named_parameters() if p.grad is not None})
        self.assertTrue(observed)
        self.assertTrue(all(max(lengths)<=7 and count<=3 for _,_,lengths,count in observed))
        self.assertIn(21,{offset for _,offset,_,_ in observed})
        for left,right in zip(values[0],values[1]):
            for a,b in zip(left,right):self.torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        self.assertEqual(set(gradients[0]),set(gradients[1]))
        for name in gradients[0]:
            self.torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=2e-5,atol=2e-6)

    def test_memory_preflight_has_no_optimizer_updates_or_base_weight_changes(self):
        from kvverify.qa_training import train_qa_mapper
        r=self.runtime
        notices=[]
        before={id(p):p.detach().clone() for m in (r.source,r.target) for p in m.parameters()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qa'
            train_qa_mapper(r,self.train,self.val,path,steps=1,max_prompt_tokens=600,
                answer_tokens=2,rank=2,checkpoint_every=1,max_seconds=30,
                memory_preflight=True,callback=notices.append)
            preflight=json.loads((path/'memory-preflight.json').read_text())
            self.assertEqual(preflight['status'],'passed')
            self.assertEqual(preflight['optimizer_updates'],0)
            self.assertTrue(all(p['mapper_parameters_unchanged'] and p['gradients_finite'] for p in preflight['probes']))
            self.assertTrue(any(row.get('phase')=='memory_preflight' for row in notices))
            for m in (r.source,r.target):
                for p in m.parameters():self.torch.testing.assert_close(p,before[id(p)],rtol=0,atol=0)

    def test_reused_step_zero_initialization_skips_ridge_fit_and_preserves_initial_values(self):
        from unittest.mock import patch
        from kvverify.qa_training import train_qa_mapper
        from kvverify.train_mapper import NeuralMapper
        r=self.runtime
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'first'
            train_qa_mapper(r,self.train,self.val,first,steps=1,max_prompt_tokens=600,
                answer_tokens=2,rank=2,checkpoint_every=1,max_seconds=30)
            original=first/'checkpoints/step-000000'
            initial=NeuralMapper.load(original,'random-smoke-source','random-smoke-target',r.tokenizer_sha256)
            expected={n:p.detach().clone() for n,p in initial.named_parameters()}
            second=Path(directory)/'second'
            with (patch.object(r,'fit',side_effect=AssertionError('Reuse must not refit ridge')),
                  patch.object(NeuralMapper,'from_ridge',side_effect=AssertionError('Reuse must not reinitialize mapper'))):
                train_qa_mapper(r,self.train,self.val,second,steps=1,max_prompt_tokens=600,
                    answer_tokens=2,rank=2,checkpoint_every=1,max_seconds=30,
                    memory_preflight=True,initial_mapper=initial)
            saved=NeuralMapper.load(second/'checkpoints/step-000000','random-smoke-source','random-smoke-target',r.tokenizer_sha256)
            for name,p in saved.named_parameters():self.torch.testing.assert_close(p,expected[name],rtol=0,atol=0)
            protocol=json.loads((second/'qa-protocol.json').read_text())
            self.assertIn('Exact reused',protocol['initialization'])
            self.assertEqual(protocol['initial_mapper_weights_sha256'],json.loads((original/'manifest.json').read_text())['weights_sha256'])

    def test_reused_initialization_rejects_trained_changed_data_and_question_registry(self):
        from unittest.mock import patch
        from kvverify.qa_training import train_qa_mapper
        from kvverify.train_mapper import NeuralMapper
        r=self.runtime
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'first'
            train_qa_mapper(r,self.train,self.val,first,steps=1,max_prompt_tokens=600,
                answer_tokens=2,rank=2,checkpoint_every=1,max_seconds=30)
            def initial():
                return NeuralMapper.load(first/'checkpoints/step-000000','random-smoke-source','random-smoke-target',r.tokenizer_sha256)
            cases=[('trained','step-zero'),('data','input examples changed'),('registry','eligible QA/source registry'),('rank','rank or KV geometry')]
            with patch.object(r,'fit',side_effect=AssertionError('Rejected reuse must not refit ridge')):
                for name,message in cases:
                    mapper=initial()
                    examples=self.train
                    rank=2
                    if name=='trained':mapper.manifest['trained_steps']=1
                    elif name=='data':examples=[replace(self.train[0],question='changed input question'),*self.train[1:]]
                    elif name=='registry':mapper.manifest['training_question_ids']=mapper.manifest['training_question_ids'][:-1]
                    elif name=='rank':rank=3
                    with self.subTest(name=name),self.assertRaisesRegex(ValueError,message):
                        train_qa_mapper(r,examples,self.val,Path(directory)/name,steps=1,
                            max_prompt_tokens=600,answer_tokens=2,rank=rank,checkpoint_every=1,
                            max_seconds=30,initial_mapper=mapper)

    def test_short_training_records_all_question_ids_and_frozen_best_mapper(self):
        from kvverify.qa_training import train_qa_mapper
        from kvverify.train_mapper import NeuralMapper
        from kvverify.data import check_heldout
        r=self.runtime
        before={id(p):p.detach().clone() for m in (r.source,r.target) for p in m.parameters()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qa'
            mapper=train_qa_mapper(r,self.train,self.val,path,steps=6,max_prompt_tokens=600,
                answer_tokens=2,rank=3,checkpoint_every=3,max_seconds=30)
            summary=json.loads((path/'training-summary.json').read_text())
            self.assertEqual(summary['completed_updates'],6)
            self.assertEqual(summary['training_questions_available'],6)
            self.assertEqual(summary['training_questions_consumed'],6)
            self.assertEqual(summary['mapper_validation_questions'],3)
            self.assertTrue(summary['base_models_frozen'])
            consumed=[json.loads(line) for line in (path/'question-consumption.jsonl').read_text().splitlines()]
            self.assertEqual({row['question_id'] for row in consumed},{e.id for e in self.train})
            self.assertTrue(all(not row['gold_answer_used'] for row in consumed))
            loaded=NeuralMapper.load(path/'best','random-smoke-source','random-smoke-target',r.tokenizer_sha256)
            with self.assertRaisesRegex(ValueError,'leaks'):check_heldout(loaded.manifest,self.val)
            for m in (r.source,r.target):
                for p in m.parameters():self.torch.testing.assert_close(p,before[id(p)],rtol=0,atol=0)

    def test_full_prompt_cap_excludes_without_cropping(self):
        from kvverify.qa_training import train_qa_mapper
        r=self.runtime
        long=fixture('too-long','long-doc','x'*700,'full question')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qa'
            train_qa_mapper(r,self.train+[long],self.val,path,steps=1,max_prompt_tokens=600,
                answer_tokens=1,rank=2,checkpoint_every=1,max_seconds=30)
            protocol=json.loads((path/'qa-protocol.json').read_text())
            self.assertEqual(protocol['eligible_training_questions'],6)
            self.assertEqual(protocol['excluded'][0]['id'],'too-long')
            self.assertIn('no cropping',protocol['excluded'][0]['reason'])


if __name__=='__main__':unittest.main()
