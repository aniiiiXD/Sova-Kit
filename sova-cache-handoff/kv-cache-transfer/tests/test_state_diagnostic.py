"""Outcome-blind selection, causal GQA probes and default-path preservation."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvverify.state_diagnostic import select_panel
from cloud import state_diagnostic_submit as submit


class DiagnosticPlanningTests(unittest.TestCase):
    def test_panel_is_length_stratified_reproducible_and_outcome_blind(self):
        from test_qa_training import fixture
        examples=[fixture(f'q{i}',f'd{i}',f'article{i}')for i in range(30)]
        lengths={e.id:100+i for i,e in enumerate(examples)}
        panel=select_panel(examples,lengths)
        self.assertEqual(len(panel),24)
        self.assertEqual([sum(r['length_band']==b for r in panel)for b in range(3)],[8,8,8])
        changed=[replace(e,answer='B')for e in reversed(examples)]
        self.assertEqual(panel,select_panel(changed,lengths))
        with self.assertRaises(ValueError):select_panel(examples[:-1],lengths)
        with self.assertRaises(ValueError):select_panel(examples,{**lengths,'q0':0})
        duplicate=[replace(e,document_id='d0')for e in examples]
        with self.assertRaises(ValueError):select_panel(duplicate,lengths)

    def test_submission_is_inference_only_has_two_exact_mapper_channels_and_caps(self):
        cfg=submit.configuration(1800,{})
        self.assertFalse(cfg['training_enabled']);self.assertEqual(cfg['dtype'],'float32')
        req=submit.diagnostic_request('sova-kv-state-diag-test',cfg)
        channels={r['ChannelName']:r['DataSource']['S3DataSource']['S3Uri']for r in req['InputDataConfig']}
        self.assertEqual(channels['initializer'],submit.STEP_ZERO_MAPPER)
        self.assertEqual(channels['mapper'],submit.TRAINED_MAPPER)
        self.assertEqual(req['StoppingCondition']['MaxRuntimeInSeconds'],1800)
        self.assertAlmostEqual(submit.cost_guard(1.515,1800,.80),.7575)
        for rate,seconds,cap in[(1.7,1800,.8),(1.515,1801,.8),(1.515,1800,.7),(float('nan'),1800,.8)]:
            with self.assertRaises(ValueError):submit.cost_guard(rate,seconds,cap)
        with self.assertRaises(ValueError):submit.configuration(1801,{})

    def test_exact_checkpoints_and_new_test_evidence_are_required(self):
        for name,step in [('step0',0),('step48',48)]:
            m={'weights_sha256':submit.EXPECTED_WEIGHTS[name],'trained_steps':step,'selected_at_step':step,'format_version':2}
            cfg=submit.checkpoint_config(name,m)
            self.assertEqual(cfg['trained_steps'],step)
            with self.assertRaises(ValueError):submit.checkpoint_config(name,{**m,'trained_steps':1})
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'smoke.json'
            report={'mode':'state_diagnostic','status':'passed','tests_run':1,'failures':0,'errors':0,'skipped':0,
                    'sources_unchanged_during_tests':True,'file_sha256':{'x':'sha'}}
            p.write_text(json.dumps(report))
            with patch.object(submit,'tested_hashes',return_value={'x':'sha'}):submit.verify_tests(p)
            with patch.object(submit,'tested_hashes',return_value={'x':'changed'}):
                with self.assertRaises(ValueError):submit.verify_tests(p)

    def test_worker_manifest_hash_is_canonical_and_corrupted_weights_are_rejected(self):
        from cloud.state_diagnostic_worker import checkpoint_manifests
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);cfg={'checkpoints':{}}
            for name,step,key in [('step0',0,'input_initializer_dir'),('step48',48,'input_mapper_dir')]:
                path=root/name;path.mkdir();weights=b'tiny-fixture-'+name.encode()
                (path/'weights.safetensors').write_bytes(weights)
                sha=hashlib.sha256(weights).hexdigest()
                m={'trained_steps':step,'weights_sha256':sha,'mapper_validation_question_ids':['q']}
                (path/'manifest.json').write_text(json.dumps(m,indent=4)+'\n\n')
                cfg[key]=str(path);cfg['checkpoints'][name]={'trained_steps':step,'weights_sha256':sha,
                    'manifest_sha256':hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
            self.assertEqual(set(checkpoint_manifests(cfg)),{'step0','step48'})
            (root/'step48/weights.safetensors').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'weights'):checkpoint_manifests(cfg)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny Torch smoke opt-in required')
class TinyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        torch.set_num_threads(1);torch.manual_seed(123)
        def model():
            c=Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
                use_sliding_window=False,attention_dropout=0.,eos_token_id=47)
            c._attn_implementation='sdpa'
            return Qwen3ForCausalLM(c).float().eval().requires_grad_(False)
        self.torch=torch;self.runtime=DenseRuntime(model(),model())

    def mapper(self):
        from kvverify.train_mapper import NeuralMapper
        from kvverify.hf_backend import RidgeMapper
        t=self.torch;h,d,l=self.runtime.target_geometry
        ridge=RidgeMapper({'source_geometry':self.runtime.source_geometry,'target_geometry':self.runtime.target_geometry,
            'source_layers':list(range(l))},{'k_weight':t.eye(d).repeat(l,h,1,1),
            'v_weight':t.eye(d).repeat(l,h,1,1),'k_bias':t.zeros(l,h,d),'v_bias':t.zeros(l,h,d)})
        m=NeuralMapper.from_ridge(ridge,self.runtime.target,rank=3,topk=2)
        with t.no_grad():m.key_log_rms.fill_(math.log(2.))
        return m.eval().requires_grad_(False)

    def test_block_staged_mapper_preserves_default_and_rms_only_changes_keys(self):
        from kvverify.state_diagnostic import map_cpu_rows,cpu_rows
        from kvverify.hf_backend import legacy
        t=self.torch;r=self.runtime;m=self.mapper()
        before={k:v.clone()for k,v in m.state_dict().items()}
        with t.inference_mode():
            source=r.forward(r.source,[1,2,3,4,5,6,7]).past_key_values
            cpu=cpu_rows(source)
            default=legacy(m.apply(source,r.source,r.target))
            on=map_cpu_rows(m,cpu,r.source,r.target,normalize_keys=True,block_size=3)
            off=map_cpu_rows(m,cpu,r.source,r.target,normalize_keys=False,block_size=3)
            directoff=legacy(m.apply(source,r.source,r.target,normalize_keys=False))
            for a,b,c,d in zip(default,on,off,directoff):
                for x,y in zip(a,b):t.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
                for x,y in zip(c,d):t.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
                t.testing.assert_close(b[1],c[1],rtol=0,atol=0)
                self.assertFalse(t.allclose(b[0],c[0]))
        for k,v in m.state_dict().items():t.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_attention_readout_matches_causal_sdpa_with_gqa_and_explicit_mask(self):
        from kvverify.state_diagnostic import attention_readout,compare_readouts
        t=self.torch;prefix=5;indices=[0,2,3];total=9
        q=t.randn(1,4,3,8);k=t.randn(1,2,total,8);v=t.randn_like(k)
        mask=t.zeros(1,1,3,total);mask[...,0]=float('-inf')
        out=attention_readout(q,k,v,prefix_tokens=prefix,query_indices=indices,scale=8**-.5,mask=mask)
        allowed=t.arange(total)[None,:]<=prefix+t.tensor(indices)[:,None]
        fullmask=mask.masked_fill(~allowed[None,None],float('-inf'))
        expected=t.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=fullmask)
        t.testing.assert_close(out[0],expected,rtol=2e-5,atol=2e-6)
        self.assertEqual(float(out[1][...,0].sum()),0.)
        self.assertEqual(float(out[1][...,0,6:].sum()),0.)
        t.testing.assert_close(t.tensor(compare_readouts(out,out)['relative_output_error']),t.zeros(1,4,3))

    def test_attention_probabilities_match_real_math_operator_operation_order(self):
        from kvverify.state_diagnostic import attention_readout
        t=self.torch;t.manual_seed(219)
        q=t.randn(1,4,11,128)*4;k=t.randn(1,2,37,128)*4;v=t.randn_like(k)*8
        prefix=26;indices=list(range(11));mask=t.zeros(1,1,11,37);mask[:,:,0,:]=float('-inf')
        allowed=t.arange(37)[None,:]<=prefix+t.arange(11)[:,None]
        fullmask=mask.masked_fill(~allowed[None,None],float('-inf'))
        for scale in(128**-.5,-.3):
            for block in(None,3):
                actual=attention_readout(q,k,v,prefix_tokens=prefix,query_indices=indices,scale=scale,mask=mask,query_block_size=block)
                native=[];prob=[]
                for start in range(0,11,block or 11):
                    end=start+(block or 11)
                    out,p=t.ops.aten._scaled_dot_product_attention_math(q[...,start:end,:],k.repeat_interleave(2,1),v.repeat_interleave(2,1),fullmask[...,start:end,:],0.,False,scale=scale)
                    native.append(out);prob.append(p)
                t.testing.assert_close(actual[1],t.cat(prob,-2),atol=0.,rtol=0.)
                t.testing.assert_close(actual[0],t.cat(native,-2),atol=0.,rtol=0.)
                self.assertTrue(t.equal(actual[0][:,:,0],t.zeros_like(actual[0][:,:,0])))

    def test_native_query_probe_reproduces_actual_qwen_readout_and_identity_is_zero(self):
        from kvverify.state_diagnostic import capture_native_queries,cpu_rows,layer_interventions
        from kvverify.blocked_attention import blocked_attention
        t=self.torch;r=self.runtime
        with t.inference_mode(),blocked_attention(3):
            cache=r.forward(r.target,[1,2,3,4,5,6,7]).past_key_values
            with capture_native_queries(r.target)as queries:out=r.forward(r.target,[8,9,10,11,12,13,14,15,16,17],cache)
            native=cpu_rows(out.past_key_values)
            mapped=tuple((k[:,:,:7].clone(),v[:,:,:7].clone())for k,v in native)
            rows=layer_interventions(r.target,native,mapped,queries,7)
        self.assertEqual(len(rows),2)
        for rec in queries.values():
            self.assertEqual(rec['full_q'].shape[-2],10)
            self.assertEqual(rec['sdpa_shapes'][0][-2],10)
            self.assertEqual(rec['actual_readout_full'].shape[-2],10)
        for row in rows:
            for arm in row['arms'].values():
                self.assertLess(float(t.tensor(arm['relative_output_error']).abs().max()),1e-6)
                self.assertLess(float(t.tensor(arm['native_to_candidate_attention_kl']).abs().max()),1e-6)
        self.assertTrue(all(not b.self_attn._forward_pre_hooks and not b.self_attn._forward_hooks for b in r.target.model.layers))

    def test_query_hooks_are_removed_on_failure_and_value_swap_keeps_addressing(self):
        from kvverify.state_diagnostic import capture_native_queries,attention_readout,compare_readouts
        t=self.torch
        original=t.nn.functional.scaled_dot_product_attention
        with self.assertRaisesRegex(RuntimeError,'abort'):
            with capture_native_queries(self.runtime.target):raise RuntimeError('abort')
        self.assertIs(t.nn.functional.scaled_dot_product_attention,original)
        self.assertTrue(all(not b.self_attn._forward_pre_hooks and not b.self_attn._forward_hooks for b in self.runtime.target.model.layers))
        q=t.randn(1,4,1,8);k=t.randn(1,2,5,8);v=t.randn_like(k)
        kw={'prefix_tokens':4,'query_indices':[0],'scale':8**-.5}
        native=attention_readout(q,k,v,**kw);changed=attention_readout(q,k,v+2,**kw)
        result=compare_readouts(native,changed)
        self.assertEqual(float(t.tensor(result['native_to_candidate_attention_kl']).abs().max()),0.)
        self.assertGreater(float(t.tensor(result['relative_output_error']).min()),0.)

    def test_readout_guard_exports_discrepancy_without_relaxing_tolerance(self):
        from kvverify.state_diagnostic import capture_native_queries,cpu_rows,layer_interventions,ReadoutMismatch
        t=self.torch;r=self.runtime
        with t.inference_mode():
            cache=r.forward(r.target,[1,2,3,4]).past_key_values
            with capture_native_queries(r.target)as queries:out=r.forward(r.target,[5,6,7,8,9],cache)
            native=cpu_rows(out.past_key_values)
            mapped=tuple((k[:,:,:4].clone(),v[:,:,:4].clone())for k,v in native)
            queries[0]['projected_output_full']=queries[0]['projected_output_full']+.1
            with self.assertRaises(ReadoutMismatch)as caught:layer_interventions(r.target,native,mapped,queries,4)
        self.assertEqual(caught.exception.diagnostics['layer'],0)
        self.assertEqual(caught.exception.diagnostics['atol'],2e-5)
        self.assertGreater(caught.exception.diagnostics['max_abs_error'],.09)


if __name__=='__main__':unittest.main()
