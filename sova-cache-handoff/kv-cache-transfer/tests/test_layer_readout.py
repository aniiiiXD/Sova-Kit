"""Exact layer replay, attention-mass intercept and constrained-solve checks."""
import os
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit offline tiny Torch checks')
class LayerReadoutTests(unittest.TestCase):
    def setUp(self):
        from test_continuation_training import ContinuationLossTests
        ContinuationLossTests.setUp(self)

    def test_layer_streaming_matches_real_full_native_handoff(self):
        from kvverify.hf_backend import legacy
        from kvverify.layer_readout import query_layer
        t=self.t;r=self.r;prefix=[1,2,3,4,5,6];suffix=[7,8,9,10]
        with t.inference_mode():
            cache=r.forward(r.target,prefix).past_key_values;rows=legacy(cache)
            snapshots=[tuple(x.clone()for x in pair)for pair in rows]
            native=r.forward(r.target,prefix+suffix).logits[0,-1]
            hidden=r.target.model.embed_tokens(t.tensor([suffix]))
            for layer,pair in enumerate(rows):hidden,_,_=query_layer(r,layer,hidden,pair)
            logits=r.target.lm_head(r.target.model.norm(hidden))[0,-1]
        t.testing.assert_close(native,logits,atol=1e-6,rtol=1e-5)
        for before,after in zip(snapshots,rows):
            for a,b in zip(before,after):self.assertTrue(t.equal(a,b))

    def test_bias_feature_is_translated_attention_mass_at_current_query(self):
        from kvverify.hf_backend import legacy
        from kvverify.layer_readout import query_layer,readout_rows
        t=self.t;r=self.r
        with t.inference_mode():
            native=legacy(r.forward(r.target,[1,2,3,4,5,6]).past_key_values)[0]
            wrong=native[1].clone();wrong[:,:,2:]+=.2
            hidden=r.target.model.embed_tokens(t.tensor([[7,8,9,10]]))
            _,current,record=query_layer(r,0,hidden,(native[0],wrong),capture=True)
            design,residual=readout_rows(record,current,native,native[1],retained_native_tokens=2)
            delta=t.zeros((2,17,8));delta[:,-1]=-.2
            t.testing.assert_close(design@delta,residual,atol=1e-6,rtol=1e-5)
            self.assertTrue(bool((design[:,:,-1]<1.).all()))

    def test_constrained_correction_is_feasible_and_satisfies_multiplier_equation(self):
        from kvverify.layer_readout import bounded_readout_delta
        t=self.t;t.manual_seed(11)
        d=t.randn(2,20,4,dtype=t.float64);g=d.transpose(-1,-2)@d;c=t.randn(2,4,3,dtype=t.float64)*10
        metric=t.diag(t.tensor([.1,1.,3.,8.],dtype=t.float64));energy=t.ones(2,dtype=t.float64)
        delta,report=bounded_readout_delta(g,c,metric,energy)
        for h,row in enumerate(report):
            self.assertGreater(row['mu'],0.)
            self.assertLessEqual(row['energy_after'],.01+1e-10)
            self.assertGreaterEqual(row['predicted_regularized_objective_reduction'],-1e-8)
            lhs=(g[h]+row['ridge']*t.eye(4,dtype=t.float64)+row['mu']*metric)@delta[h]
            t.testing.assert_close(lhs,c[h],atol=1e-8,rtol=1e-8)

    def test_folded_all_head_value_update_matches_predicted_real_attention_change(self):
        from kvverify.affine_cache import AffineCacheMapper
        from kvverify.hf_backend import legacy
        from kvverify.layer_readout import mapped_layer_pair,query_layer,readout_rows
        t=self.t;r=self.r
        m=AffineCacheMapper({'source_geometry':[2,8,2],'target_geometry':[2,8,2],'source_layers':[0,1],
            'source_layer_groups':[[0],[1]],'head_mixing':True}).requires_grad_(False)
        with t.inference_mode():
            m.weight.copy_(t.eye(16).repeat(2,2,1,1));m.bias.zero_()
            source=legacy(r.forward(r.source,[1,2,3,4,5,6]).past_key_values)[0]
            hidden=r.target.model.embed_tokens(t.tensor([[7,8,9,10]]))
            pair=mapped_layer_pair(r,m,0,source,source,native_tokens=2)
            _,current,old=query_layer(r,0,hidden,pair,capture=True)
            design,_=readout_rows(old,current,source,source[1],retained_native_tokens=2)
            delta=t.randn(2,17,8)*.003;keys=m.weight[0].clone()
            m.weight[1,0].add_(delta[:,:-1].permute(1,0,2).reshape(16,16));m.bias[1,0].add_(delta[:,-1].reshape(16))
            updated=mapped_layer_pair(r,m,0,source,source,native_tokens=2)
            _,_,new=query_layer(r,0,hidden,updated,capture=True)
            actual=(new['actual_readout_full']-old['actual_readout_full'])[0].reshape(2,-1,8)
            t.testing.assert_close(actual,design@delta,atol=1e-6,rtol=1e-5)
            self.assertTrue(t.equal(keys,m.weight[0]))

    def test_zero_energy_means_zero_change_and_invalid_metric_is_rejected(self):
        from kvverify.layer_readout import bounded_readout_delta
        t=self.t;g=t.eye(3)[None];c=t.ones(1,3,2)
        delta,_=bounded_readout_delta(g,c,t.eye(3),t.zeros(1))
        self.assertEqual(float(delta.abs().sum()),0.)
        with self.assertRaisesRegex(ValueError,'positive definite'):
            bounded_readout_delta(g,c,-t.eye(3),t.ones(1))


if __name__=='__main__':unittest.main()
