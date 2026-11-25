"""Weighted radius, absolute positions and immutable-value contracts."""
import os
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny offline Torch checks')
class KeyGeometryTests(unittest.TestCase):
    def setUp(self):
        from test_continuation_training import ContinuationLossTests
        ContinuationLossTests.setUp(self)

    def test_projection_restores_per_token_weighted_radius_and_absolute_rope(self):
        from kvverify.key_geometry import project_target_key_radius
        from kvverify.train_mapper import NeuralMapper
        t=self.t;model=self.r.target
        with t.no_grad():
            gamma=model.model.layers[0].self_attn.k_norm.weight
            gamma.copy_(t.linspace(.5,2.,8))
            raw=t.randn(1,2,7,8);raw=raw/raw.square().mean(-1,keepdim=True).sqrt()
            native=NeuralMapper._rotate_token_block(raw*gamma,model,17)
            wrong=native*t.linspace(.2,2.,7)[None,None,:,None];snapshot=wrong.clone()
            result=project_target_key_radius(wrong,model,0,position_offset=17)
            t.testing.assert_close(result,native,atol=2e-6,rtol=2e-6)
            t.testing.assert_close(project_target_key_radius(result,model,0,position_offset=17),result,atol=2e-6,rtol=2e-6)
            self.assertTrue(t.equal(wrong,snapshot))

    def test_cpu_cache_staging_preserves_values_and_refuses_undefined_direction(self):
        from kvverify.key_geometry import project_cache_cpu_rows,project_target_key_radius
        from kvverify.hf_backend import legacy
        t=self.t
        with t.no_grad():
            rows=legacy(self.r.forward(self.r.target,[1,2,3,4,5,6]).past_key_values)
            snapshots=[tuple(x.clone()for x in pair)for pair in rows]
            result=project_cache_cpu_rows(rows,self.r.target,block_size=2)
            for before,after,source in zip(snapshots,result,rows):
                self.assertTrue(t.equal(before[1],after[1]))
                self.assertTrue(t.equal(before[0],source[0]))
            with self.assertRaisesRegex(ValueError,'direction'):
                project_target_key_radius(t.zeros_like(rows[0][0]),self.r.target,0)

    def test_resident_and_staged_key_projection_are_equivalent(self):
        from kvverify.key_geometry import project_cache_cpu_rows,project_target_cache
        from kvverify.hf_backend import legacy
        t=self.t
        with t.no_grad():
            cache=self.r.forward(self.r.target,list(range(1,20))).past_key_values
            staged=project_cache_cpu_rows(legacy(cache),self.r.target,block_size=3)
            resident=legacy(project_target_cache(cache,self.r.target))
            for left,right in zip(staged,resident):
                for a,b in zip(left,right):t.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
