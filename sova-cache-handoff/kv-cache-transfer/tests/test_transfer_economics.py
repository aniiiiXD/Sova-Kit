"""Real tiny-model cold/warm parity and honest partial-prefill accounting."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny CPU Torch checks')
class TransferEconomicsTests(unittest.TestCase):
    def setUp(self):
        from test_reference_eval import TinyLikelihoodTests
        TinyLikelihoodTests.setUp(self)
        self.mapper.manifest['native_prefix_policy']='256'

    def test_all_four_arms_match_identity_but_report_distinct_cache_work(self):
        from kvverify.transfer_economics import measure_arm,ARMS
        full=[5,6,7,8]*80;prefix=full[:300];suffix=full[300:]
        with self.runtime.torch.inference_mode():
            results={arm:measure_arm(self.runtime,self.mapper,full,prefix,suffix,arm)for arm in ARMS}
        for row,vector in results.values():
            self.runtime.torch.testing.assert_close(vector,results['native_cold'][1],atol=1e-6,rtol=1e-5)
            self.assertGreater(row['request_ms'],0.)
        cold=results['transfer_cold'][0];warm=results['transfer_source_cache_hit'][0]
        self.assertEqual(cold['source_prefill_tokens'],300)
        self.assertEqual(warm['source_prefill_tokens'],0)
        self.assertEqual(warm['target_prefill_tokens'],276)
        self.assertEqual(warm['imported_tokens'],44)
        self.assertGreater(warm['resident_prefix_cache_bytes'],0)
        self.assertEqual(warm['cache_residency'],'same_cpu')
        self.assertFalse(warm['transport_measured'])
        self.assertEqual(results['native_target_cache_hit'][0]['target_prefill_tokens'],20)

    def test_complete_small_protocol_preserves_denominators_and_logs_warmup(self):
        from kvverify.data import Example
        from kvverify.transfer_economics import run_economics
        example=Example('q1','d1','long text '*40,'Which?',dict(zip('ABCD',['a','b','c','d'])),'A','eval')
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary)/'run'
            result=run_economics(self.runtime,self.mapper,[example],root,example_count=1,repeats=2)
            self.assertEqual(result['timed_requests'],8)
            self.assertTrue(all(v['count']==2 for v in result['request_ms'].values()))
            self.assertEqual(len((root/'warmup.jsonl').read_text().splitlines()),4)

    def test_cold_and_warm_apply_the_identical_nontrivial_key_policy(self):
        from kvverify.transfer_economics import measure_arm
        self.mapper.manifest['key_projection']='target_rms_ellipsoid'
        for layer in self.runtime.target.model.layers:layer.self_attn.k_norm.variance_epsilon=.1
        full=[5,6,7,8]*80
        with self.runtime.torch.inference_mode():
            cold,x=measure_arm(self.runtime,self.mapper,full,full[:300],full[300:],'transfer_cold')
            warm,y=measure_arm(self.runtime,self.mapper,full,full[:300],full[300:],'transfer_source_cache_hit')
        self.runtime.torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
        self.assertEqual(cold['source_prefill_tokens'],300)
        self.assertEqual(warm['source_prefill_tokens'],0)


if __name__=='__main__':unittest.main()
