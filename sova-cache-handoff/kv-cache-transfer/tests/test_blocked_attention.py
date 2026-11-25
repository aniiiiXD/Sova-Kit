import os
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny CPU smoke opt-in')
class BlockedAttentionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;torch.manual_seed(29)
        from torch.nn import functional as F
        self.F=F
        self.q=torch.randn(1,2,19,8,requires_grad=True)
        self.k=torch.randn(1,2,19,8,requires_grad=True)
        self.v=torch.randn(1,2,19,8,requires_grad=True)

    def compare(self,mask=None,causal=False,rematerialize=False):
        from kvverify.blocked_attention import blocked_attention
        expected=self.F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=mask,is_causal=causal)
        grads=self.torch.autograd.grad(expected.square().sum(),(self.q,self.k,self.v))
        with blocked_attention(4,rematerialize=rematerialize):
            actual=self.F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=mask,is_causal=causal)
        self.torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-6)
        actual_grads=self.torch.autograd.grad(actual.square().sum(),(self.q,self.k,self.v))
        for a,b in zip(grads,actual_grads):self.torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)

    def test_implicit_causal_preserves_global_query_position_and_gradients(self):self.compare(causal=True)

    def test_explicit_cached_suffix_causal_mask_preserves_keys_and_gradients(self):
        self.q=self.q[:,:,:7].detach().requires_grad_()
        mask=self.torch.arange(19)[None,:]<=(12+self.torch.arange(7))[:,None]
        self.compare(mask=mask)

    def test_rematerialized_suffix_backward_matches_after_context_exits(self):
        self.q=self.q[:,:,:7].detach().requires_grad_()
        mask=self.torch.arange(19)[None,:]<=(12+self.torch.arange(7))[:,None]
        self.compare(mask=mask,rematerialize=True)

    def test_additive_and_broadcast_masks(self):
        mask=self.torch.zeros(1,1,19,19);mask[:,:,:,-3:]=-float('inf')
        self.compare(mask=mask)
        self.compare(mask=mask[:,:,:1,:])

    def test_context_restores_function_after_error_and_rejects_dropout(self):
        from kvverify.blocked_attention import blocked_attention
        original=self.F.scaled_dot_product_attention
        with self.assertRaises(ValueError):
            with blocked_attention(4):self.F.scaled_dot_product_attention(self.q,self.k,self.v,dropout_p=.1)
        self.assertIs(self.F.scaled_dot_product_attention,original)

    def test_real_tiny_qwen_full_and_native_cache_match_under_blocking(self):
        from test_hf_optional import TinyHFTests
        from test_qa_training import OffsetTokenizer,fixture
        from cloud.precision_worker import check_example
        from kvverify.blocked_attention import blocked_attention
        helper=TinyHFTests();helper.setUp();r=helper.runtime
        r.tokenizer=OffsetTokenizer();r.prompt_token_limit=1024
        e=fixture();base,native,_=check_example(r,e)
        with blocked_attention(11):row,actual,_=check_example(r,e)
        self.assertTrue(row['oracle']['passed'])
        self.torch.testing.assert_close(native,actual,rtol=1e-4,atol=1e-6)


if __name__=='__main__':unittest.main()
