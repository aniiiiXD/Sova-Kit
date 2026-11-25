"""Boundary-straddling headers and immutable positioned-KV replacement."""
import os
import unittest
from kvverify.native_prefix import header_tokens


class HeaderBoundaryTests(unittest.TestCase):
    def test_header_does_not_include_document_characters(self):
        from test_qa_training import BoundaryRuntime,fixture,MergedOffsetTokenizer
        e=fixture(context='actual document')
        self.assertEqual(header_tokens(BoundaryRuntime(),e),len('<u>Document:\n'))
        r=BoundaryRuntime(MergedOffsetTokenizer());count=header_tokens(r,e)
        self.assertEqual(count,len('<u>Document:\n')//2)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Tiny explicit Torch tests')
class NativePrefixTests(unittest.TestCase):
    def test_oracle_substitutes_only_named_component_and_keeps_common_native_prefix(self):
        import torch
        from transformers import DynamicCache
        from kvverify.native_prefix import oracle_components
        from kvverify.hf_backend import legacy
        a=tuple((torch.ones(1,2,9,8),torch.full((1,2,9,8),2.))for _ in range(2))
        b=tuple((torch.full((1,2,9,8),10.),torch.full((1,2,9,8),20.))for _ in range(2))
        for keys,values in((True,False),(False,True),(True,True),(False,False)):
            result=legacy(oracle_components(DynamicCache.from_legacy_cache(a),DynamicCache.from_legacy_cache(b),
                keys=keys,values=values,retained_native_tokens=4))
            for old,new,actual in zip(a,b,result):
                for before,native,after,all_native in zip(old,new,actual,(keys,values)):
                    torch.testing.assert_close(after[:,:,:4],native[:,:,:4],rtol=0,atol=0)
                    torch.testing.assert_close(after[:,:,4:],native[:,:,4:]if all_native else before[:,:,4:],rtol=0,atol=0)
        self.assertTrue(torch.equal(a[0][0],torch.ones_like(a[0][0])))

    def test_positioned_prefix_and_translated_body_preserved_without_mutation(self):
        import torch
        from transformers import DynamicCache
        from kvverify.native_prefix import replace_prefix
        from kvverify.hf_backend import legacy
        a=tuple((torch.full((1,2,9,8),float(i)),torch.full((1,2,9,8),float(i+1)))for i in range(2))
        b=tuple((torch.full((1,2,4,8),float(i+10)),torch.full((1,2,4,8),float(i+11)))for i in range(2))
        ma=DynamicCache.from_legacy_cache(a);nb=DynamicCache.from_legacy_cache(b)
        result=legacy(replace_prefix(ma,nb,4))
        for row,original,native in zip(result,a,b):
            for x,y,z in zip(row,original,native):
                torch.testing.assert_close(x[:,:,:4],z,rtol=0,atol=0)
                torch.testing.assert_close(x[:,:,4:],y[:,:,4:],rtol=0,atol=0)
        torch.testing.assert_close(legacy(ma)[0][0],a[0][0],rtol=0,atol=0)
        zero=legacy(replace_prefix(ma,nb,0))
        torch.testing.assert_close(zero[0][0],a[0][0],rtol=0,atol=0)
        with self.assertRaises(ValueError):replace_prefix(ma,nb,5)
