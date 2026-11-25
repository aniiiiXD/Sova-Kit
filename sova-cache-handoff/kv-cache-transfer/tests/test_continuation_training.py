"""Real causal-alignment and frozen-base gradient contracts on tiny Qwen."""
import os
import unittest
from kvverify.continuation_training import ContinuationSampler


class ContinuationSamplerTests(unittest.TestCase):
    def test_document_uniform_seeded_windows_and_exact_continuation(self):
        docs={str(i):list(range(100))for i in range(4)}
        a=ContinuationSampler(docs,tail_tokens=4);b=ContinuationSampler(dict(reversed(list(docs.items()))),tail_tokens=4)
        seen=[]
        for _ in range(8):
            x=a.next();self.assertEqual(x,b.next());doc,prefix,tail,start=x
            self.assertEqual(prefix+tail,docs[doc][start:start+len(prefix)+4]);seen.append(doc)
        self.assertEqual(len(set(seen[:4])),4);self.assertEqual(len(set(seen[4:])),4)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Tiny explicit Torch tests')
class ContinuationLossTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        from kvverify.affine_cache import AffineCacheMapper
        torch.set_num_threads(1);torch.manual_seed(8);self.t=torch
        c=Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.,use_sliding_window=False,eos_token_id=47)
        c._attn_implementation='sdpa';model=Qwen3ForCausalLM(c).float().eval().requires_grad_(False)
        self.r=DenseRuntime(model,model)
        self.m=AffineCacheMapper({'source_geometry':[2,8,2],'target_geometry':[2,8,2],'source_layers':[0,1],
            'source_layer_groups':[[0],[1]],'head_mixing':False})
        with torch.no_grad():self.m.weight.copy_(torch.eye(8).repeat(2,2,2,1,1));self.m.bias.zero_()

    def test_identity_cache_matches_every_native_continuation_position(self):
        from kvverify.continuation_training import continuation_loss
        loss,metrics=continuation_loss(self.r,self.m,[1,2,3,4,5,6],[7,8,9,10])
        self.assertLess(abs(float(loss)),1e-6);self.assertEqual(metrics['mapped_prefix_tokens'],5)
        self.assertEqual(metrics['supervised_tokens'],4);self.assertFalse(metrics['gold_qa_used'])

    def test_perturbed_mapper_receives_finite_gradients_without_base_grads(self):
        from kvverify.continuation_training import continuation_loss
        with self.t.no_grad():self.m.bias.add_(.3)
        loss,_=continuation_loss(self.r,self.m,[1,2,3,4,5,6],[7,8,9,10]);loss.backward()
        self.assertGreater(float(loss),0.)
        self.assertTrue(all(p.grad is not None and self.t.isfinite(p.grad).all()for p in self.m.parameters()))
        self.assertTrue(all(p.grad is None for p in self.r.target.parameters()))


if __name__=='__main__':unittest.main()
