"""Cross-head identifiability, held-out reconstruction and frozen-LM gradients."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny model tests')
class AffineCacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        torch.manual_seed(2026);torch.set_num_threads(1);self.t=torch
        def model():
            c=Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=1024,
                attention_dropout=0.,use_sliding_window=False,eos_token_id=47)
            c._attn_implementation='sdpa'
            return Qwen3ForCausalLM(c).float().eval().requires_grad_(False)
        self.r=DenseRuntime(model(),model())

    def synthetic(self,count):
        from transformers import DynamicCache
        from kvverify.hf_backend import unrotate
        t=self.t;source=[];target=[]
        for layer in range(2):
            pair=[];expected=[]
            for side in range(2):
                x=t.randn(1,2,count,8)
                y=x.flip(1)*1.3+.2
                pair.append(unrotate(x,self.r.source)if side==0 else x)
                expected.append(unrotate(y,self.r.target)if side==0 else y)
            source.append(tuple(pair));target.append(tuple(expected))
        return DynamicCache.from_legacy_cache(tuple(source)),DynamicCache.from_legacy_cache(tuple(target))

    def fit(self,mix=True):
        from kvverify.affine_cache import CrossHeadMoments
        with self.t.no_grad():
            source,target=self.synthetic(128);moments=CrossHeadMoments((2,8,2),[0,1],'cpu')
            moments.update(source,target,self.r.source,self.r.target)
            m=moments.fit(1e-7,mix,{'source_sha256':'random-smoke-source','target_sha256':'random-smoke-target',
                'tokenizer_sha256':self.r.tokenizer_sha256,'calibration_ids':['train'],
                'calibration_documents':['train'],'calibration_context_hashes':['train']})
        return m

    def test_all_head_recovers_permuted_heads_on_unseen_tokens(self):
        from kvverify.hf_backend import legacy
        all_head=self.fit(True);same_head=self.fit(False)
        with self.t.no_grad():
            source,target=self.synthetic(37)
            actual=legacy(all_head.apply(source,self.r.source,self.r.target))
            weak=legacy(same_head.apply(source,self.r.source,self.r.target));expected=legacy(target)
        good=sum(float((a-b).square().mean())for p,q in zip(actual,expected)for a,b in zip(p,q))
        bad=sum(float((a-b).square().mean())for p,q in zip(weak,expected)for a,b in zip(p,q))
        self.assertLess(good,1e-9);self.assertGreater(bad,.5)

    def test_sparse_full_history_positions_are_not_renumbered_before_derotation(self):
        from kvverify.affine_cache import CrossHeadMoments
        from kvverify.hf_backend import legacy
        with self.t.no_grad():
            source,target=self.synthetic(400);moments=CrossHeadMoments((2,8,2),[0,1],'cpu')
            indices=list(range(270,398,2));moments.update(source,target,self.r.source,self.r.target,token_indices=indices)
            self.assertEqual(moments.n,len(indices))
            m=moments.fit(1e-7,True,{});fresh,expected=self.synthetic(37)
            actual=legacy(m.apply(fresh,self.r.source,self.r.target));expected=legacy(expected)
            error=sum(float((x-y).square().mean())for a,b in zip(actual,expected)for x,y in zip(a,b))
            self.assertLess(error,1e-9)
            with self.assertRaisesRegex(ValueError,'repeated'):
                moments.update(source,target,self.r.source,self.r.target,token_indices=[3,3])

    def test_variance_relative_ridge_is_invariant_to_source_feature_units(self):
        from transformers import DynamicCache
        from kvverify.affine_cache import CrossHeadMoments
        from kvverify.hf_backend import legacy

        source, target = self.synthetic(128)
        resized = DynamicCache.from_legacy_cache(tuple((key * .125, value * .125) for key, value in legacy(source)))
        expected = CrossHeadMoments((2, 8, 2), [0, 1], "cpu")
        shifted = CrossHeadMoments((2, 8, 2), [0, 1], "cpu")
        with self.t.inference_mode():
            expected.update(source, target, self.r.source, self.r.target)
            shifted.update(resized, target, self.r.source, self.r.target)
            native_units = expected.fit(.01, True, {}, relative_regularization=True)
            changed_units = shifted.fit(.01, True, {}, relative_regularization=True)
        self.assertFalse(native_units.weight.is_inference())
        self.t.testing.assert_close(native_units.weight * 8, changed_units.weight, rtol=2e-5, atol=2e-6)
        self.t.testing.assert_close(native_units.bias, changed_units.bias, rtol=2e-5, atol=2e-6)

    def test_depth_mismatched_fit_exports_source_geometry(self):
        from transformers import DynamicCache
        from cloud.harvey_pilot_worker import aligned_layers
        from kvverify.affine_cache import CrossHeadMoments
        from kvverify.hf_backend import legacy

        layers = aligned_layers(2, 3)
        source, target = self.synthetic(80)
        target_rows = legacy(target)
        target = DynamicCache.from_legacy_cache(tuple(target_rows[layer] for layer in layers))
        moments = CrossHeadMoments((2, 8, 3), layers, 'cpu', source_geometry=(2, 8, 2))
        with self.t.no_grad():
            moments.update(source, target, self.r.source, self.r.target)
            mapper = moments.fit(1e-7, True, {})
            actual = legacy(mapper.apply(source, self.r.source, self.r.target))
        self.assertEqual(mapper.manifest['source_geometry'], [2, 8, 2])
        self.assertEqual(mapper.manifest['target_geometry'], [2, 8, 3])
        self.assertEqual(len(actual), 3)
        for predicted, expected in zip(actual, legacy(target)):
            for left, right in zip(predicted, expected):
                self.t.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)

    def test_normalized_ridge_units_and_saved_identity(self):
        from kvverify.affine_cache import AffineCacheMapper
        m=self.fit();self.assertAlmostEqual(m.manifest['raw_sum_lambda_equivalent'],128e-7)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'mapper';m.save(path)
            loaded=AffineCacheMapper.load(path,'random-smoke-source','random-smoke-target',self.r.tokenizer_sha256)
            for a,b in zip(m.parameters(),loaded.parameters()):self.t.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'identity'):AffineCacheMapper.load(path,'bad','random-smoke-target',self.r.tokenizer_sha256)
            with (path/'weights.safetensors').open('ab')as f:f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'checksum'):AffineCacheMapper.load(path,'random-smoke-source','random-smoke-target',self.r.tokenizer_sha256)

    def test_cpu_staging_and_absolute_block_positions_match_direct_mapping(self):
        from kvverify.state_diagnostic import map_cpu_rows,cpu_rows
        from kvverify.hf_backend import legacy
        m=self.fit()
        with self.t.no_grad():
            source,_=self.synthetic(19)
            a=legacy(m.apply_checkpointed(source,self.r.source,self.r.target,token_block_size=5))
            b=map_cpu_rows(m,cpu_rows(source),self.r.source,self.r.target,normalize_keys=False,block_size=3)
        for p,q in zip(a,b):
            for x,y in zip(p,q):self.t.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)

    def test_fast_full_head_mapper_preserves_rotary_positions_across_blocks(self):
        from kvverify.fast_affine import map_affine_fast
        from kvverify.hf_backend import legacy

        mapper = self.fit()
        with self.t.inference_mode():
            source, _ = self.synthetic(57)
            reference = legacy(mapper.apply_checkpointed(source, self.r.source, self.r.target, token_block_size=5))
            for fast_block in (11, 128):
                actual = legacy(map_affine_fast(mapper, source, self.r.source, self.r.target, block_tokens=fast_block))
                for want, got in zip(reference, actual):
                    for left, right in zip(want, got):
                        self.t.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)

    def test_answer_distillation_updates_only_low_rank_factors_and_folds(self):
        from kvverify.answer_mapper import AnswerMapper, answer_loss
        from kvverify.hf_backend import legacy
        from kvverify.train_mapper import model_forward

        model = AnswerMapper(self.fit(), rank=3)
        prefix, suffix = [1, 8, 3, 5], [4, 9]
        options = {"A": 2, "B": 3, "C": 4, "D": 5}
        loss, metrics = answer_loss(self.r, model, prefix+suffix, prefix, suffix, options,
                                    native_prefix_tokens=1)
        loss.backward()
        self.assertTrue(self.t.isfinite(model.up.grad).all() and model.up.grad.abs().sum() > 0)
        self.assertTrue(self.t.isfinite(model.down.grad).all())
        self.assertTrue(all(p.grad is None for m in (self.r.source, self.r.target, model.base)
                            for p in m.parameters()))
        with self.t.no_grad():
            model.up.sub_(.01 * model.up.grad)
            source = model_forward(self.r.source, prefix).past_key_values
            factored = legacy(model.apply_checkpointed(source, self.r.source, self.r.target))
            folded = legacy(model.folded().apply(source, self.r.source, self.r.target))
        self.assertEqual(set(metrics), {"loss", "full_kl", "option_kl", "prefix_tokens"})
        self.assertFalse(self.t.equal(model.folded().weight, model.base.weight))
        for expected, actual in zip(factored, folded):
            for left, right in zip(expected, actual):
                self.t.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    def test_actual_mapped_readout_updates_selected_layer_without_model_gradients(self):
        from kvverify.answer_mapper import AnswerMapper
        from kvverify.readout_mapper import fit_document

        mapper = AnswerMapper(self.fit(), rank=4)
        optimizer = self.t.optim.AdamW((mapper.down, mapper.up), lr=.003, weight_decay=0.)
        initial = mapper.base.weight.detach().clone()
        result = fit_document(self.r, mapper, optimizer, list(range(20)), [21, 20, 5, 2], [1],
                              native_head_tokens=4, query_count=3, student_trajectory=True)
        with self.t.inference_mode():
            folded = mapper.folded()
        self.assertFalse(folded.weight.is_inference())
        self.assertGreater(result["gradient_norm"], 0)
        self.assertEqual(result["query_positions"], [0, 1, 3])
        self.assertTrue(self.t.equal(initial[:, 0], folded.weight[:, 0]))
        self.assertFalse(self.t.equal(initial[:, 1], folded.weight[:, 1]))
        self.assertTrue(all(parameter.grad is None for model in (self.r.source, self.r.target, mapper.base)
                            for parameter in model.parameters()))

    def test_layer_selection_exports_one_paired_source_bank_for_each_target_layer(self):
        from kvverify.layer_alignment import assemble_selected_layers, functional_layer_errors

        reference, alternate = self.fit(), self.fit()
        alternate.manifest["source_layers"] = [1, 0]
        errors = functional_layer_errors(self.r, [reference, alternate], [1, 2, 3, 4] * 70, [5, 6, 7, 8])
        self.assertEqual(errors.shape, (2, 2))
        self.assertTrue(self.t.isfinite(errors).all())
        selected, choices = assemble_selected_layers([reference, alternate], self.t.tensor([[2., 1.], [1., 2.]]), {})
        self.assertEqual(choices, [1, 0])
        self.assertEqual(selected.manifest["source_layers"], [1, 1])
        self.t.testing.assert_close(selected.weight[:, 0], alternate.weight[:, 0])
        self.t.testing.assert_close(selected.weight[:, 1], reference.weight[:, 1])

    def test_sequence_distillation_aligns_spaced_suffix_positions_and_reaches_mapper(self):
        from kvverify.answer_mapper import AnswerMapper, sequence_kl_loss

        mapper = AnswerMapper(self.fit(), rank=4)
        options = {name: i for i, name in enumerate("ABCD", 1)}
        loss, metrics = sequence_kl_loss(self.r, mapper, [1, 2, 3, 4] * 10, list(range(12)), options,
                                         native_prefix_tokens=4, positions=8)
        loss.backward()
        self.assertEqual(metrics["query_positions"], [0, 1, 3, 4, 6, 7, 9, 11])
        self.assertTrue(self.t.isfinite(mapper.up.grad).all() and mapper.up.grad.abs().sum() > 0)
        self.assertTrue(all(parameter.grad is None for model in (self.r.source, self.r.target)
                            for parameter in model.parameters()))

    def test_functional_loss_reaches_mapper_and_leaves_models_frozen(self):
        from test_qa_training import OffsetTokenizer,fixture
        from kvverify.qa_training import split_document_prompt,qa_training_loss
        m=self.fit().requires_grad_(True);self.r.tokenizer=OffsetTokenizer()
        full,prefix,suffix=split_document_prompt(self.r,fixture(context='short evidence'))
        loss,_=qa_training_loss(self.r,m,full,prefix,suffix,answer_tokens=2,ce_weight=0.,kl_weight=1.)
        loss.backward()
        self.assertTrue(all(p.grad is not None and self.t.isfinite(p.grad).all()for p in m.parameters()))
        self.assertTrue(all(p.grad is None for model in(self.r.source,self.r.target)for p in model.parameters()))


if __name__=='__main__':unittest.main()
