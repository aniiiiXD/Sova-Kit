"""Reference scorer contracts; optional tiny random CPU Qwen cache parity."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

from kvverify.data import Example
from kvverify.reference_eval import (HARNESS_REVISION, HELLASWAG_FILE, QUALITY_FILE, TRAIN_FILE,
    _ending_ll, _prepare, _summary, encode_pair, hellaswag_preprocess, load_reference_data,
    run_hellaswag_reference, run_quality_reference, score_endings, validate_hellaswag_record)


class CharTokenizer:
    is_fast = True
    special_tokens_map = {'eos_token': '\x7f'}
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        value = {'input_ids': [ord(c)%127 for c in text]}
        if return_offsets_mapping:
            value['offset_mapping'] = [(i, i+1) for i in range(len(text))]
        return value
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids if not(skip_special_tokens and i == 127))
    def get_vocab(self):
        return {chr(i): i for i in range(128)}
    def apply_chat_template(self, messages, **kwargs):
        return 'user:\n'+messages[0]['content']+'\nassistant:\n'


def hellaswag_record(identifier='1'):
    row = {'benchmark_id': 'hella:'+identifier, 'source_id': 'source:'+identifier,
        'official_split': 'validation', 'activity_label': 'Test', 'ctx_a': 'A thing', 'ctx_b': 'happens',
        'endings': [' next.', ' instead.', ' again.', ' never.'], 'gold_index': 0, 'label': '0'}
    row['query'] = 'Test: A thing Happens'
    row['choices'] = [hellaswag_preprocess(x) for x in row['endings']]
    row['completion_char_lengths'] = [len(x) for x in row['choices']]
    return row


def fake_runtime():
    config = types.SimpleNamespace(max_position_embeddings=4096)
    runtime = types.SimpleNamespace(source=types.SimpleNamespace(config=config),
        target=types.SimpleNamespace(config=config), tokenizer=CharTokenizer(),
        identities={'source': {'sha256': 's'}, 'target': {'sha256': 't'}},
        tokenizer_sha256='tok', prompt_token_limit=1024)
    mapper = types.SimpleNamespace(manifest={'source_sha256': 's', 'target_sha256': 't',
        'tokenizer_sha256': 'tok', 'calibration_ids': [], 'calibration_documents': [],
        'calibration_context_hashes': []})
    return runtime, mapper


class ReferenceProtocolTests(unittest.TestCase):
    def test_quality_drops_cache_and_decode_output_before_next_variant(self):
        class Cache:
            pass
        class Logits:
            def __getitem__(self, key): return self
            def argmax(self): return types.SimpleNamespace(item=lambda: 65)
        runtime, mapper = fake_runtime()
        runtime.target.config.eos_token_id = 127
        runtime.sync = lambda: None
        runtime.forward = lambda model, ids, cache: types.SimpleNamespace(past_key_values=cache, logits=Logits())
        references = []
        def prepare(*args):
            self.assertTrue(all(ref() is None for ref in references),
                'Previous cache/decode output survived into the next cold attempt')
            cache = Cache()
            references.append(weakref.ref(cache))
            return cache, Logits(), 65, {'cold_first_token_ms': 1.}
        helper = types.SimpleNamespace(split_document_prompt=lambda runtime, example: ([1,2,3],[1],[2,3]))
        example = Example('q1','d1','doc','question',dict(zip('ABCD',['1','2','3','4'])),'A','eval')
        with (tempfile.TemporaryDirectory() as temporary,
              patch.dict('sys.modules', {'kvverify.qa_training': helper}),
              patch('kvverify.reference_eval._prepare', side_effect=prepare),
              patch('kvverify.reference_eval.score_option_logits', return_value={'decision_correct': True})):
            result = run_quality_reference(runtime, [example], mapper, Path(temporary)/'result',
                freegen_max_new_tokens=2, max_seconds=30)
        self.assertTrue(result['fully_completed'])
        self.assertEqual(len(references), 3)
        self.assertTrue(all(ref() is None for ref in references))

    def test_hellaswag_drops_prefix_cache_before_next_variant(self):
        class Cache:
            pass
        runtime, mapper = fake_runtime()
        runtime.sync = lambda: None
        references = []
        def prepare(*args):
            self.assertTrue(all(ref() is None for ref in references),
                'Previous likelihood prefix cache survived into the next attempt')
            cache = Cache()
            references.append(weakref.ref(cache))
            return cache, None, 0, {'cold_first_token_ms': 1.}
        with (tempfile.TemporaryDirectory() as temporary,
              patch('kvverify.reference_eval._prepare', side_effect=prepare),
              patch('kvverify.reference_eval._ending_ll', new=lambda *args: -1.)):
            result = run_hellaswag_reference(runtime, [hellaswag_record()], mapper, Path(temporary)/'result', max_seconds=30)
        self.assertTrue(result['fully_completed'])
        self.assertEqual(len(references), 3)
        self.assertTrue(all(ref() is None for ref in references))

    def test_character_normalization_excludes_delimiter_and_utf8_bytes(self):
        value = score_endings([-1.2, -2., -100., -100.], ['é', 'aa', 'bb', 'cc'], 1)
        self.assertFalse(value['acc'])
        self.assertTrue(value['acc_norm'])
        self.assertEqual(value['choice_character_lengths'], [1, 2, 2, 2])
        self.assertEqual(value['prediction'], 0)
        self.assertEqual(value['prediction_norm'], 1)

    def test_pair_encoding_moves_context_whitespace_into_ending(self):
        tokenizer = CharTokenizer()
        prefix, ending = encode_pair(tokenizer, 'abc  ', ' next')
        self.assertEqual(prefix, tokenizer('abc')['input_ids'])
        self.assertEqual(ending, tokenizer('   next')['input_ids'])
        self.assertEqual(prefix+ending, tokenizer('abc   next')['input_ids'])

    def test_contextual_pair_encoding_not_independent_ending_tokenization(self):
        class BoundaryTokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': {'ctx':[1,2], 'ctx ending':[1,2,9], ' ending':[8]}[text]}
        self.assertEqual(encode_pair(BoundaryTokenizer(), 'ctx', ' ending'), ([1,2],[9]))

    def test_unstable_context_prefix_rejected(self):
        class Unstable:
            def __call__(self, text, **kwargs):
                return {'input_ids': [1,2] if text == 'ctx' else [1,9,3]}
        with self.assertRaisesRegex(ValueError, 'tokenization changes'):
            encode_pair(Unstable(), 'ctx', ' ending')

    def test_pinned_preprocessing_and_character_manifest_validated(self):
        row = hellaswag_record()
        validate_hellaswag_record(row)
        self.assertEqual(hellaswag_preprocess('  A [title] X [step] y  '), 'A. X y')
        row['completion_char_lengths'][0] += 1
        with self.assertRaisesRegex(ValueError, 'character lengths'):
            validate_hellaswag_record(row)

    def test_budget_rows_preserve_hellaswag_denominators(self):
        runtime, mapper = fake_runtime()
        with tempfile.TemporaryDirectory() as temporary, patch('kvverify.reference_eval.time.monotonic', side_effect=[0.]+[2.]*50):
            result = run_hellaswag_reference(runtime, [hellaswag_record('1'),hellaswag_record('2')], mapper,
                Path(temporary)/'result', max_seconds=1)
        for variant in ('source','native','mapper'):
            self.assertEqual(result['variants'][variant]['planned_items'], 2)
            self.assertEqual(result['variants'][variant]['not_run_time_budget'], 2)
            self.assertEqual(result['variants'][variant]['acc_norm'], 0.)
        self.assertFalse(result['fully_completed'])

    def test_budget_rows_preserve_quality_denominators(self):
        runtime, mapper = fake_runtime()
        examples = [Example('q1','d1','doc','question',dict(zip('ABCD',['1','2','3','4'])),'A','eval')]
        helper = types.SimpleNamespace(split_document_prompt=lambda runtime, example: ([1,2,3],[1],[2,3]))
        with (tempfile.TemporaryDirectory() as temporary,
              patch.dict('sys.modules', {'kvverify.qa_training': helper}),
              patch('kvverify.reference_eval.time.monotonic', side_effect=[0.]+[2.]*50)):
            result = run_quality_reference(runtime, examples, mapper, Path(temporary)/'result', max_seconds=1)
        self.assertEqual(result['variants']['mapper']['not_run_time_budget'], 1)
        self.assertEqual(result['paired']['decision_correct']['documents'], 1)

    def test_error_rows_cannot_turn_stale_success_into_accuracy(self):
        rows = []
        for variant in ('source','native','mapper'):
            for i,status in enumerate(('ok','error','over_limit','not_run_time_budget')):
                rows.append({'id':str(i),'document_id':'doc-'+str(i),'context_hash':'ctx',
                    'prompt_hash':'p','run_config_hash':'c','repeat':0,'variant':variant,
                    'status':status,'acc':True,'acc_norm':True})
        result = _summary(rows, ('acc','acc_norm'))
        self.assertEqual(result['variants']['mapper']['acc'], .25)
        self.assertEqual(result['paired']['acc']['examples'], 4)

    def test_native_agreement_retains_failed_rows_in_denominator_and_labels_fallbacks(self):
        rows=[{'variant':v,'id':str(i),'document_id':str(i),'repeat':0,'status':'ok'if i==0 else'error',
               'context_hash':'ctx','prompt_hash':'prompt','run_config_hash':'config',
               'decision_correct':True,'decision_answer':'A','fallback':v=='mapper'and i==0}
              for v in('source','native','mapper')for i in range(2)]
        result=_summary(rows,('decision_correct',))['variants']['mapper']
        self.assertEqual(result['native_agreement_decision_answer'],.5)
        self.assertEqual(result['completed_native_fallback_items'],1)
        self.assertEqual(result['completed_transfer_items'],0)

    def test_duplicate_reference_rows_rejected(self):
        row = {'variant':'native','id':'1'}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _summary([row,row], ('acc',))

    def test_frozen_loader_verifies_selected_files_but_not_unbundled_raws(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = {'id':'q1','document_id':'quality:1','context':'doc','question':'q',
                'choices':dict(zip('ABCD',['1','2','3','4'])),'answer':'A','split':'eval','official_split':'dev'}
            payloads = {QUALITY_FILE:[q], TRAIN_FILE:[dict(q, id='train',split='calibration',official_split='train')],
                        HELLASWAG_FILE:[hellaswag_record()]}
            manifest = {'schema_version':1,'official_test_accessed':False,
                'upstream_revisions':{'lm_evaluation_harness_github':HARNESS_REVISION},'processed':{}}
            for relative,rows in payloads.items():
                path=root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                manifest['processed'][relative]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bytes':path.stat().st_size,'examples':len(rows)}
            manifest['sources']={'raw/absent':'not bundled'}
            path=root/'manifest.json'
            path.write_text(json.dumps(manifest))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            quality,hella,_=load_reference_data(root,digest,expected_quality=1,expected_hellaswag=1)
            self.assertEqual(len(quality),1)
            self.assertEqual(len(hella),1)
            with self.assertRaisesRegex(ValueError,'manifest checksum'):
                load_reference_data(root,'bad')
            (root/QUALITY_FILE).write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'file mismatch'):
                load_reference_data(root,digest,expected_quality=1,expected_hellaswag=1)


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1', 'Explicit tiny CPU Torch tests required')
class TinyLikelihoodTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from kvverify.hf_backend import DenseRuntime
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=24,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
            max_position_embeddings=2048,use_sliding_window=False,eos_token_id=127)).eval().float()
        self.runtime=DenseRuntime(model,model,CharTokenizer())
        self.runtime.prompt_token_limit=1024
        self.mapper=self.runtime.identity_mapper()
        self.mapper.manifest.update(source_sha256=self.runtime.identities['source']['sha256'],
            target_sha256=self.runtime.identities['target']['sha256'],tokenizer_sha256=self.runtime.tokenizer_sha256,
            calibration_ids=[],calibration_documents=[],calibration_context_hashes=[])

    def test_partial_native_prefix_preserves_exact_identity_and_counts_cost(self):
        self.mapper.manifest['native_prefix_policy']='256'
        ids=[5,6,7,8]*80
        with self.runtime.torch.inference_mode():
            _,native,_,_=_prepare(self.runtime,self.mapper,'native',ids,300)
            _,candidate,_,timing=_prepare(self.runtime,self.mapper,'mapper',ids,300)
        self.runtime.torch.testing.assert_close(native,candidate,atol=1e-6,rtol=1e-5)
        self.assertEqual(timing['native_prefix_tokens'],256)
        self.assertEqual(timing['imported_tokens'],44)
        self.assertEqual(timing['target_prefill_tokens'],276)
        self.assertTrue(timing['transfer_used']);self.assertFalse(timing['fallback'])

    def test_short_prefix_native_fallback_never_calls_mapper(self):
        self.mapper.manifest['native_prefix_policy']='256'
        with self.runtime.torch.inference_mode(),patch.object(self.mapper,'apply',side_effect=AssertionError('Must not map')):
            _,native,_,_=_prepare(self.runtime,self.mapper,'native',[5,6,7,8],3)
            _,candidate,_,timing=_prepare(self.runtime,self.mapper,'mapper',[5,6,7,8],3)
        self.runtime.torch.testing.assert_close(native,candidate,atol=0.,rtol=0.)
        self.assertEqual(timing['imported_tokens'],0)
        self.assertEqual(timing['target_prefill_tokens'],4)
        self.assertFalse(timing['transfer_used']);self.assertTrue(timing['fallback'])

    def test_cached_likelihood_matches_uncached_teacher_forcing_every_token(self):
        torch=self.runtime.torch
        context=[5,6,7,8]
        endings=[[9],[10,11,12],[13,14]]
        prepared={}
        for variant in ('source','native','mapper'):
            cache,logits,_,_= _prepare(self.runtime,self.mapper,variant,context,len(context)-1)
            before=cache.get_seq_length()
            scores=[]
            for ending in endings:
                cached=_ending_ll(self.runtime,self.runtime.target,cache,logits,ending)
                with torch.inference_mode():
                    ids=torch.tensor([context+ending[:-1]])
                    out=self.runtime.target(input_ids=ids,use_cache=True,logits_to_keep=0)
                    relevant=out.logits[0,len(context)-1:].float()
                    expected=torch.log_softmax(relevant,dim=-1).gather(-1,torch.tensor(ending)[:,None]).sum()
                self.assertAlmostEqual(cached,float(expected),places=5)
                self.assertEqual(cache.get_seq_length(),before,'Each ending must clone shared prefix cache')
                scores.append(cached)
            prepared[variant]=scores
        self.assertEqual(prepared['source'],prepared['native'])
        for actual,expected in zip(prepared['mapper'],prepared['native']):
            self.assertAlmostEqual(actual,expected,places=5)

    def test_real_hellaswag_loop_emits_three_arms_and_four_full_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=run_hellaswag_reference(self.runtime,[hellaswag_record()],self.mapper,
                Path(temporary)/'result',max_seconds=30)
            rows=[json.loads(x) for x in (Path(temporary)/'result/rows.jsonl').read_text().splitlines()]
        self.assertTrue(result['fully_completed'])
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertEqual(row['status'],'ok')
            self.assertEqual(len(row['choice_loglikelihoods']),4)
            self.assertEqual(row['choice_character_lengths'],[len(x) for x in hellaswag_record()['choices']])

    def test_real_quality_loop_keeps_question_native_and_scores_generation_separately(self):
        example=Example('quality:dev:tiny','quality:tiny','A complete document.',
            'Which choice?',dict(zip('ABCD',['one','two','three','four'])),'A','eval')
        with tempfile.TemporaryDirectory() as temporary:
            result=run_quality_reference(self.runtime,[example],self.mapper,Path(temporary)/'result',
                max_seconds=30,freegen_max_new_tokens=4)
            rows=[json.loads(x) for x in (Path(temporary)/'result/rows.jsonl').read_text().splitlines()]
        self.assertTrue(result['fully_completed'])
        self.assertEqual(len(rows),3)
        native=next(r for r in rows if r['variant']=='native')
        mapped=next(r for r in rows if r['variant']=='mapper')
        self.assertEqual(mapped['status'],'ok')
        self.assertEqual(mapped['decision_answer'],native['decision_answer'])
        self.assertGreater(mapped['target_prefill_tokens'],1)
        self.assertEqual(mapped['imported_tokens']+mapped['target_prefill_tokens'],mapped['prompt_tokens'])
        self.assertIn('free_generation_strict_correct',mapped)
        self.assertIn('option_probability_mass',mapped)


if __name__=='__main__':
    unittest.main()
