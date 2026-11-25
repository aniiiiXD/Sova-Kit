"""Small pure-Python integrity checks; no model execution or downloads."""
import copy
import unittest

from kvverify.data import json_digest
from kvverify.eval_resume import plan_resume,merge_resume,verify_evaluation_config
from kvverify.reference_eval import QUALITY_PROTOCOL,VARIANTS


def config():
    row={'protocol':QUALITY_PROTOCOL,'mapper':{'weights_sha256':'fixed'},'runtime':{'source':'s','target':'t'},
         'prompt_tokens_limit':9000,'freegen_max_new_tokens':16,'acceptance':'not_established'}
    return {**row,'run_config_hash':json_digest(row)}


def items():
    return [{'id':str(i),'document_id':'doc'+str(i),'context_hash':'ctx'+str(i),'gold_answer':'A'} for i in range(3)]


def rows(cfg,selected=None):
    return [{**item,'variant':variant,'status':'ok','protocol':QUALITY_PROTOCOL,
        'fallback':False,'run_config_hash':cfg['run_config_hash'],'decision_correct':False,
        'repeat':0,'prompt_hash':'prompt'+item['id']}
        for item in items() if selected is None or item['id'] in selected for variant in VARIANTS]


class EvalResumeTests(unittest.TestCase):
    def test_only_status_determines_selection_not_correctness(self):
        cfg=config();parent=rows(cfg)
        parent[4]['status']='not_run_time_budget'
        plan=plan_resume(items(),parent,QUALITY_PROTOCOL,cfg)
        self.assertEqual(plan['pending_ids'],['1'])
        self.assertEqual(plan['complete_parent_ids'],['0','2'])
        changed=copy.deepcopy(parent)
        for row in changed:row['decision_correct']=True
        self.assertEqual(plan_resume(items(),changed,QUALITY_PROTOCOL,cfg)['pending_ids'],['1'])

    def test_missing_variant_reruns_all_variants_and_whole_triplet_is_replaced(self):
        cfg=config();parent=rows(cfg)
        del parent[4]
        plan=plan_resume(items(),parent,QUALITY_PROTOCOL,cfg)
        child=rows(cfg,{'1'})
        child[0]['decision_correct']=True
        merged=merge_resume(plan,parent,child,cfg,parent_job='p',child_job='c',mapper_weights_sha256='fixed')
        self.assertEqual(len(merged),9)
        for row in merged:
            self.assertEqual(row['resume_provenance']['source_job'],'c' if row['id']=='1' else 'p')
            self.assertFalse(row['resume_provenance']['latency_headline_permitted'])

    def test_partial_child_never_borrows_successful_parent_variant(self):
        cfg=config();parent=rows(cfg)
        parent[5]['status']='error'
        plan=plan_resume(items(),parent,QUALITY_PROTOCOL,cfg)
        child=rows(cfg,{'1'})[:1]
        child[0]['status']='error'
        merged=merge_resume(plan,parent,child,cfg,parent_job='p',child_job='c',mapper_weights_sha256='fixed')
        selected=[row for row in merged if row['id']=='1']
        self.assertEqual([row['status'] for row in selected],['error','not_run_time_budget','not_run_time_budget'])
        self.assertTrue(all(r['resume_provenance']['source_job']=='c' for r in selected))

    def test_duplicates_unknowns_label_context_and_config_drift_are_rejected(self):
        cfg=config();parent=rows(cfg)
        mutations=[parent+[parent[0]],copy.deepcopy(parent),copy.deepcopy(parent),copy.deepcopy(parent),copy.deepcopy(parent)]
        mutations[1][0]['id']='unknown'
        mutations[2][0]['gold_answer']='B'
        mutations[3][0]['context_hash']='changed'
        mutations[4][0]['run_config_hash']='changed'
        for changed in mutations:
            with self.subTest(changed=changed[0]):
                with self.assertRaises(ValueError):plan_resume(items(),changed,QUALITY_PROTOCOL,cfg)
        bad=config();bad['freegen_max_new_tokens']=8
        with self.assertRaises(ValueError):plan_resume(items(),parent,QUALITY_PROTOCOL,bad)

    def test_missing_parent_and_missing_child_keep_all_fixed_items(self):
        plan=plan_resume(items(),[],QUALITY_PROTOCOL,None)
        merged=merge_resume(plan,[],[],None,parent_job='p',child_job='c',mapper_weights_sha256='fixed')
        self.assertEqual(len(merged),9)
        self.assertTrue(all(row['status']=='not_run_time_budget' for row in merged))
        from kvverify.reference_eval import _summary
        self.assertFalse(_summary(merged,('decision_correct',))['fully_completed'])

    def test_plan_tampering_and_rerun_of_complete_id_rejected(self):
        cfg=config();parent=rows(cfg);parent[5]['status']='error'
        plan=plan_resume(items(),parent,QUALITY_PROTOCOL,cfg)
        with self.assertRaises(ValueError):
            merge_resume(plan,parent,rows(cfg,{'0'}),cfg,parent_job='p',child_job='c',mapper_weights_sha256='fixed')
        plan['pending_ids']=[]
        with self.assertRaises(ValueError):
            merge_resume(plan,parent,[],None,parent_job='p',child_job='c',mapper_weights_sha256='fixed')

    def test_numerical_and_selected_checkpoint_configs_must_match(self):
        cfg=config()
        verify_evaluation_config(cfg,mapper_weights_sha256='fixed',runtime_identities=cfg['runtime'],quality=True)
        for field,value in [('mapper',{'weights_sha256':'other'}),('runtime',{}),('prompt_tokens_limit',8192),('freegen_max_new_tokens',8)]:
            bad={**cfg,field:value}
            with self.assertRaises(ValueError):verify_evaluation_config(bad,mapper_weights_sha256='fixed',runtime_identities=cfg['runtime'],quality=True)


if __name__=='__main__':unittest.main()
