import json
from pathlib import Path
import tempfile
import unittest
import re
from unittest.mock import patch

from cloud import recovery_submit as submit


class RecoveryCloudTests(unittest.TestCase):
    def test_every_mode_has_a_sagemaker_valid_job_name(self):
        for mode in submit.MODE_MODULES:
            name=submit.job_name(mode)
            self.assertIsNotNone(re.fullmatch(r'[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}',name),name)
            self.assertLessEqual(len(name),63)
            self.assertNotIn('_',name)

    def test_probe_limits_and_selected_mapper(self):
        cfg=submit.config('probe',1800)
        request=submit.request('sova-kv-probe-test',cfg)
        self.assertEqual(request['StoppingCondition']['MaxRuntimeInSeconds'],1800)
        self.assertEqual(request['ResourceConfig']['InstanceCount'],1)
        self.assertEqual(request['InputDataConfig'][1]['DataSource']['S3DataSource']['S3Uri'],submit.SAVED_MAPPER)
        self.assertFalse(request['EnableManagedSpotTraining'])

    def test_limit_cannot_silently_expand(self):
        for mode,seconds in [('probe',1801),('repair',3601),('probe',599)]:
            with self.assertRaises(ValueError):submit.config(mode,seconds)

    def test_probe_whitelist_excludes_pending_repair(self):
        names=submit.source_files('probe')
        self.assertNotIn('kvverify/qa_training.py',names)
        self.assertNotIn('kvverify/reference_eval.py',names)
        self.assertIn('kvverify/boundary_probe.py',names)
        self.assertFalse(any('reports/'in n or '.env'in n for n in names))

    def test_smoke_rejects_changed_or_unexecuted_code(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'smoke.json'
            report={'status':'passed','mode':'probe','tests_run':50,'failures':0,'errors':0,'skipped':0,
                    'file_sha256':{'a':'original'},'sources_unchanged_during_tests':True}
            path.write_text(json.dumps(report))
            with patch.object(submit,'tested_hashes',return_value={'a':'original'}):submit.verify_smoke('probe',path)
            with patch.object(submit,'tested_hashes',return_value={'a':'changed'}):
                with self.assertRaises(ValueError):submit.verify_smoke('probe',path)
            with self.assertRaises(ValueError):submit.verify_smoke('repair',path)

    def test_prior_spend_keeps_both_scoped_jobs_bounded(self):
        estimate,total=submit.compute_budget(1.515,1800,.9363541666666666,3.25)
        self.assertAlmostEqual(estimate,.7575)
        self.assertLess(total+1.515,3.25)

    def test_explicit_repair_goal_cap_preserves_historical_guard_and_finite_bounds(self):
        estimate,total=submit.recovery_budget(1.515,3600,2.1706583335,5.)
        self.assertAlmostEqual(estimate,1.515)
        self.assertAlmostEqual(total,3.6856583335)
        with self.assertRaises(ValueError):submit.compute_budget(1.515,3600,2.1706583335,5.)
        with self.assertRaises(RuntimeError):submit.recovery_budget(1.515,3600,4.,5.)
        for rate,seconds,prior,budget in[(float('nan'),3600,0.,5.),(1.515,3601,0.,5.),
            (1.515,3600,float('inf'),5.),(1.515,3600,0.,5.01)]:
            with self.assertRaises(ValueError):submit.recovery_budget(rate,seconds,prior,budget)

    def test_continuation_rejects_a_mutating_parent_before_any_s3_transfer(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'parent.json'
            path.write_text(json.dumps({'region':submit.REGION,'profile':submit.PROFILE,
                'mode':'repair','job_name':'sova-kv-repair-test','file_sha256':{}}))
            with patch.object(submit,'aws',return_value=json.dumps({'TrainingJobStatus':'InProgress'}))as call:
                with self.assertRaisesRegex(ValueError,'terminal'):submit.continuation_inputs(path,{})
                self.assertEqual(call.call_count,1)

    def test_continuation_rejects_changed_scoring_source_before_transfer(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'parent.json'
            path.write_text(json.dumps({'region':submit.REGION,'profile':submit.PROFILE,
                'mode':'repair','job_name':'sova-kv-repair-test','file_sha256':{}}))
            with patch.object(submit,'aws',return_value=json.dumps({'TrainingJobStatus':'Completed'}))as call:
                with self.assertRaisesRegex(ValueError,'source drift'):submit.continuation_inputs(path,{})
                self.assertEqual(call.call_count,1)

    def test_continuation_cannot_train_or_extend_its_runtime(self):
        cfg=submit.config('eval_resume',1800)
        self.assertFalse(cfg['training_enabled'])
        self.assertIn('kvverify/eval_resume.py',submit.source_files('eval_resume'))
        self.assertIn('cloud/eval_resume_worker.py',submit.source_files('eval_resume'))
        with self.assertRaises(ValueError):submit.config('eval_resume',1801)

    def test_mutation_between_gate_and_packaging_is_rejected(self):
        with patch.object(submit,'source_files',return_value={'kvverify/one.py':Path('unused')}):
            smoke={'file_sha256':{'kvverify/one.py':'original'}}
            submit.verify_packaged_sources('probe',{'kvverify/one.py':'original'},smoke)
            with self.assertRaises(ValueError):
                submit.verify_packaged_sources('probe',{'kvverify/one.py':'changed'},smoke)

    def test_failed_or_partial_oracle_cannot_authorize_repair(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'probe.json'
            data={'protocol':'tail-boundary-v1','oracle_gate':{'status':'passed','matches_native':48}}
            path.write_text(json.dumps(data));submit.verify_probe_summary(path)
            data['oracle_gate']['matches_native']=47;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):submit.verify_probe_summary(path)

    def test_historical_failed_raw_gate_requires_explicit_fresh_worker_gate(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'probe.json'
            data={'protocol':'tail-boundary-v1','oracle_gate':{'status':'failed','attempts':48,'checked_or_failed':48,'matches_native':45},
                'arms':{'oracle_suffix_'+str(t):{'returned':16,'errors':0,'native_first_token_agreement_all_attempts':1.}for t in (1,8,32)}}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):submit.verify_probe_summary(path)
            evidence=submit.verify_probe_summary(path,fresh_oracle_gate=True)
            self.assertFalse(evidence['historical_gate_passed'])
            self.assertEqual(evidence['oracle_gate']['status'],'failed')
            self.assertTrue(submit.config('repair',3600)['require_fresh_native_oracle_gate'])
            data['arms']['oracle_suffix_1']['errors']=1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):submit.verify_probe_summary(path,fresh_oracle_gate=True)

    def test_partial_or_mismatched_precision_cannot_enable_training(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'precision.json'
            data={'status':'completed','settings':{'matmul_tf32':False,'cudnn_tf32':False,'float32_matmul_precision':'highest'},
                'arms':{'fp32_math':{'all_passed':True,'examples':16,'passed':16,'query_block_size':256,
                    'rows':[{'id':str(i),'status':'ok','oracle':{'passed':True}}for i in range(16)]}}}
            path.write_text(json.dumps(data));submit.verify_precision_summary(path)
            data['arms']['fp32_math']['rows'].pop();path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):submit.verify_precision_summary(path)
            data['arms']['fp32_math']['rows'].append({'id':'15','status':'ok','oracle':{'passed':True}})
            data['settings']['matmul_tf32']=True;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):submit.verify_precision_summary(path)


if __name__=='__main__':unittest.main()
