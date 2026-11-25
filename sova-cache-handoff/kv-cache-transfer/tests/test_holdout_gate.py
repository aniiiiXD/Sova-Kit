"""Acceptance stays sealed until the prospectively declared development gate."""
import unittest
from cloud.holdout_submit import promotion,sources,stopped_checkpoint_summary
from cloud.package_integrity import verify_worker_imports


class HoldoutGateTests(unittest.TestCase):
    def test_economics_requires_complete_independent_long_context_quality(self):
        from cloud.holdout_worker import economics_qualification
        base={'planned_items':267,'completed_items':267,'errors':0,'decision_correct':.6}
        mapped={**base,'decision_correct':.55,'native_agreement_decision_answer':.85,
                'completed_transfer_items':267,'completed_native_fallback_items':0}
        quality={'fully_completed':True,'variants':{'source':base,'native':base,'mapper':mapped}}
        self.assertTrue(economics_qualification(quality)['passed'])
        competitor={'planned_items':267,'completed_items':267,'errors':0,'decision_correct':.5}
        self.assertTrue(economics_qualification(quality,competitor)['passed'])
        self.assertFalse(economics_qualification(quality,{**competitor,'decision_correct':.56})['passed'])
        for changes in({'decision_correct':.5},{'native_agreement_decision_answer':.79},
                        {'completed_native_fallback_items':1},{'errors':1}):
            changed={**quality,'variants':{**quality['variants'],'mapper':{**mapped,**changes}}}
            self.assertFalse(economics_qualification(changed)['passed'])

    def test_requires_all_three_metrics_on_the_selected_checkpoint(self):
        row={'step':250,'documents':30,'native_correct':16,'correct':15,'native_agreement':24,'mean_native_kl':.4}
        summary={'selected_step':250,'validation_history':[row]}
        self.assertEqual(promotion(summary),row)
        for changes in({'correct':14},{'native_agreement':23},{'mean_native_kl':.51},{'native_correct':11}):
            with self.assertRaises(ValueError):promotion({**summary,'validation_history':[{**row,**changes}]})
        with self.assertRaises(ValueError):promotion({**summary,'selected_step':0})
        with self.assertRaises(ValueError):promotion({**summary,'validation_history':[{**row,'mean_native_kl':float('nan')}]})

    def test_stopped_checkpoint_requires_matching_validated_identity_and_still_passes_gate(self):
        row={'step':50,'documents':30,'native_correct':16,'correct':15,'native_agreement':24,'mean_native_kl':.4}
        best={**row,'native_prefix_policy':'256','weights_sha256':'frozen'}
        summary=stopped_checkpoint_summary(best,[row])
        self.assertEqual(summary['native_prefix_policy'],'256')
        self.assertIn('not completed',summary['parent_training_state'])
        with self.assertRaises(ValueError):stopped_checkpoint_summary({**best,'mean_native_kl':.3},[row])
        with self.assertRaises(ValueError):stopped_checkpoint_summary({**best,'native_agreement':23},[{**row,'native_agreement':23}])

    def test_acceptance_worker_dependency_closure(self):
        dependencies=verify_worker_imports(sources(),'cloud/holdout_worker.py')
        self.assertIn('kvverify/affine_cache.py',dependencies)
        self.assertIn('kvverify/reference_eval.py',dependencies)


if __name__=='__main__':unittest.main()
