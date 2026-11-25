"""Pre-GPU dependency closure catches omitted nested-function imports."""
from pathlib import Path
import tempfile
import unittest
from cloud.package_integrity import verify_worker_imports


class WorkerPackageTests(unittest.TestCase):
    def test_matched_fit_bundle_has_its_sampling_and_projection_dependencies(self):
        from cloud.matched_fit_submit import sources
        from cloud.package_integrity import verify_worker_imports
        closure=verify_worker_imports(sources(),'cloud/matched_fit_worker.py')
        self.assertIn('kvverify/affine_cache.py',closure)
        self.assertIn('kvverify/key_geometry.py',closure)

    def test_layer_readout_worker_bundle_has_every_new_dependency(self):
        from cloud.layer_readout_submit import sources
        from cloud.package_integrity import verify_worker_imports
        closure=verify_worker_imports(sources(),'cloud/layer_readout_worker.py')
        self.assertIn('kvverify/layer_readout.py',closure)
        self.assertIn('kvverify/state_diagnostic.py',closure)

    def test_missing_nested_import_is_rejected_before_submission(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'worker.py';p.write_text('def main():\n    from kvverify.activation_offload import selective_saved_tensors_cpu\n')
            with self.assertRaisesRegex(ValueError,'activation_offload'):
                verify_worker_imports({'cloud/worker.py':p},'cloud/worker.py')

    def test_actual_continuation_bundle_is_closed_over_worker_dependencies(self):
        from cloud.continuation_submit import sources
        dependencies=verify_worker_imports(sources(),'cloud/continuation_worker.py')
        self.assertIn('kvverify/activation_offload.py',dependencies)
        self.assertIn('kvverify/affine_cache.py',dependencies)


if __name__=='__main__':unittest.main()
