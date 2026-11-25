"""Cloud packaging/request guards: no SDK, model, network or AWS calls."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cloud.full_submit import (BUCKET, IMAGE, build_request, configuration, digest,
                               make_bundle, prepared_files, resume_uri, valid_relative,
                               smoke_gate, tested_source_hashes, compute_budget)


class FullCloudTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def dataset(self):
        files, splits = {}, {}
        for name, official, split in (('train', 'train', 'calibration'),
                                      ('mapper_validation', 'train', 'calibration'), ('dev', 'dev', 'eval')):
            path = self.root/(name+'.jsonl')
            path.write_text(json.dumps({'id': name})+'\n')
            files[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            splits[name] = {'file': path.name, 'sha256': digest(path),
                            'official_split': official, 'record_split': split}
        manifest = {'schema_version': 1, 'files': files, 'splits': splits}
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def save_manifest(self, manifest):
        (self.root/'manifest.json').write_text(json.dumps(manifest))

    def test_prepared_checksums_and_split_preservation(self):
        self.dataset()
        _, files = prepared_files(self.root)
        self.assertEqual(set(files), {'inputs/manifest.json', 'inputs/train.jsonl',
                                     'inputs/mapper_validation.jsonl', 'inputs/dev.jsonl'})

    def test_modified_data_rejected(self):
        self.dataset()
        (self.root/'dev.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prepared_files(self.root)

    def test_test_data_cannot_be_disguised_as_dev_in_manifest(self):
        manifest = self.dataset()
        manifest['splits']['dev']['official_split'] = 'test'
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'splits'):
            prepared_files(self.root)

    def test_unlisted_extra_credentials_cannot_enter_bundle(self):
        manifest = self.dataset()
        (self.root/'token.json').write_text('sensitive')
        manifest['files']['token.json'] = {'sha256': digest(self.root/'token.json')}
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Only declared'):
            prepared_files(self.root)

    def test_symlink_data_rejected(self):
        manifest = self.dataset()
        value = (self.root/'train.jsonl').read_text()
        (self.root/'train.jsonl').unlink()
        target = self.root/'outside.jsonl'
        target.write_text(value)
        (self.root/'train.jsonl').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'ordinary files'):
            prepared_files(self.root)

    def test_split_and_file_hash_disagreement_rejected(self):
        manifest = self.dataset()
        manifest['splits']['dev']['sha256'] = '0'*64
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            prepared_files(self.root)

    def test_unsafe_relative_names_rejected(self):
        for name in ('/tmp/x.py', '../x.py', 'a/../../x.py', '.env', 'x\\y.py', 'a//x.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                valid_relative(name)

    def test_data_attribution_is_supported(self):
        self.assertEqual(valid_relative('inputs/SOURCE_README.md'), 'inputs/SOURCE_README.md')
        self.assertEqual(valid_relative('inputs/LICENSE'), 'inputs/LICENSE')

    def test_one_gpu_digest_pinned_runtime_and_checkpoint_sync(self):
        config = configuration('train-eval', 'a'*64)
        request = build_request('sova-kv-full-fixture', config)
        self.assertEqual(request['ResourceConfig']['InstanceCount'], 1)
        self.assertEqual(request['ResourceConfig']['InstanceType'], 'ml.g5.2xlarge')
        self.assertIn('@sha256:', request['AlgorithmSpecification']['TrainingImage'])
        self.assertEqual(request['StoppingCondition']['MaxRuntimeInSeconds'], 7200)
        self.assertEqual(request['CheckpointConfig']['LocalPath'], '/opt/ml/checkpoints')
        self.assertEqual(request['Environment']['KV_WORKER_TIMEOUT'], '6900')
        self.assertFalse(request['EnableManagedSpotTraining'])

    def test_runtime_cannot_silently_expand(self):
        for seconds in (0, 599, 7201, 86400, 7200.0, True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                configuration('train-eval', 'a'*64, seconds)

    def test_eval_only_requires_resume(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            configuration('eval', 'a'*64)
        config = configuration('eval', 'a'*64, resume=f's3://{BUCKET}/kv-transfer/old/checkpoints')
        request = build_request('sova-kv-full-resume', config)
        self.assertEqual(request['InputDataConfig'][1]['ChannelName'], 'resume')
        self.assertEqual(config['resume_dir'], '/opt/ml/input/data/resume')

    def test_foreign_or_unsafe_resume_prefix_rejected(self):
        for value in ('https://example.com/x', 's3://foreign/kv-transfer/x',
                      f's3://{BUCKET}/other-project/x', f's3://{BUCKET}/kv-transfer/../secret',
                      f's3://{BUCKET}/kv-transfer/x?credentials=yes'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resume_uri(value)

    def test_bundle_is_byte_reproducible_and_content_hashed(self):
        self.dataset()
        _, files = prepared_files(self.root)
        config = configuration('train-eval', digest(self.root/'manifest.json'))
        first, second = self.root/'first.tar.gz', self.root/'second.tar.gz'
        hashes = make_bundle(first, files, config)
        os.utime(self.root/'train.jsonl', (999999999, 999999999))
        self.assertEqual(hashes, make_bundle(second, dict(reversed(list(files.items()))), config))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with tarfile.open(first) as archive:
            self.assertEqual(archive.getnames(), sorted(hashes))
            for member in archive:
                self.assertEqual(member.mtime, 0)
                self.assertEqual(hashlib.sha256(archive.extractfile(member).read()).hexdigest(), hashes[member.name])

    def full_smoke(self):
        for name in ('kvverify/backend.py', 'cloud/full_worker.py', 'cloud/full_bootstrap.sh', 'tests/test_mapper.py'):
            path = self.root/name
            path.parent.mkdir(exist_ok=True)
            path.write_text('# tested source\n')
        report = {'status': 'passed', 'tests_run': 50, 'skipped': 0, 'errors': 0, 'failures': 0,
                  'file_sha256': tested_source_hashes(self.root)}
        path = self.root/'reports/smoke-full-v1/test-results.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        return path, report

    def test_current_full_suite_is_required(self):
        self.full_smoke()
        with patch('cloud.full_submit.ROOT', self.root):
            result = smoke_gate()
        self.assertEqual(result['tests_run'], 50)

    def test_modified_code_invalidates_full_suite_evidence(self):
        self.full_smoke()
        (self.root/'cloud/full_worker.py').write_text('# changed after tests\n')
        with patch('cloud.full_submit.ROOT', self.root), self.assertRaisesRegex(ValueError, 'stale'):
            smoke_gate()

    def test_new_untested_module_invalidates_full_suite_evidence(self):
        self.full_smoke()
        (self.root/'kvverify/new_mapper.py').write_text('# untested\n')
        with patch('cloud.full_submit.ROOT', self.root), self.assertRaisesRegex(ValueError, 'stale'):
            smoke_gate()

    def test_skipped_or_insufficient_full_suite_is_rejected(self):
        path, report = self.full_smoke()
        for update in ({'skipped': 1}, {'tests_run': 49}, {'errors': 1}, {'failures': 1}):
            path.write_text(json.dumps({**report, **update}))
            with patch('cloud.full_submit.ROOT', self.root), self.assertRaisesRegex(ValueError, 'complete current'):
                smoke_gate()

    def test_corrective_run_includes_prior_compute_in_guard(self):
        estimate, total = compute_budget(1.515, 7200, .1515, 3.25)
        self.assertAlmostEqual(estimate, 3.03)
        self.assertAlmostEqual(total, 3.1815)
        with self.assertRaisesRegex(RuntimeError, 'Prior compute'):
            compute_budget(1.515, 7200, .30, 3.25)

    def test_nonfinite_or_negative_prior_compute_is_rejected(self):
        for prior in (-.01, 3.26, float('nan'), float('inf')):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                compute_budget(1.515, 7200, prior, 3.25)


if __name__ == '__main__':
    unittest.main()
