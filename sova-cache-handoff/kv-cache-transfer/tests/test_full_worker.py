"""No model imports, downloads, AWS clients, GPU or network in these tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud import full_worker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    directory = root/'inputs'
    directory.mkdir()
    manifest = {'schema_version': 1, 'files': {}, 'splits': {}}
    for index, name in enumerate(('train', 'mapper_validation', 'dev')):
        split = 'eval' if name == 'dev' else 'calibration'
        row = {'id': name+'-question', 'document_id': name+'-document', 'context': name+' unique document',
               'question': 'Which option?', 'choices': {'A': 'yes', 'B': 'no'}, 'answer': 'A', 'split': split}
        path = directory/(name+'.jsonl')
        path.write_text(json.dumps(row)+'\n')
        manifest['files'][path.name] = {'sha256': digest(path)}
        manifest['splits'][name] = {'file': path.name, 'record_split': split, 'examples': 1, 'documents': 1}
    path = directory/'manifest.json'
    path.write_text(json.dumps(manifest))
    return {'data_manifest': 'inputs/manifest.json', 'data_manifest_sha256': digest(path)}


def refresh(root, config, callback):
    path = root/'inputs/manifest.json'
    manifest = json.loads(path.read_text())
    callback(manifest)
    path.write_text(json.dumps(manifest))
    config['data_manifest_sha256'] = digest(path)


class FullWorkerTests(unittest.TestCase):
    def test_frozen_inputs_use_correct_validation_name_and_record_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            output = root/'outputs'
            with patch.object(full_worker, 'ROOT', root), patch.object(full_worker, 'OUT', output):
                result = full_worker.verify_inputs(config)
            self.assertEqual(set(result), {'train', 'mapper_validation', 'dev'})
            self.assertEqual(result['mapper_validation'][0].split, 'calibration')
            self.assertEqual(result['dev'][0].split, 'eval')
            self.assertTrue((output/'dataset-manifest.json').is_file())

    def test_manifest_and_member_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            with patch.object(full_worker, 'ROOT', root), patch.object(full_worker, 'OUT', root/'outputs'):
                wrong = dict(config, data_manifest_sha256='bad')
                with self.assertRaisesRegex(ValueError, 'manifest hash'):
                    full_worker.verify_inputs(wrong)
                with (root/'inputs/train.jsonl').open('a') as stream:
                    stream.write('\n')
                with self.assertRaisesRegex(ValueError, 'file hash'):
                    full_worker.verify_inputs(config)

    def test_leakage_rejected_even_if_file_and_manifest_hashes_are_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            path = root/'inputs/mapper_validation.jsonl'
            row = json.loads(path.read_text())
            row['context'] = 'train unique document'
            path.write_text(json.dumps(row)+'\n')
            refresh(root, config, lambda m: m['files'][path.name].update(sha256=digest(path)))
            with patch.object(full_worker, 'ROOT', root), patch.object(full_worker, 'OUT', root/'outputs'):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    full_worker.verify_inputs(config)
            self.assertFalse((root/'outputs').exists())

    def test_example_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            refresh(root, config, lambda m: m['splits']['dev'].update(examples=2))
            with patch.object(full_worker, 'ROOT', root), patch.object(full_worker, 'OUT', root/'outputs'):
                with self.assertRaisesRegex(ValueError, 'count mismatch'):
                    full_worker.verify_inputs(config)

    def test_budget_expiry_exports_progress_before_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(full_worker, 'OUT', root),
                  patch.object(full_worker, 'CONFIG', {'worker_timeout_seconds': 300}),
                  patch.object(full_worker, 'STARTED', 100.),
                  patch.object(full_worker.time, 'monotonic', return_value=281.),
                  patch.object(full_worker, 'upload') as upload):
                with self.assertRaisesRegex(TimeoutError, 'final upload'):
                    full_worker.notice({'phase': 'training'})
                upload.assert_called_once()
            progress = json.loads((root/'progress.json').read_text())
            self.assertEqual(progress['elapsed_worker_seconds'], 181.)
            self.assertEqual(progress['phase'], 'training')

    def test_checkpoint_notice_copies_to_managed_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source/step-000050'
            source.mkdir(parents=True)
            (source/'manifest.json').write_text('{}')
            with (patch.object(full_worker, 'OUT', root/'output'),
                  patch.object(full_worker, 'CHECKPOINTS', root/'managed'),
                  patch.object(full_worker, 'CONFIG', {}), patch.object(full_worker, 'upload')):
                full_worker.notice({'checkpoint_path': str(source)})
            self.assertTrue((root/'managed/step-000050/manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
