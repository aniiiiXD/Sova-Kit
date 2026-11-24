"""The Aegis subprocess bridge stores evidence without inventing executable history."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from switchit.aegis_bridge import EvidenceBridge
from switchit.contracts import ContractError
from switchit.module_demo import synthetic_events

ROOT = Path(__file__).resolve().parents[1]


def task_snapshot():
    bundle = json.loads((ROOT / 'tests/fixtures/aegis-task.json').read_text())
    return {'format': 'aegis-snapshot', 'version': 1, 'bundle': bundle,
            'runs': [{'id': 'run-demo', 'task_id': 'task-demo', 'agent': 'codex', 'status': 'completed'}],
            'events': [{**row, 'task_id': 'task-demo'} for row in bundle['messages']],
            'verifications': bundle['verifications'], 'coverage': [{'resource': 'events', 'status': 'complete'}]}


class AegisBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='switchit-aegis-bridge-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bridge = EvidenceBridge(self.root / 'store')

    def request(self, operation, **kwargs):
        return self.bridge.handle({'bridge_version': 1, 'operation': operation, **kwargs})

    def captured(self):
        return self.request('capture', snapshot=task_snapshot())

    def built(self):
        self.captured()
        self.request('attach_trace', task_id='task-demo', run_id='run-demo', events=synthetic_events())
        return self.request('build_workflow', task_id='task-demo')

    def reviewed(self, text='The account is active'):
        built = self.built()
        workflow_id = built['workflows'][0]['workflow_id']
        return self.request('review_workflow', task_id='task-demo', workflow_id=workflow_id,
                            reviewer='synthetic-reviewer', checks=[{'type': 'output_equals', 'text': text}])

    def test_capture_is_idempotent_and_metadata_cannot_become_workflow(self):
        first = self.captured()
        second = self.captured()
        self.assertEqual(first['task']['revision'], second['task']['revision'])
        self.assertEqual(first['task']['event_count'], second['task']['event_count'])
        self.assertEqual(second['task']['evidence_status'], 'evidence_only')
        self.assertEqual(second['task']['native_event_count'], 0)
        self.assertEqual({e['evidence'] for e in second['events']}, {'metadata_only'})
        self.assertEqual(self.request('list')['tasks'][0]['task_id'], 'task-demo')
        self.assertEqual(len(list((self.root / 'store/tasks').glob('*/captures/*.json'))), 1)
        with self.assertRaisesRegex(ContractError, 'native model/tool trace'):
            self.request('build_workflow', task_id='task-demo')

    def test_timestamp_run_ids_remain_exact_and_do_not_inflate_run_count(self):
        snapshot = task_snapshot()
        run_id = 'run-1794340470123-2'
        snapshot['runs'][0]['id'] = run_id
        for row in snapshot['events']:
            row['run_id'] = run_id
        captured = self.request('capture', snapshot=snapshot)
        self.assertEqual(captured['task']['run_count'], 1)
        attached = self.request('attach_trace', task_id='task-demo', run_id=run_id, events=synthetic_events())
        self.assertEqual(attached['task']['run_count'], 1)

    def test_native_attachment_bound_to_task_runs_and_retains_metadata(self):
        captured = self.captured()
        with self.assertRaisesRegex(ContractError, 'run_id'):
            self.request('attach_trace', task_id='task-demo', run_id='another-run', events=synthetic_events())
        attached = self.request('attach_trace', task_id='task-demo', run_id='run-demo', events=synthetic_events())
        self.assertEqual(attached['task']['native_event_count'], 10)
        self.assertEqual(attached['task']['event_count'], captured['task']['event_count'] + 10)
        self.assertEqual(attached['task']['evidence_status'], 'trace_attached')
        native = [e for e in attached['events'] if e['evidence'] != 'metadata_only']
        self.assertTrue(all(e['source']['aegis_run_id'] == 'run-demo' for e in native))
        self.assertTrue(all('caller_supplied_trace' in e['uncertainty'] for e in native))
        repeated = self.request('attach_trace', task_id='task-demo', run_id='run-demo', events=synthetic_events())
        self.assertEqual(attached['task']['revision'], repeated['task']['revision'])

    def test_review_requires_outcome_check_and_preserves_failing_results(self):
        built = self.built()
        workflow_id = built['workflows'][0]['workflow_id']
        self.assertEqual(built['workflows'][0]['status'], 'draft')
        with self.assertRaisesRegex(ContractError, 'outcome check'):
            self.request('review_workflow', task_id='task-demo', workflow_id=workflow_id,
                         reviewer='reviewer', checks=[{'type': 'no_error'}])
        reviewed = self.request('review_workflow', task_id='task-demo', workflow_id=workflow_id,
                               reviewer='reviewer', checks=[{'type': 'output_equals', 'text': 'wrong answer'}])
        entry = reviewed['workflows'][0]
        self.assertEqual(entry['status'], 'reviewed')
        self.assertEqual(entry['verdicts']['fail'], 2)
        self.assertEqual(entry['evaluation']['execution'], 'recorded_fixtures_only')
        self.assertFalse(entry['stale'])

    def test_review_export_and_fixture_verifier_roundtrip_from_subprocess(self):
        reviewed = self.reviewed()
        entry = reviewed['workflows'][0]
        self.assertEqual(entry['verdicts']['pass'], 2)
        export = self.request('export_skill', task_id='task-demo', workflow_id=entry['workflow_id'])['export']
        path = Path(export['path'])
        self.assertTrue(path.is_relative_to((self.root / 'store').resolve()))
        self.assertFalse((path / 'references/cases.json').exists())
        self.assertFalse(export['includes_private_evidence'])
        with_cases = self.request('export_skill', task_id='task-demo', workflow_id=entry['workflow_id'], include_cases=True)['export']
        verified = subprocess.run([sys.executable, str(Path(with_cases['path']) / 'scripts/verify.py')],
                                  cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True, timeout=15)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        self.assertEqual(len(json.loads(verified.stdout)), 2)
        self.assertEqual(self.request('export_skill', task_id='task-demo', workflow_id=entry['workflow_id'])['export']['path'], str(path))

    def test_changed_trace_revision_stales_review_and_retains_history(self):
        reviewed = self.reviewed()
        original = reviewed['workflows'][0]
        changed = copy.deepcopy(synthetic_events())
        changed[-1]['payload']['response']['text'] = 'different answer'
        attached = self.request('attach_trace', task_id='task-demo', run_id='run-demo', events=changed)
        self.assertTrue(attached['workflows'][0]['stale'])
        with self.assertRaisesRegex(ContractError, 'evidence changed'):
            self.request('export_skill', task_id='task-demo', workflow_id=original['workflow_id'])
        stored = self.request('get', task_id='task-demo', artifact='store')['document']
        self.assertGreater(len(stored['events']), attached['task']['event_count'])
        self.assertTrue(list((self.root / 'store/tasks').glob('*/reviews/*.json')))

    def test_configured_native_source_is_explicit_and_unknown_paths_rejected(self):
        source = self.root / 'selected-session.jsonl'
        source.write_text('\n'.join(json.dumps(row) for row in [
            {'type': 'session_meta', 'payload': {'id': 'selected-native-session'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Say hello'}]}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'hello'}]}},
        ]) + '\n')
        self.bridge = EvidenceBridge(self.root / 'store', {'selected': {'agent': 'codex', 'path': str(source), 'label': 'Selected session'}})
        self.captured()
        self.assertEqual(self.request('sources')['sources'], [{'source_id': 'selected', 'agent': 'codex', 'label': 'Selected session'}])
        with self.assertRaisesRegex(ContractError, 'not configured'):
            self.request('attach_trace', task_id='task-demo', source_id=str(source))
        attached = self.request('attach_trace', task_id='task-demo', source_id='selected', run_id='run-demo')
        self.assertEqual(attached['task']['native_event_count'], 2)
        self.assertTrue(any('selected-native-session' in e['trace_id'] for e in attached['events']))


    def test_unchanged_build_keeps_reviewed_revision_exportable(self):
        reviewed = self.reviewed()
        original = reviewed['workflows'][0]
        rebuilt = self.request('build_workflow', task_id='task-demo')
        self.assertEqual(len(rebuilt['workflows']), 1)
        self.assertEqual(rebuilt['workflows'][0]['status'], 'reviewed')
        self.assertEqual(rebuilt['workflows'][0]['revision'], original['revision'])
        self.assertEqual(rebuilt['workflows'][0]['verdicts']['pass'], 2)
        exported = self.request('export_skill', task_id='task-demo', workflow_id=original['workflow_id'], revision=original['revision'])
        self.assertEqual(exported['export']['manifest']['workflow_revision'], original['revision'])
        with self.assertRaisesRegex(ContractError, 'not stored'):
            self.request('export_skill', task_id='task-demo', workflow_id=original['workflow_id'], revision='missing-revision')

    def test_store_symlink_components_and_leaves_cannot_read_or_write_external(self):
        self.captured()
        folder = next((self.root / 'store/tasks').iterdir())
        sentinel = self.root / 'external-sentinel.json'
        sentinel.write_text('{"private":"do not read or overwrite"}')
        original = sentinel.read_bytes()
        metadata = folder / 'metadata.json'
        saved = metadata.read_bytes()
        metadata.unlink()
        metadata.symlink_to(sentinel)
        with self.assertRaisesRegex(ContractError, 'symlink'):
            self.request('get', task_id='task-demo')
        with self.assertRaisesRegex(ContractError, 'symlink'):
            self.captured()
        self.assertEqual(sentinel.read_bytes(), original)
        metadata.unlink();metadata.write_bytes(saved)
        captures = folder / 'captures'
        captures.rename(folder / 'captures-original')
        external = self.root / 'external-captures';external.mkdir()
        captures.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, 'symlink'):
            self.captured()
        self.assertEqual(list(external.iterdir()), [])
        self.assertEqual(sentinel.read_bytes(), original)

    def test_linked_export_parent_cannot_write_external_directory(self):
        reviewed = self.reviewed()
        folder = next((self.root / 'store/tasks').iterdir())
        external = self.root / 'external-exports';external.mkdir()
        (folder / 'exports').symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, 'symlink'):
            self.request('export_skill', task_id='task-demo', workflow_id=reviewed['workflows'][0]['workflow_id'])
        self.assertEqual(list(external.iterdir()), [])

    def test_request_file_protocol_errors_and_artifact_allowlist(self):
        request = self.root / 'request.json'
        request.write_text(json.dumps({'bridge_version': 1, 'operation': 'capture', 'snapshot': task_snapshot()}))
        command = [sys.executable, '-m', 'switchit.aegis_bridge', '--store', str(self.root / 'store'), '--request-file', str(request)]
        response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(response.returncode, 0, response.stderr)
        document = json.loads(response.stdout)
        self.assertTrue(document['ok'])
        self.assertEqual(document['bridge_version'], 1)
        request.write_text('{"bridge_version":1,"bridge_version":1,"operation":"list"}')
        response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(response.returncode, 0)
        self.assertFalse(json.loads(response.stdout)['ok'])
        with self.assertRaisesRegex(ContractError, 'unknown evidence artifact'):
            self.request('get', task_id='task-demo', artifact='../../private')


if __name__ == '__main__':
    unittest.main()
