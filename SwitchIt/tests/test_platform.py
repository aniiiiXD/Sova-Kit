"""Local platform integration: durable evidence, explicit review and HTTP boundaries."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from switchit.platform import create_server
from switchit.platform.jobs import JobConflict, JobManager
from switchit.platform.server import MAX_BODY_BYTES


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = create_server(port=0, data_dir=self.temp.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None, raw=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        data = raw if raw is not None else json.dumps(body) if body is not None else None
        headers = {'Content-Type': 'application/json', **(headers or {})}
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        content = response.read()
        result = json.loads(content) if response.getheader('Content-Type', '').startswith('application/json') else content
        status = response.status
        connection.close()
        return status, result

    def wait_job(self, job_id, expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, job = self.request('GET', '/api/jobs/' + job_id)
            self.assertEqual(status, 200)
            if job['status'] in expected:
                return job
            if job['status'] == 'failed':
                self.fail(job.get('error'))
            time.sleep(.015)
        self.fail(f'Job did not reach {expected}: {job}')

    def submit_demo(self):
        status, job = self.request('POST', '/api/jobs', {'demo': True, 'name': 'Evidence review'})
        self.assertEqual(status, 202)
        return self.wait_job(job['id'], {'needs_review'})

    def test_end_to_end_review_artifacts_and_failed_outcome(self):
        job = self.submit_demo()
        base = '/api/jobs/' + job['id']
        self.assertTrue(job['synthetic'])
        self.assertNotIn('evaluation.json', job['artifacts'])
        status, graph = self.request('GET', base + '/artifacts/graph.json')
        self.assertEqual(status, 200)
        self.assertEqual(len(graph['nodes']), job['counts']['events'])
        self.assertTrue(graph['edges'])
        for review in ({'checks': [{'type': 'output_equals', 'text': 'active'}]},
                       {'reviewer': 'Reviewer', 'checks': [{'type': 'no_error'}]},
                       {'reviewer': 'Reviewer', 'checks': [{'type': 'output_regex', 'pattern': '.*'}]}):
            self.assertEqual(self.request('POST', base + '/review', review)[0], 400)
        review = {'reviewer': 'Fixture test reviewer', 'checks': [{'type': 'output_equals', 'text': 'The account is active'}]}
        self.assertEqual(self.request('POST', base + '/review', review)[0], 202)
        done = self.wait_job(job['id'], {'completed'})
        self.assertEqual(done['verdicts'], {'pass': 2, 'fail': 0, 'error': 0, 'uncertain': 0})
        status, result = self.request('GET', base + '/artifacts/evaluation.json')
        self.assertEqual(status, 200)
        self.assertEqual(result['mode'], 'recorded_fixture_verification')
        self.assertEqual(len(result['trials']), 2)
        workflow = self.request('GET', base + '/artifacts/workflow.json')[1]
        self.assertEqual(workflow['reviewed_by'], review['reviewer'])
        self.assertTrue(workflow['revision'])
        review['checks'][0]['text'] = 'The account is inactive'
        self.assertEqual(self.request('POST', base + '/review', review)[0], 202)
        done = self.wait_job(job['id'], {'completed'})
        self.assertEqual(done['verdicts']['fail'], 2)
        self.assertEqual(done['verdicts']['pass'], 0)

    def test_json_file_input_and_review_bindings(self):
        from switchit.module_demo import synthetic_events
        status, job = self.request('POST', '/api/jobs', {'name': '<script>unsafe</script>', 'events': synthetic_events()})
        self.assertEqual(status, 202)
        job = self.wait_job(job['id'], {'needs_review'})
        base = '/api/jobs/' + job['id']
        self.assertFalse(job['synthetic'])
        review = {'reviewer': 'Human', 'checks': [{'type': 'output_equals', 'text': 'The account is active'}],
                  'inputs': {'task': 'Look up the account status'}, 'outputs': []}
        self.assertEqual(self.request('POST', base + '/review', review)[0], 202)
        self.wait_job(job['id'], {'completed'})
        spec = self.request('GET', base + '/artifacts/workflow.json')[1]
        self.assertEqual(spec['inputs'], review['inputs'])

    def test_http_boundaries(self):
        for raw in ('[]', '{bad', '{"demo":NaN}', '{"demo":1e999}', '{"demo":true,"demo":true}', '{"events": [null]}', '{"demo": true, "command": "ls"}', '{"events": []}'):
            self.assertEqual(self.request('POST', '/api/jobs', raw=raw)[0], 400, raw)
        self.assertEqual(self.request('POST', '/api/jobs', {'demo': True}, {'Content-Type': 'text/plain'})[0], 400)
        self.assertEqual(self.request('POST', '/api/jobs', {'demo': True}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('GET', '/api/jobs', headers={'Host': f'evil.example:{self.port}'})[0], 403)
        self.assertEqual(self.request('GET', '/api/jobs', headers={'Sec-Fetch-Site': 'cross-site'})[0], 403)
        self.assertEqual(self.request('POST', '/api/jobs', raw='{}', headers={'Content-Length': str(MAX_BODY_BYTES + 1)})[0], 413)
        for path in ('/../../etc/passwd', '/api/jobs/../job.json', '/api/jobs/' + 'a'*32 + '/artifacts/%2e%2e%2fjob.json'):
            self.assertEqual(self.request('GET', path)[0], 404)
        self.assertEqual(self.request('GET', '/api/jobs/' + 'a'*32)[0], 404)
        for path in ('/', '/app.js', '/style.css'):
            status, content = self.request('GET', path)
            self.assertEqual(status, 200)
            self.assertGreater(len(content), 100)

    def test_cancel_resume_and_single_owner(self):
        with self.assertRaises(JobConflict):
            JobManager(self.temp.name)
        job = self.submit_demo()
        base = '/api/jobs/' + job['id']
        self.assertEqual(self.request('POST', base + '/cancel', {})[0], 202)
        self.assertEqual(self.request('GET', base)[1]['status'], 'cancelled')
        self.assertEqual(self.request('POST', base + '/resume', {})[0], 202)
        self.wait_job(job['id'], {'needs_review'})
        self.assertEqual(self.request('POST', base + '/resume', {})[0], 409)
        self.assertEqual(self.request('POST', base + '/cancel', {'command': 'x'})[0], 400)

    def test_symlink_artifact_does_not_escape(self):
        job = self.submit_demo()
        directory = Path(self.temp.name) / 'jobs' / job['id']
        (directory / 'evaluation.json').symlink_to(Path(self.temp.name) / 'jobs' / job['id'] / '..' / '..' / '.platform.lock')
        self.assertEqual(self.request('GET', '/api/jobs/' + job['id'] + '/artifacts/evaluation.json')[0], 404)


class DurableJobsTests(unittest.TestCase):
    def test_restart_recovers_interrupted_job_and_saved_input(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory)
            with patch.object(manager, '_enqueue'):
                job = manager.submit({'demo': True})
            manager.close()
            manager = JobManager(directory)
            try:
                self.assertEqual(manager.get(job['id'])['status'], 'interrupted')
                self.assertTrue(json.loads(manager.artifact(job['id'], 'input.json'))['events'])
                manager.resume(job['id'])
                manager.futures[job['id']].result(timeout=5)
                self.assertEqual(manager.get(job['id'])['status'], 'needs_review')
            finally:
                manager.close()

    def test_cancel_during_evaluation_wins_completion_race(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory)
            try:
                job = manager.submit({'demo': True})
                manager.futures[job['id']].result(timeout=5)
                checkpoint = manager._checkpoint
                def cancel_at_complete(job_id, phase, progress, message):
                    checkpoint(job_id, phase, progress, message)
                    if phase == 'complete':
                        manager.cancel(job_id)
                with patch.object(manager, '_checkpoint', side_effect=cancel_at_complete):
                    manager.review(job['id'], {'reviewer': 'Test', 'checks': [{'type': 'output_equals', 'text': 'The account is active'}]})
                    manager.futures[job['id']].result(timeout=5)
                self.assertEqual(manager.get(job['id'])['status'], 'cancelled')
                self.assertNotIn('verdicts', manager.get(job['id']))
            finally:
                manager.close()


if __name__ == '__main__':
    unittest.main()
