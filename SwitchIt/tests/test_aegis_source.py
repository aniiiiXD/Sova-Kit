"""Aegis compatibility fixtures, real loopback HTTP transport, and evidence semantics."""
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from switchit.aegis_source import AegisClient, AegisReadError, collect_aegis
from switchit.cli import main
from switchit.event_store import empty_event_store, merge_events
from switchit.execution_graph import build_execution_graph
from switchit.importers.aegis import import_aegis
from switchit.import_pipeline import active_events
from switchit.mcp_server import handle_message
from switchit.workflow_commands import build_bundle


def bundle():
    return json.loads((Path(__file__).parent / 'fixtures' / 'aegis-task.json').read_text())


def snapshot():
    data = bundle()
    events = [{**row, 'task_id': 'task-demo'} for row in data['messages']]
    events[1]['type'] = 'agent.message.completed'
    events.extend([
        {'id': 'command-start', 'task_id': 'task-demo', 'run_id': 'run-demo', 'agent': 'codex',
         'type': 'command.started', 'content': 'python3 -m unittest', 'timestamp': 1794280005, 'sequence': 2},
        {'id': 'command-end', 'task_id': 'task-demo', 'run_id': 'run-demo', 'agent': 'codex',
         'type': 'command.completed', 'content': 'python3 -m unittest', 'timestamp': 1794280009, 'sequence': 3},
    ])
    return {'format': 'aegis-snapshot', 'version': 1, 'bundle': data,
            'runs': [{'id': 'run-demo', 'task_id': 'task-demo', 'agent': 'codex', 'status': 'completed',
                      'started_at': 1794280000, 'finished_at': 1794280010, 'external_session_id': 'provider-thread', 'model': None}],
            'events': events,
            'verifications': [{**data['verifications'][0], 'task_id': 'task-demo', 'run_id': 'run-demo',
                               'freshness': 'current', 'started_at': 1794280011, 'finished_at': 1794280012}],
            'coverage': [{'resource': 'events', 'status': 'complete', 'records': 4}]}


@contextlib.contextmanager
def daemon():
    """Serve the inspected Aegis wire format, without starting any agent."""
    data, requests = snapshot(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(('GET', self.path))
            parsed = urlsplit(self.path)
            if parsed.path.endswith('/bundle'):
                result = data['bundle']
            elif parsed.path.endswith('/runs'):
                result = data['runs']
            elif parsed.path.endswith('/verifications'):
                result = data['verifications']
            elif parsed.path == '/api/events':
                cursor = int(parse_qs(parsed.query)['before_sequence'][0])
                # Return only two items to exercise multiple real HTTP requests.
                rows = sorted([r for r in data['events'] if r['sequence'] < cursor], key=lambda r: r['sequence'])
                selected = rows[-2:]
                result = {'events': selected, 'has_more': len(rows) > 2,
                          'next_cursor': selected[0]['sequence'] if selected else 0}
            elif parsed.path == '/api/redirect':
                self.send_response(302)
                self.send_header('Location', '/api/tasks/task-demo/bundle')
                self.end_headers()
                return
            elif parsed.path == '/api/error':
                self.send_error(500, 'sensitive server body')
                return
            elif parsed.path == '/api/bad-json':
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'not json')
                return
            else:
                self.send_error(404)
                return
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown(); server.server_close(); thread.join()


class AegisImportTests(unittest.TestCase):
    def test_bundle_preserves_context_and_does_not_invent_runs_or_replay(self):
        result = import_aegis(bundle())
        self.assertEqual(len(result['events']), 5)
        self.assertFalse(result['replay_ready'])
        self.assertEqual({e['evidence'] for e in result['events']}, {'metadata_only'})
        self.assertEqual({d['reason'] for d in result['diagnostics']}, {'run_not_imported'})
        self.assertFalse(any(e['kind'] in ('model_call', 'tool_call', 'tool_result', 'approval') for e in result['events']))
        self.assertEqual(result['events'][0]['payload']['context']['goal'], 'Fix search')
        self.assertEqual(build_execution_graph(result['events'])['issues'], [])

    def test_snapshot_hierarchy_sequence_and_historical_verifications(self):
        result = import_aegis(snapshot())
        self.assertEqual(len(result['events']), 8)
        graph = build_execution_graph(result['events'])
        self.assertEqual(graph['issues'], [])
        run = next(e for e in result['events'] if e['source']['record_type'] == 'run')
        messages = [e for e in result['events'] if e['source']['record_type'] == 'event']
        self.assertTrue(all(e['parent_id'] == run['event_id'] for e in messages))
        self.assertEqual([e['payload']['sequence'] for e in messages], [1, 2, 3, 4])
        verification = next(e for e in result['events'] if e['source']['record_type'] == 'verification')
        self.assertEqual(verification['payload']['freshness'], 'historical')
        self.assertEqual(verification['payload']['observed_freshness'], 'current')
        with tempfile.TemporaryDirectory() as folder:
            built = build_bundle({'events': result['events']}, folder)
            self.assertEqual(built['drafts'], 0)

    def test_reimport_is_idempotent_and_changed_revision_keeps_history(self):
        first = import_aegis(snapshot())
        store = merge_events(empty_event_store(), first['events'])
        self.assertEqual(merge_events(store, first['events']), store)
        changed = snapshot(); changed['events'][0]['content'] = 'Updated goal'
        second = import_aegis(changed)
        self.assertNotEqual(first['revision'], second['revision'])
        self.assertEqual(first['source_id'], second['source_id'])
        self.assertEqual(len(merge_events(store, second['events'])['events']), 16)
        changed['bundle']['source_origin'] = 'another-store'
        self.assertNotEqual(first['source_id'], import_aegis(changed)['source_id'])

    def test_redaction_unknown_events_and_unresolved_runs(self):
        data = snapshot(); data['events'][0]['content'] = 'api_key=supersecret'
        data['events'][0]['type'] = 'future.event'
        data['events'][0]['run_id'] = 'missing-run'
        result = import_aegis(data)
        self.assertNotIn('supersecret', json.dumps(result))
        self.assertEqual(result['diagnostics'][0]['reason'], 'run_not_imported')
        self.assertEqual(build_execution_graph(result['events'])['issues'], [])

    def test_malformed_and_cross_task_records_fail(self):
        for mutate in (
            lambda d: d.update(version=True),
            lambda d: d['events'].append(copy.deepcopy(d['events'][0])),
            lambda d: d['runs'][0].update(task_id='other'),
            lambda d: d['events'][0].update(task_id='other'),
            lambda d: d['events'][0].update(sequence=True),
            lambda d: d['verifications'][0].update(command='shell string'),
            lambda d: d['bundle'].update(source_origin=''),
        ):
            with self.subTest(mutate=mutate):
                data = snapshot(); mutate(data)
                with self.assertRaises(ValueError):
                    import_aegis(data)

    def test_cli_mcp_revision_store_and_no_implicit_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'bundle.json'; output = root / 'out'
            source.write_text(json.dumps(bundle()))
            args = ['sources', 'aegis', '--file', str(source), '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main([*args, '--store', str(output / 'store.json')]), 0)
            self.assertEqual(len(active_events(json.loads((output / 'store.json').read_text()))), 5)
            response = handle_message({'id': 1, 'method': 'tools/call', 'params': {
                'name': 'switchit_sources_aegis', 'arguments': {'file': str(source), 'output': str(root / 'mcp')}}})
            self.assertFalse(response['result']['isError'], response)
            with patch('switchit.aegis_source.collect_aegis') as collect:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([*args, '--url', 'http://127.0.0.1:1234'])
                collect.assert_not_called()

    def test_platform_import_is_evidence_only(self):
        from switchit.platform.jobs import JobManager
        with tempfile.TemporaryDirectory() as folder:
            manager = JobManager(folder)
            try:
                job = manager.submit({'name': 'Synthetic Aegis evidence', 'aegis': snapshot()})
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    result = manager.get(job['id'])
                    if result['status'] not in ('queued', 'running'):
                        break
                    time.sleep(.01)
                self.assertEqual(result['status'], 'evidence_only', result)
                self.assertIn('graph.json', result['artifacts'])
                self.assertNotIn('workflow.json', result['artifacts'])
            finally:
                manager.close()

    def test_platform_rejects_mixed_inputs_and_retains_coverage(self):
        from switchit.platform.jobs import validate_submission
        with self.assertRaises(ValueError):
            validate_submission({'aegis': bundle(), 'events': []})
        data = snapshot(); data['coverage'][0]['status'] = 'limited'
        _, events, _ = validate_submission({'aegis': data})
        self.assertEqual(events[0]['payload']['coverage'][0]['status'], 'limited')
        self.assertTrue(events[0]['payload']['limitations'])


class AegisCollectionTests(unittest.TestCase):
    def test_real_loopback_transport_pagination_and_cli(self):
        with daemon() as (url, requests), tempfile.TemporaryDirectory() as folder:
            result = collect_aegis(url, 'task-demo')
            self.assertEqual(len(result['events']), 4)
            self.assertEqual(result['coverage'][0]['status'], 'complete')
            self.assertEqual(len(import_aegis(result)['events']), 8)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['sources', 'aegis', '--url', url, '--task', 'task-demo', '--output', folder]), 0)
            self.assertEqual({method for method, _ in requests}, {'GET'})
            self.assertTrue(all('/api/tasks/task-demo/' in path or path.startswith('/api/events?') for _, path in requests))

    def test_limited_history_is_saved_but_cli_returns_nonzero(self):
        with daemon() as (url, _), tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['sources', 'aegis', '--url', url, '--task', 'task-demo',
                                       '--max-pages', '1', '--output', folder]), 1)
            self.assertEqual(json.loads((Path(folder) / 'diagnostics.json').read_text())['coverage'][0]['status'], 'limited')

    def test_external_urls_credentials_redirects_errors_and_size_limits(self):
        for url in ('https://example.com', 'http://example.com:80', 'http://127.0.0.1:80@evil.test:80',
                    'http://localhost:80/path', 'http://localhost:80?token=secret', 'http://localhost',
                    'http://127.0.0.1:80#fragment'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                AegisClient(url)
        with daemon() as (url, _):
            client = AegisClient(url)
            for path in ('/api/redirect', '/api/error', '/api/bad-json'):
                with self.subTest(path=path), self.assertRaises(AegisReadError) as error:
                    client.get(path)
                self.assertNotIn('sensitive server body', str(error.exception))
            with self.assertRaisesRegex(ValueError, 'size limit'):
                AegisClient(url, max_bytes=8).get('/api/tasks/task-demo/bundle')

    def test_invalid_cursor_and_cross_page_duplicates_fail(self):
        data = snapshot()
        for page in (
            {'events': data['events'], 'has_more': True, 'next_cursor': 0},
            {'events': [], 'has_more': True, 'next_cursor': 1},
            {'events': data['events'], 'has_more': True, 'next_cursor': 2},
        ):
            client = AegisClient('http://127.0.0.1:1234')
            with patch.object(client, 'get', side_effect=[data['bundle'], data['runs'], data['verifications'], page]):
                with self.assertRaises(ValueError):
                    collect_aegis(client.url, 'task-demo', client=client)


if __name__ == '__main__':
    unittest.main()
