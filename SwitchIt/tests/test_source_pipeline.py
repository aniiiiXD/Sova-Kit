import json
from pathlib import Path
import tempfile
import unittest

from switchit.source_inventory import inventory_sources
from switchit.import_pipeline import import_inventory, active_events
from switchit.importers import codex, claude_code
from switchit.events import events_from_calls
from switchit.normalization import make_call


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.defaults = {'claude-code': [self.root]}
        self.path = self.root/'session.jsonl'
        self.path.write_text(json.dumps({'type': 'user', 'sessionId': 'test', 'message': {'content': 'Hello'}})+'\n')

    def test_inventory_covers_duplicates_limits_and_missing_roots(self):
        (self.root/'duplicate.jsonl').write_bytes(self.path.read_bytes())
        inventory = inventory_sources(self.defaults)
        self.assertEqual({s['status'] for s in inventory['sources']}, {'ready', 'duplicate'})
        limited = inventory_sources(self.defaults, limit=1)
        self.assertFalse(limited['complete'])
        missing = inventory_sources({'codex': [self.root/'missing']})
        self.assertEqual(missing['coverage'][0]['status'], 'missing')

    def test_incremental_import_and_active_revisions(self):
        inventory = inventory_sources(self.defaults)
        first = import_inventory(inventory)
        self.assertEqual(len(active_events(first['store'])), 1)
        unchanged = import_inventory(inventory, first['store'])
        self.assertEqual(first['store'], unchanged['store'])
        with self.path.open('a') as f:
            f.write(json.dumps({'type': 'user', 'sessionId': 'test', 'message': {'content': 'Next task'}})+'\n')
        changed = import_inventory(inventory_sources(self.defaults, previous=inventory), first['store'])
        self.assertEqual(len(active_events(changed['store'])), 2)
        self.assertEqual(len(changed['store']['events']), 3)

    def test_malformed_record_is_reported(self):
        with self.path.open('a') as f:
            f.write('not-json\n')
        result = import_inventory(inventory_sources(self.defaults))
        self.assertEqual(result['diagnostics'][0]['status'], 'partial')
        self.assertEqual(result['diagnostics'][0]['issues'][0]['line'], 2)
        self.assertIn('source_has_invalid_records', result['store']['events'][0]['uncertainty'])

    def test_changed_source_requires_rescan(self):
        inventory = inventory_sources(self.defaults)
        self.path.write_text('{}\n')
        result = import_inventory(inventory)
        self.assertEqual(result['diagnostics'][0]['status'], 'error')
        self.assertEqual(result['store']['events'], [])

    def test_codex_tool_identity_survives_import(self):
        records = [{'type': 'session_meta', 'payload': {'id': 'session'}},
                   {'type': 'response_item', 'payload': {'type': 'function_call', 'call_id': 'tool-1', 'name': 'lookup', 'arguments': '{}'}},
                   {'type': 'response_item', 'payload': {'type': 'function_call_output', 'call_id': 'tool-1', 'output': 'ok'}}]
        self.path.write_text('\n'.join(map(json.dumps, records)))
        calls = [make_call(raw, 'codex', str(self.path)) for raw in codex.import_codex(self.path)]
        events = events_from_calls(calls)
        self.assertEqual([e['tool_call_id'] for e in events if e['kind'] in ('tool_call', 'tool_result')], ['tool-1', 'tool-1'])

    def test_claude_parallel_results_match_ids_not_arrival_order(self):
        records = [{'type': 'assistant', 'message': {'id': 'm', 'content': [
            {'type': 'tool_use', 'id': 'first', 'name': 'one', 'input': {}},
            {'type': 'tool_use', 'id': 'second', 'name': 'two', 'input': {}}]}},
            {'type': 'user', 'message': {'content': [
                {'type': 'tool_result', 'tool_use_id': 'second', 'content': 'second result'},
                {'type': 'tool_result', 'tool_use_id': 'first', 'content': 'first result'}]}}]
        self.path.write_text('\n'.join(map(json.dumps, records)))
        calls = claude_code.import_claude_code(self.path)
        results = [c['request']['messages'][0] for c in calls if c['model'] == 'tool-result']
        self.assertEqual([(r['name'],r['tool_call_id']) for r in results], [('two','second'),('one','first')])
