"""Sanitized Codex custom-tool records preserve freeform source semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from switchit.events import events_from_calls, validate_event
from switchit.execution_graph import build_execution_graph
from switchit.fixtures import build_fixtures
from switchit.importers.codex import import_codex
from switchit.normalization import make_call, normalize_tool_call
from switchit.source_inventory import source_digest
from switchit.workflow_patterns import discover_workflows
from switchit.workflow_builder import build_workflow
from switchit.import_pipeline import import_inventory, active_events

FIXTURE = Path(__file__).parent / 'fixtures/codex-custom-tools.jsonl'


class CodexCustomToolTests(unittest.TestCase):
    def test_native_freeform_input_call_identity_and_output_blocks_survive(self):
        raw = import_codex(FIXTURE)
        self.assertEqual(len(raw), 4)
        calls = [make_call(row, 'codex', str(FIXTURE)) for row in raw]
        custom = next(c for c in calls if c['model'] == 'custom-tool-call')
        tool = custom['response']['tool_calls'][0]
        original = json.loads(FIXTURE.read_text().splitlines()[2])['payload']['input']
        self.assertEqual(tool['type'], 'custom')
        self.assertEqual(tool['input_format'], 'freeform')
        self.assertEqual(tool['input'], original)
        self.assertEqual(tool['arguments'], {'_custom_input': original})
        self.assertEqual(tool['id'], 'call-custom-1')
        self.assertEqual(custom['source_record_id'], 'ctc-synthetic')
        result = next(c for c in calls if c['model'] == 'tool-result')
        original_output = json.loads(FIXTURE.read_text().splitlines()[3])['payload']['output']
        message = result['request']['messages'][0]
        self.assertEqual(result['provider'], 'openai')
        self.assertEqual(message['tool_call_id'], 'call-custom-1')
        self.assertEqual(message['content'], original_output)
        self.assertEqual(message['content_blocks'], original_output)
        self.assertEqual(message['output_format'], 'content_blocks')
        events = events_from_calls(calls)
        for event in events:
            validate_event(event)
        event = next(e for e in events if e['kind'] == 'tool_call')
        self.assertEqual(event['payload']['input'], original)
        self.assertEqual(event['payload']['tool_format'], 'custom')
        self.assertIn('custom_tool_requires_adapter', event['uncertainty'])
        graph = build_execution_graph(events)
        pairs = [e for e in graph['edges'] if e['relation'] == 'tool_result']
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['evidence'], 'recorded')
        fixtures = build_fixtures(graph)
        self.assertEqual(fixtures['issues'], [])
        self.assertEqual(fixtures['fixtures'][0]['result'], original_output)
        candidate = discover_workflows(graph)['candidates'][0]
        workflow = build_workflow(candidate, graph)
        custom_step = next(step for step in workflow['steps'] if step['kind'] == 'tool')
        self.assertEqual(custom_step['tool_format'], 'custom')
        self.assertEqual(custom_step['input'], original)
        self.assertIn('custom_tools_require_runtime_adapter', workflow['uncertainty'])

    def test_custom_json_looking_input_remains_freeform_and_secrets_are_redacted(self):
        text = '{"command":"echo safe"}\n// api_key=sk-abcdefghijklmnop1234\nnext();\n'
        normalized = normalize_tool_call({'type': 'custom', 'name': 'custom', 'input': text, 'call_id': 'c'})
        self.assertIsInstance(normalized['input'], str)
        self.assertEqual(normalized['arguments'], {'_custom_input': normalized['input']})
        self.assertIn('\n// api_key=[REDACTED]\nnext();\n', normalized['input'])
        self.assertNotIn('abcdefghijklmnop', normalized['input'])
        function = normalize_tool_call({'name': 'lookup', 'arguments': '{"account":"A"}'})
        self.assertEqual(function['arguments'], {'account': 'A'})
        self.assertNotIn('type', function)

    def test_inventory_import_preserves_custom_events_and_missing_identity_is_unknown(self):
        inventory = {'inventory_version': 1, 'sources': [{'source_id': 'selected-custom', 'path': str(FIXTURE),
                     'agent': 'codex', 'status': 'ready', 'digest': source_digest(FIXTURE)}]}
        imported = import_inventory(inventory)
        self.assertEqual(imported['diagnostics'][0]['status'], 'imported')
        events = active_events(imported['store'])
        self.assertEqual(len([e for e in events if e['kind'] == 'tool_call']), 1)
        self.assertEqual(len([e for e in events if e['kind'] == 'tool_result']), 1)
        self.assertTrue(any('\nconst result' in e['payload'].get('input', '') for e in events))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unknown.jsonl'
            path.write_text(json.dumps({'type':'response_item','payload':{'type':'custom_tool_call','name':'exec'}})+'\n')
            calls = [make_call(c, 'codex', str(path)) for c in import_codex(path)]
            tool = calls[0]['response']['tool_calls'][0]
            self.assertEqual(tool['input_format'], 'unknown')
            self.assertIsNone(tool['input'])
            self.assertNotIn('id', tool)


if __name__ == '__main__':
    unittest.main()
