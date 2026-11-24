import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

from switchit.contracts import ContractError
from switchit.events import events_from_calls, normalize_event, validate_event
from switchit.event_store import empty_event_store, ingest_events, merge_events
from switchit.execution_graph import build_execution_graph
from switchit.normalization import make_call
from tests.workflow_support import event, evidence


class EventTests(unittest.TestCase):
    def test_normalization_is_idempotent_and_preserves_relationships(self):
        record = event('child', parent_id='parent', tool_call_id='tool-7', payload={'text': 'email me user@example.com'})
        self.assertEqual(normalize_event(record), record)
        self.assertEqual(validate_event(record), record)
        self.assertNotIn('user@example.com', json.dumps(record))
        self.assertEqual(record['parent_id'], 'parent')

    def test_invalid_events_are_rejected(self):
        for updates in ({'kind': 'invented'}, {'trace_id': ''}, {'index': -1}, {'source': {}}):
            with self.subTest(updates=updates), self.assertRaises(ContractError):
                normalize_event({**event('e'), **updates})

    def test_missing_and_truncated_evidence_is_explicit(self):
        record = event('large', payload={'text': 'x'*10000})
        self.assertIn('timestamp_missing', record['uncertainty'])
        self.assertIn('payload_truncated', record['uncertainty'])

    def test_incremental_ingest_is_idempotent(self):
        first = merge_events(empty_event_store(), [event('one')])
        second = merge_events(first, [event('one')])
        self.assertEqual(first, second)
        self.assertEqual(merge_events(second, [event('two')])['revision'], 2)

    def test_conflicting_batch_is_atomic(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'store.json'
            ingest_events(path, [event('one')])
            previous = path.read_bytes()
            with self.assertRaises(ContractError):
                ingest_events(path, [event('two'), event('one', payload={'different': True})])
            self.assertEqual(path.read_bytes(), previous)

    def test_canonical_bridge_preserves_tool_ids_and_history_deduplication(self):
        raw = {'trace_id': 't', 'call_id': 'c', 'parentSpanId': 'root', 'timestamp': '2026-01-01',
               'request': {'messages': [{'role': 'tool', 'tool_call_id': 'tool-1', 'content': 'done'}]},
               'response': {'tool_calls': [{'id': 'tool-1', 'name': 'lookup', 'arguments': {}}]}}
        call = make_call(raw, 'fixture', 'fixture.jsonl')
        later = make_call({**raw, 'call_id': 'c2', 'response': {}}, 'fixture', 'fixture.jsonl')
        records = events_from_calls([call, later])
        self.assertEqual(sum(e['kind'] == 'tool_result' for e in records), 1)
        tool = next(e for e in records if e['kind'] == 'tool_call')
        self.assertEqual(tool['tool_call_id'], 'tool-1')
        self.assertEqual(tool['parent_span_id'], 'root')


class ExecutionGraphTests(unittest.TestCase):
    def test_tool_results_link_by_identity_not_name(self):
        graph = build_execution_graph(evidence())
        links = [e for e in graph['edges'] if e['relation'] == 'tool_result']
        self.assertEqual(len(links), 2)
        self.assertEqual(graph['issues'], [])
        self.assertEqual(links[0]['source'], 'trace-a-call')

    def test_missing_and_ambiguous_references_are_not_invented(self):
        graph = build_execution_graph([event('a', 'tool_call', tool_call_id='same'),
                                       event('b', 'tool_call', tool_call_id='same'),
                                       event('c', 'tool_result', tool_call_id='same'),
                                       event('d', parent_id='missing')])
        self.assertEqual({i['reason'] for i in graph['issues']}, {'missing_reference', 'ambiguous_reference'})
        self.assertFalse(any(e['relation'] == 'tool_result' for e in graph['edges']))

    def test_cycles_are_reported_without_corrupting_graph(self):
        graph = build_execution_graph([event('a', parent_id='b'), event('b', parent_id='a')])
        self.assertTrue(any(i['reason'] == 'cycle' for i in graph['issues']))
        self.assertEqual(set(graph['order']), {'a', 'b'})

    def test_subagent_retry_and_handoff_edges(self):
        events = [event('root', 'agent_start', agent_id='main'),
                  event('child', 'agent_start', index=1, agent_id='child', parent_agent_id='main'),
                  event('attempt', index=2, parent_id='child'),
                  event('retry', index=3, retry_of='attempt'),
                  event('next', 'handoff', index=4, handoff_from='retry')]
        graph = build_execution_graph(events)
        self.assertTrue({'subagent', 'parent', 'retry', 'handoff'} <= {e['relation'] for e in graph['edges']})

    def test_duplicate_event_ids_are_rejected(self):
        with self.assertRaises(ContractError):
            build_execution_graph([event('same'), event('same')])


class MetadataBoundaryTests(unittest.TestCase):
    def test_devin_metadata_remains_non_executable_evidence(self):
        from switchit.devin import import_devin
        calls = [make_call(raw, 'devin', 'synthetic') for raw in import_devin([{'session_id': 's', 'title': 'A task', 'status': 'finished'}])]
        events = events_from_calls(calls)
        self.assertTrue(events)
        self.assertTrue(all(e['evidence'] == 'metadata_only' for e in events))

    def test_new_event_schema_versions_are_not_silently_downgraded(self):
        with self.assertRaises(ContractError):
            normalize_event({**event('x'), 'event_version': 99})
