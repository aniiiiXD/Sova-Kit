from copy import deepcopy
import unittest
from switchit import evaluation, providers, replay


class ScriptedAdapter:
    def __init__(self, responses):
        self.responses = deepcopy(responses)
        self.requests = []
    def replay(self, request, case=None):
        self.requests.append(deepcopy(request))
        return self.responses.pop(0)


def response(text='', calls=None, tokens=1, latency=1):
    return {'text': text, 'tool_calls': calls or [], 'error': None, 'finish_reason': 'stop',
            'usage': {'input_tokens': tokens, 'output_tokens': tokens, 'basis': 'recorded'}, 'latency_ms': latency}


class ReplayCorrectnessTests(unittest.TestCase):
    def test_repeated_tool_calls_are_not_collapsed(self):
        expected = [{'name': 'lookup', 'arguments': {'id': n}} for n in (1, 2)]
        findings = evaluation.compare_tools(expected, expected[1:])
        self.assertTrue(any(f['kind'] == 'missing_tool' for f in findings))
        self.assertTrue(any(f['kind'] == 'arg_drift' for f in findings))

    def test_chain_ids_and_all_turn_usage_are_preserved(self):
        tool = {'name': 'lookup', 'arguments': {'id': 1}}
        target = ScriptedAdapter([response(calls=[tool], tokens=100, latency=20), response('done', tokens=5, latency=3)])
        result, metadata = replay._chain_target_turns({'request': {}, 'tool_fixtures': {'lookup': 'value'}}, target)
        converted = providers.to_openai_messages(target.requests[1]['messages'])
        self.assertEqual(converted[0]['tool_calls'][0]['id'], converted[1]['tool_call_id'])
        self.assertEqual(result['usage']['input_tokens'], 105)
        self.assertEqual(result['latency_ms'], 23)
        self.assertTrue(metadata['completed'])

    def test_exact_fixture_mismatch_stops_before_next_model_call(self):
        target = ScriptedAdapter([response(calls=[{'name': 'lookup', 'arguments': {'id': 2}}])])
        fixture = {'fixture_id': 'f', 'trace_id': 't', 'name': 'lookup', 'arguments': {'id': 1}, 'occurrence': 0, 'result': 'value'}
        result, metadata = replay._chain_target_turns({'trace_id': 't', 'request': {},
                'fixture_set': {'fixtures_version': 1, 'fixtures': [fixture]}}, target)
        self.assertIn('fixture missing', result['error'])
        self.assertEqual(len(target.requests), 1)

    def test_openai_ids_are_unique_and_explicit_ids_are_retained(self):
        messages = [{'role': 'assistant', 'tool_calls': [{'name': 'a'}, {'name': 'b'}]},
                    {'role': 'tool', 'content': 'one'}, {'role': 'tool', 'content': 'two'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'actual', 'name': 'c'}]},
                    {'role': 'tool', 'tool_call_id': 'actual', 'content': 'three'}]
        converted = providers.to_openai_messages(messages)
        ids = [c['id'] for m in converted for c in m.get('tool_calls', [])]
        self.assertEqual(ids, ['call_0', 'call_1', 'actual'])
        self.assertEqual(converted[-1]['tool_call_id'], 'actual')

    def test_anthropic_and_bedrock_preserve_tool_result_blocks(self):
        messages = [{'role': 'assistant', 'tool_calls': [{'id': 'tool-1', 'name': 'lookup', 'arguments': {}}]},
                    {'role': 'tool', 'tool_call_id': 'tool-1', 'content': 'value'}]
        _, anthropic = providers.to_anthropic_messages(messages)
        self.assertEqual(anthropic[-1]['content'][0]['tool_use_id'], 'tool-1')
        _, bedrock = providers.to_bedrock_messages(messages)
        self.assertTrue(all(isinstance(m['content'], list) for m in bedrock))
        self.assertEqual(bedrock[-1]['content'][0]['toolResult']['toolUseId'], 'tool-1')

    def test_repeated_check_types_compare_the_matching_reference(self):
        case = {'case_id': 'c', 'historical': response('alpha'), 'checks': [
            {'type': 'output_contains', 'text': 'alpha'}, {'type': 'output_contains', 'text': 'beta'}]}
        result = evaluation.finalize_case(case, response('alpha'), response('beta'))
        self.assertFalse(any('also fails against the historical reference' in note for note in result['uncertainty']))
