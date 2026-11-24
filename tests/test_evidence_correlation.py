"""Exact references must never become authorship or executable dependencies."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from switchit.cli import main
from switchit.evidence_correlation import combine_evidence, correlate_github
from switchit.events import normalize_event
from switchit.execution_graph import build_execution_graph
from switchit.importers.github import import_github
from switchit.mcp_server import handle_message
from tests.test_github_source import bundle
from tests.workflow_support import event, evidence

SHA='a'*40


def github():
    data=bundle();data['commits'][0]['sha']=SHA;data['runs'][0]['head_sha']=SHA
    return import_github(data)['events']


def agent(text, **payload):
    return event('agent-ref',trace='agent',payload={'response':{'text':text},**payload})


class CorrelationTests(unittest.TestCase):
    def test_explicit_url_and_scoped_commit_reference(self):
        record=agent('See https://github.com/sample/project/pull/4/files#diff-x.',
                     github_reference={'repository':'sample/project','commit_sha':SHA})
        report=correlate_github([record,*github()])
        self.assertEqual({link['resource']['kind'] for link in report['links']},{'pull','commit'})
        self.assertTrue(all(not l['causal'] and not l['proves_authorship'] for l in report['links']))
        self.assertEqual(report['unresolved'],[])
        graph=build_execution_graph([record,*github()])
        self.assertEqual(len([e for e in graph['edges'] if e['relation']=='references_github']),2)
        self.assertTrue(all(e['causal'] is False for e in graph['edges'] if e['relation']=='references_github'))

    def test_urls_in_tool_results_are_observations_not_authorship(self):
        record=event('tool-output','tool_result',payload={'content':'Created https://github.com/sample/project/pull/4'})
        before=deepcopy(record)
        result=correlate_github([record,*github()])
        self.assertEqual(record,before)
        self.assertEqual(result['links'][0]['observations'][0]['payload_path'],['content'])
        self.assertFalse(result['links'][0]['proves_authorship'])

    def test_never_guess_bare_short_cross_repo_or_spoofed_references(self):
        text='PR #4 main aaaaaaa '+SHA+' https://github.com/other/project/pull/4 '+\
             'https://github.com.evil.test/sample/project/pull/4 https://github.com@evil.test/sample/project/pull/4 '+\
             'https://github.com/sample/project/commit/aaaaaaa'
        result=correlate_github([agent(text),*github()])
        self.assertEqual(result['links'],[])
        self.assertEqual(len(result['unresolved']),1)
        self.assertEqual(result['unresolved'][0]['resource']['repository'],'other/project')

    def test_multiple_imported_snapshots_remain_ambiguous(self):
        data=bundle();data['pull_requests'][0]['state']='closed'
        report=correlate_github([agent('https://github.com/sample/project/pull/4'),*github(),*import_github(data)['events']])
        self.assertEqual(report['links'],[])
        self.assertEqual(report['unresolved'][0]['reason'],'ambiguous_snapshot')
        self.assertEqual(len(report['unresolved'][0]['candidate_event_ids']),2)

    def test_multiple_run_attempts_are_ambiguous(self):
        data=bundle();data['runs'][0]['run_attempt']=3;data['jobs'][0]['run_attempt']=3
        result=correlate_github([agent('https://github.com/sample/project/actions/runs/20'),*github(),*import_github(data)['events']])
        self.assertEqual(result['links'],[])
        self.assertEqual(result['unresolved'][0]['reason'],'ambiguous_snapshot')

    def test_combination_dedupes_exact_events_and_refuses_conflicts(self):
        record=agent('hello')
        self.assertEqual(combine_evidence([[record],{'events':[record]}]),[record])
        changed=deepcopy(record);changed['payload']['response']['text']='different'
        with self.assertRaisesRegex(ValueError,'conflicting'):
            combine_evidence([[record],[changed]])

    def test_active_store_excludes_historical_snapshot(self):
        from switchit.event_store import empty_event_store,merge_events
        first=github();second=deepcopy(first)
        for e in first:e['source']['source_id']='snapshot'
        for e in second:
            e['event_id']+='-changed';e['source']['source_id']='snapshot'
        store=merge_events(empty_event_store(),first+second,{'snapshot':{'event_ids':[e['event_id'] for e in second]}})
        combined=combine_evidence([store,[agent('https://github.com/sample/project/pull/4')]])
        self.assertEqual(len(combined),len(second)+1)
        self.assertEqual(len(correlate_github(combined)['links']),1)

    def test_metadata_does_not_become_workflow_or_holdout_case(self):
        from switchit.workflow_builder import build_workflow,review_workflow
        from switchit.workflow_cases import build_workflow_cases
        from switchit.workflow_patterns import discover_workflows
        local=evidence()
        # Sorting puts this GitHub task before the actual agent traces.
        data=bundle();data['pull_requests'][0]['title']='Look up account status'
        remote=import_github(data)['events']
        graph=build_execution_graph(remote+local)
        candidate=next(c for c in discover_workflows(graph)['candidates'] if c['task']=='Look up account status')
        spec=review_workflow(build_workflow(candidate,graph),reviewer='fixture',success_checks=[{'type':'output_contains','text':'active'}])
        self.assertTrue(all(s['kind'] in ('model','tool') for s in spec['steps']))
        suite=build_workflow_cases(spec,candidate,graph)
        self.assertEqual(len(suite['cases']),2)
        self.assertTrue(suite['excluded_metadata_segments'])
        self.assertTrue(all(not case['trace_id'].startswith('github:') for case in suite['cases']))

    def test_public_cli_and_mcp_reconstruct_same_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'agent.json';g=root/'github.json'
            a.write_text(json.dumps({'events':[agent('https://github.com/sample/project/pull/4')]}))
            g.write_text(json.dumps({'events':github()}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['sources','correlate','--agent-events',str(a),'--github-events',str(g),'--output',str(root/'cli')]),0)
            result=handle_message({'id':1,'method':'tools/call','params':{'name':'switchit_sources_correlate','arguments':{
                'agent_events':str(a),'github_events':str(g),'output':str(root/'mcp')}}})
            self.assertFalse(result['result']['isError'],result)
            report=json.loads((root/'cli/correlation.json').read_text())
            self.assertEqual(len(report['links']),1)
            self.assertEqual((root/'cli/graph.json').read_bytes(),(root/'mcp/graph.json').read_bytes())
            events=json.loads((root/'cli/events.json').read_text())['events']
            self.assertEqual(build_execution_graph(events)['correlation'],report)


if __name__=='__main__':unittest.main()
