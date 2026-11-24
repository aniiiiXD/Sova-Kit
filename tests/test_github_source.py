"""GitHub collection and evidence semantics without credentials or network calls."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from switchit.cli import main
from switchit.event_store import merge_events, empty_event_store
from switchit.execution_graph import build_execution_graph
from switchit.github_source import GitHubClient, GitHubReadError, collect_github
from switchit.importers.github import import_github, repository_name
from switchit.mcp_server import handle_message


def bundle():
    return {'github_export_version':1,'repository':'sample/project','scope':{'pr':4},
      'pull_requests':[{'number':4,'title':'Fix search','body':'Expected result','state':'open','head':{'sha':'abc'},'created_at':'2026-01-01T00:00:00Z'}],
      'comments':[{'id':11,'issue_number':4,'body':'Please test edge cases','created_at':'2026-01-02T00:00:00Z'}],
      'reviews':[{'id':12,'pull_number':4,'state':'APPROVED','body':'Reviewed'}],
      'commits':[{'sha':'abc','pull_number':4,'commit':{'message':'Fix search','committer':{'date':'2026-01-03T00:00:00Z'}},'parents':[]}],
      'runs':[{'id':20,'run_attempt':2,'head_sha':'abc','status':'completed','conclusion':'success','pull_requests':[{'number':4}]}],
      'jobs':[{'id':21,'run_id':20,'run_attempt':2,'name':'test','conclusion':'success','steps':[{'number':1,'name':'Run tests','conclusion':'success'}]}],
      'coverage':[{'path':'fixture','status':'complete'}]}


class GitHubEvidenceTests(unittest.TestCase):
    def test_hierarchy_links_and_no_fabricated_execution(self):
        result=import_github(bundle());events=result['events'];graph=build_execution_graph(events)
        self.assertEqual(len(events),7)
        self.assertEqual({e['evidence'] for e in events},{'metadata_only'})
        self.assertFalse(any(e['kind'] in ('tool_call','model_call','approval') for e in events))
        self.assertEqual(len(graph['edges']),7)
        self.assertEqual(sum(edge.get('causal') is False for edge in graph['edges']),2)
        self.assertFalse(any(e['relation']=='sequence' for e in graph['edges']))
        self.assertEqual({r['relation'] for r in result['relationships']},{'tested_commit','references_pull_request'})
        self.assertEqual(graph['issues'],[])
        self.assertEqual(result['diagnostics'],[])
        from switchit.workflow_patterns import discover_workflows
        from switchit.workflow_builder import build_workflow
        for candidate in discover_workflows(graph)['candidates']:
            with self.assertRaisesRegex(ValueError,'metadata-only'):
                build_workflow(candidate,graph)

    def test_revision_identity_idempotency_and_unresolved_links(self):
        first=import_github(bundle())
        store=merge_events(empty_event_store(),first['events'])
        self.assertEqual(merge_events(store,first['events']),store)
        changed=bundle();changed['pull_requests'][0]['state']='closed'
        second=import_github(changed)
        self.assertNotEqual(first['revision'],second['revision'])
        self.assertEqual(len(merge_events(store,second['events'])['events']),14)
        missing=bundle();missing['commits']=[]
        self.assertTrue(import_github(missing)['diagnostics'])

    def test_redaction_and_duplicate_validation(self):
        data=bundle();data['comments'][0]['body']='token sk-abcdefghijklmnopqrstuvwxyz1234567890'
        imported=json.dumps(import_github(data))
        self.assertNotIn('sk-abcdefghijklmnopqrstuvwxyz1234567890',imported)
        data['comments'].append(copy.deepcopy(data['comments'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            import_github(data)
        for repo in ('https://github.com/a/b','a/../b','../b','a/b?token=x'):
            with self.assertRaises(ValueError):repository_name(repo)

    def test_offline_cli_and_mcp(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'export.json';path.write_text(json.dumps(bundle()))
            output=root/'out'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['sources','github','--file',str(path),'--output',str(output)]),0)
            self.assertTrue((output/'graph.json').exists())
            response=handle_message({'id':1,'method':'tools/call','params':{
                'name':'switchit_sources_github','arguments':{'file':str(path),'output':str(root/'mcp')}}})
            self.assertFalse(response['result']['isError'],response)
            with patch('switchit.github_source.collect_github') as collect:
                with self.assertRaises(SystemExit):
                    main(['sources','github','--repo','a/b','--pr','1','--output',str(output)])
                collect.assert_not_called()

    def test_platform_shows_activity_without_claiming_replay(self):
        from switchit.platform.jobs import JobManager
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder)
            try:
                job=manager.submit({'name':'GitHub evidence','events':import_github(bundle())['events']})
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    result=manager.get(job['id'])
                    if result['status'] not in ('queued','running'):break
                    time.sleep(.01)
                self.assertEqual(result['status'],'evidence_only',result)
                self.assertIn('graph.json',result['artifacts'])
                self.assertNotIn('workflow.json',result['artifacts'])
            finally:manager.close()


class GitHubCollectionTests(unittest.TestCase):
    def test_pagination_does_not_follow_external_links_and_reports_limit(self):
        client=GitHubClient(token='secret',max_pages=1)
        with patch.object(client,'get',return_value=([{'id':1}],{'Link':'<https://evil.test>; rel="next"'})) as get:
            rows,coverage=client.pages('/repos/a/b/issues')
            self.assertEqual(rows,[{'id':1}]);self.assertEqual(coverage['status'],'limited')
            self.assertEqual(get.call_count,1)
        client=GitHubClient(max_pages=2)
        with patch.object(client,'get',side_effect=[([{'id':1}],{'Link':'<x>; rel="next"'}),GitHubReadError('rate limited')]):
            rows,coverage=client.pages('/repos/a/b/issues')
            self.assertEqual(len(rows),1);self.assertEqual(coverage['status'],'partial')

    def test_only_selected_resources_are_requested_and_attempt_is_pinned(self):
        class Fake:
            def __init__(self):self.paths=[]
            def get(self,path,params=None):
                self.paths.append(path)
                if path.endswith('/pulls/4'):return bundle()['pull_requests'][0],{}
                raise AssertionError(path)
            def pages(self,path,**kwargs):
                self.paths.append(path)
                if path.endswith('/actions/runs'):rows=bundle()['runs']
                elif '/attempts/2/jobs' in path:rows=bundle()['jobs']
                else:rows=[]
                return rows,{'path':path,'status':'complete','records':len(rows)}
        client=Fake();data=collect_github('sample/project',pr=4,client=client)
        self.assertIn('/repos/sample/project/actions/runs/20/attempts/2/jobs',client.paths)
        self.assertEqual(data['jobs'][0]['run_attempt'],2)
        self.assertFalse(any('/logs' in path or '/artifacts' in path for path in client.paths))

    def test_request_budget_and_credential_safe_http_error(self):
        client=GitHubClient(token='secret',max_requests=1)
        import urllib.error
        error=urllib.error.HTTPError('https://api.github.com',403,'secret credential detail',{},None)
        with patch.object(client.opener,'open',side_effect=error):
            with self.assertRaises(GitHubReadError) as raised:client.get('/repos/a/b/issues/1')
            self.assertNotIn('secret',str(raised.exception))
        with self.assertRaisesRegex(GitHubReadError,'budget'):client.get('/repos/a/b/issues/1')


if __name__=='__main__':unittest.main()
