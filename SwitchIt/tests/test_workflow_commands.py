"""Public integration tests: evidence to reviewed fixture run and portable skill."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from switchit.cli import main
from switchit.mcp_server import handle_message
from switchit.skill_export import export_skill
from switchit.workflow_commands import execute_case
from switchit.recipes import make_recipe
from tests.workflow_support import evidence, pipeline


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value))
        return str(path)

    def cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(list(args))

    def test_full_public_path_with_frozen_checks_and_skill_verification(self):
        events = self.write('events.json', evidence())
        bundle = self.root/'bundle'
        self.assertEqual(self.cli('workflow','build','--events',events,'--output',str(bundle)), 0)
        checks = self.write('checks.json',[{'type':'output_contains','text':'active'}])
        specpath = str(self.root/'workflow.json')
        self.assertEqual(self.cli('workflow','review','--workflow',str(bundle/'drafts.json'),
                                 '--checks',checks,'--reviewer','test-owner','--output',specpath),0)
        suitepath = str(self.root/'cases.json')
        self.assertEqual(self.cli('workflow','cases','--workflow',specpath,'--graph',str(bundle/'graph.json'),
                                 '--candidates',str(bundle/'candidates.json'),'--output',suitepath),0)
        suite = json.loads(Path(suitepath).read_text())
        resultpath = str(self.root/'result.json')
        self.assertEqual(self.cli('workflow','run','--workflow',specpath,'--case',suitepath,
                                 '--case-id',suite['cases'][0]['case_id'],'--output',resultpath),0)
        self.assertEqual(json.loads(Path(resultpath).read_text())['evaluation']['verdict'],'pass')
        export = self.root/'account-skill'
        self.assertEqual(self.cli('workflow','export-skill','--workflow',specpath,'--cases',suitepath,
                                 '--output',str(export)),0)
        env = {**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        process = subprocess.run([sys.executable,'scripts/verify.py'],cwd=export,env=env,capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        contract = export/'references/workflow.json'
        contract.write_text(contract.read_text()+' ')
        process = subprocess.run([sys.executable,'scripts/verify.py'],cwd=export,env=env,capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('integrity',process.stderr)

    def test_modified_case_checks_and_missing_review_rejected(self):
        _,_,_,spec,suite = pipeline()
        altered = copy.deepcopy(suite['cases'][0]); altered['checks'] = [{'type':'no_error'}]
        with self.assertRaisesRegex(ValueError,'criteria'):
            execute_case(spec,altered)
        draft = copy.deepcopy(spec);draft['status']='draft'
        with self.assertRaises(ValueError):
            export_skill(draft,self.root/'bad')
        with self.assertRaises(ValueError):
            export_skill(spec,self.root/'bad',name='../escape')

    def test_failure_keeps_evidence_and_returns_nonzero(self):
        _,_,_,spec,suite = pipeline()
        case = copy.deepcopy(suite['cases'][0])
        for output in case['step_outputs'].values():
            output['text']='unverified'
        output = self.root/'failed.json'
        code = self.cli('workflow','run','--workflow',self.write('spec.json',spec),
                        '--case',self.write('case.json',case),'--output',str(output))
        self.assertEqual(code,1)
        self.assertEqual(json.loads(output.read_text())['evaluation']['verdict'],'fail')

    def test_mcp_workflow_and_profile_dispatch(self):
        result = handle_message({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
            'name':'switchit_workflow_build','arguments':{'events':self.write('events.json',evidence()),'output':str(self.root/'bundle')}}})
        self.assertFalse(result['result']['isError'],result)
        bad = handle_message({'id':2,'method':'tools/call','params':{
            'name':'switchit_workflow_run','arguments':{'workflow':'x','case':'x','output':'x','remote':'false'}}})
        self.assertTrue(bad['result']['isError'])
        self.assertIn('invalid type',bad['result']['content'][0]['text'])
        from switchit.mcp_server import _run_cli_tool
        from unittest.mock import patch
        with patch('switchit.cli.cmd_profile',return_value=0) as profile:
            _run_cli_tool('switchit_profile',{})
            profile.assert_called_once()

    def test_control_mcp_preserves_parent_command_selectors(self):
        def call(name, arguments):
            return handle_message({'id':3,'method':'tools/call','params':{'name':name,'arguments':arguments}})['result']
        result = call('switchit_harness_status',{'store':str(self.root/'promotion.json')})
        self.assertFalse(result['isError'],result)
        self.assertEqual(json.loads(result['content'][0]['text'])['generation'],0)
        result = call('switchit_orchestration_status',{'output':str(self.root/'graph-run')})
        self.assertFalse(result['isError'],result)
        self.assertEqual(json.loads(result['content'][0]['text'])['status'],'not_started')
        from switchit.workflow_mcp import definitions
        cases = next(d for d in definitions() if d['name']=='switchit_workflow_cases')
        self.assertIn('holdout_fraction',cases['inputSchema']['properties'])

    def test_recipe_comparison_and_local_promotion(self):
        _,_,_,spec,suite = pipeline()
        recipe = make_recipe('recorded','fixtures','1',spec['revision'],spec['allowed_tools'])
        experiment = self.root/'experiment.json'
        self.assertEqual(self.cli('harness','compare','--workflow',self.write('spec.json',spec),
            '--cases',self.write('cases.json',suite),'--recipes',self.write('recipes.json',[recipe]),
            '--output',str(experiment)),0)
        pointer = self.root/'promotion.json'
        self.assertEqual(self.cli('harness','promote','--store',str(pointer),'--experiment',str(experiment),
                                 '--expected-generation','0'),0)
        self.assertEqual(json.loads(pointer.read_text())['generation'],1)

    def test_explicit_source_import_to_active_workflow_bundle(self):
        source = self.root/'sources'
        source.mkdir()
        log = source/'session.jsonl'
        log.write_text(json.dumps({'type':'user','sessionId':'local-test','message':{'content':'Check status'}})+'\n')
        inventory = str(self.root/'inventory.json')
        self.assertEqual(self.cli('sources','inventory','--roots',self.write('roots.json',{'claude-code':[str(source)]}),
                                 '--output',inventory),0)
        imported = self.root/'imported'
        self.assertEqual(self.cli('sources','import','--inventory',inventory,'--output',str(imported)),0)
        self.assertEqual(len(json.loads((imported/'events.json').read_text())['events']),1)
        bundle = self.root/'bundle'
        self.assertEqual(self.cli('workflow','build','--events',str(imported/'store.json'),'--output',str(bundle)),0)
        self.assertEqual(len(json.loads((bundle/'events.json').read_text())['events']),1)

    def test_exact_outcome_check_rejects_inactive(self):
        from switchit.evaluation import evaluate_check
        check = {'type':'output_equals','text':'The account is active'}
        self.assertEqual(evaluate_check(check,{'text':'The account is inactive'})['status'],'fail')
        self.assertEqual(evaluate_check(check,{'text':'The account is active'})['status'],'pass')

    def test_command_execution_requires_opt_in(self):
        _,_,_,spec,suite = pipeline()
        with self.assertRaisesRegex(ValueError,'opt-in'):
            execute_case(spec,suite['cases'][0],runtime='command',command={
                'runtime_id':'test','argv':[sys.executable,'-c','print("{}")'],'version':'1'})


if __name__ == '__main__':
    unittest.main()
