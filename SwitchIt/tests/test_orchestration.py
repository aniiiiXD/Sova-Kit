"""Offline command graph contracts: no real agents, providers, pushes, or installs."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from switchit.contracts import ContractError
from switchit.orchestration import CoordinatorBusy, cancel, generate_development_plan, run_plan, smoke_plan, status, validate_plan


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'run'

    def task(self, name, script='print("done")', **options):
        cwd = self.base/name
        cwd.mkdir(exist_ok=True)
        return {'id': name, 'command': [sys.executable, '-c', script], 'cwd': str(cwd), **options}

    def plan(self, tasks, **options):
        return {'plan_version': 1, 'tasks': tasks, **options}

    def test_smoke_fanout_join_and_artifact_digests(self):
        result = run_plan(smoke_plan(self.base/'work'), self.root)
        self.assertEqual(result['status'], 'succeeded')
        join = result['tasks']['join']
        artifact = next(a for a in join['artifacts'] if a['kind'] == 'output')
        data = json.loads((self.root/artifact['path']).read_text())
        self.assertEqual(data['dependencies'], ['left', 'right'])
        self.assertEqual(len(artifact['sha256']), 64)
        for name in ('left', 'right'):
            self.assertLessEqual(datetime.fromisoformat(result['tasks'][name]['attempts'][0]['finished_at']),
                                 datetime.fromisoformat(join['attempts'][0]['started_at']))

    def test_parallel_tasks_overlap_and_ceiling_is_enforced(self):
        tasks = [self.task(name, 'import time; time.sleep(.12)') for name in ('a', 'b', 'c')]
        result = run_plan(self.plan(tasks, max_parallel=2), self.root, poll_interval=.01)
        attempts = [result['tasks'][name]['attempts'][0] for name in ('a','b','c')]
        self.assertLess(datetime.fromisoformat(attempts[1]['started_at']), datetime.fromisoformat(attempts[0]['finished_at']))
        self.assertGreaterEqual(datetime.fromisoformat(attempts[2]['started_at']), datetime.fromisoformat(attempts[0]['finished_at']))
        with self.assertRaises(ContractError):
            run_plan(self.plan(tasks, max_parallel=2), self.base/'other', max_parallel=3)

    def test_same_cwd_tasks_are_serialized(self):
        first = self.task('a', 'import time; time.sleep(.05)')
        second = {**first, 'id': 'b'}
        result = run_plan(self.plan([first, second], max_parallel=2), self.root, poll_interval=.01)
        self.assertLessEqual(result['tasks']['a']['attempts'][0]['finished_at'], result['tasks']['b']['attempts'][0]['started_at'])

    def test_failed_dependency_blocks_only_its_descendants(self):
        bad = self.task('bad', 'raise SystemExit(2)')
        dependent = self.task('dependent', 'raise AssertionError("must not run")', depends_on=['bad'])
        good = self.task('good')
        result = run_plan(self.plan([bad, dependent, good]), self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['tasks']['dependent']['status'], 'blocked')
        self.assertEqual(result['tasks']['dependent']['attempts'], [])
        self.assertEqual(result['tasks']['good']['status'], 'succeeded')

    def test_retry_receives_previous_attempt_and_is_bounded(self):
        script = ('import json,sys; from pathlib import Path; r=json.load(sys.stdin); '
                  'Path("result.json").write_text(json.dumps(r["previous_attempt"])); '
                  'sys.exit(1 if r["attempt"]==1 else 0)')
        task = self.task('retry', script, max_attempts=2, outputs=['result.json'])
        result = run_plan(self.plan([task]), self.root)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(len(result['tasks']['retry']['attempts']), 2)
        output = next(a for a in result['tasks']['retry']['artifacts'] if a['kind']=='output')
        self.assertEqual(json.loads((self.root/output['path']).read_text())['status'], 'failed')
        failed = run_plan(self.plan([self.task('always-fail','raise SystemExit(1)',max_attempts=2)]), self.base/'fail')
        self.assertEqual(len(failed['tasks']['always-fail']['attempts']), 2)
        self.assertEqual(failed['status'], 'failed')

    def test_timeout_and_output_limits_fail_tasks(self):
        task = self.task('slow', 'import time; time.sleep(10)', timeout_seconds=.05)
        result = run_plan(self.plan([task]), self.root, poll_interval=.01)
        self.assertEqual(result['tasks']['slow']['attempts'][0]['status'], 'timed_out')
        task = self.task('large', 'print("x"*200)', output_limit_bytes=100)
        result = run_plan(self.plan([task]), self.base/'large-run')
        self.assertEqual(result['status'], 'failed')

    def test_successful_resume_does_not_execute_again(self):
        task = self.task('once', 'from pathlib import Path; p=Path("count"); p.write_text(str(int(p.read_text())+1) if p.exists() else "1")', outputs=['count'])
        plan = self.plan([task])
        first = run_plan(plan, self.root)
        second = run_plan(plan, self.root)
        self.assertEqual(first, second)
        self.assertEqual((Path(task['cwd'])/'count').read_text(), '1')

    def test_changed_plan_or_accepted_artifact_refuses_resume(self):
        plan = self.plan([self.task('a')])
        result = run_plan(plan, self.root)
        changed = deepcopy(plan)
        changed['tasks'][0]['command'][-1] = 'print("different")'
        with self.assertRaises(ContractError):
            run_plan(changed, self.root)
        artifact = result['tasks']['a']['artifacts'][0]
        (self.root/artifact['path']).write_text('tampered')
        with self.assertRaises(ContractError):
            run_plan(plan, self.root)

    def test_cancellation_locking_and_resume(self):
        script = ('from pathlib import Path; import time; p=Path("started"); '
                  'first=not p.exists(); p.write_text("yes"); time.sleep(5 if first else 0)')
        plan = self.plan([self.task('wait', script)])
        outcomes = []
        thread = threading.Thread(target=lambda: outcomes.append(run_plan(plan, self.root, poll_interval=.01)))
        thread.start()
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            snapshot = status(self.root)
            if snapshot.get('tasks', {}).get('wait', {}).get('attempts') and (self.base/'wait'/'started').exists():
                break
            time.sleep(.01)
        try:
            with self.assertRaises(CoordinatorBusy):
                run_plan(plan, self.root)
            self.assertTrue(cancel(self.root)['cancel_requested'])
        finally:
            if thread.is_alive():
                cancel(self.root)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes[0]['status'], 'cancelled')
        resumed = run_plan(plan, self.root)
        self.assertEqual(resumed['status'], 'succeeded')
        self.assertEqual(len(resumed['tasks']['wait']['attempts']), 2)

    def test_interruption_persists_resumable_state(self):
        plan = self.plan([self.task('a', 'import time; time.sleep(.1)')])
        original_sleep = time.sleep
        calls = []
        def interrupt_once(seconds):
            if not calls:
                calls.append(True)
                raise KeyboardInterrupt
            original_sleep(seconds)
        with patch('switchit.orchestration.scheduler.time.sleep', side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                run_plan(plan, self.root)
        self.assertEqual(status(self.root)['status'], 'interrupted')
        self.assertEqual(run_plan(plan, self.root)['status'], 'succeeded')

    def test_invalid_contracts_are_rejected_before_commands(self):
        base = self.task('a')
        for changes in ({'command':'echo nope'}, {'cwd':'.'}, {'timeout_seconds':0}, {'outputs':['../escape']}, {'depends_on':['missing']}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                validate_plan(self.plan([{**base, **changes}]))
        with self.assertRaises(ContractError):
            validate_plan(self.plan([{**base, 'depends_on':['a']}]))

    def test_declared_output_missing_is_not_success(self):
        result = run_plan(self.plan([self.task('a',outputs=['missing.json'])]),self.root)
        self.assertEqual(result['status'],'failed')

    def test_generator_has_real_configured_wrappers_and_no_implicit_publish(self):
        roles = ['investigate_downloads','investigate_claude','investigate_codex','code_source','code_workflow','code_target',
                 'test_source','test_workflow','test_target','verify_integration','package_skill','package_harness','package_platform','verify_release']
        workers = {role:{'command':[sys.executable,'-c','print("configured")'],'cwd':str(self.base)} for role in roles}
        sources = {name:str(self.base/name) for name in ('downloads','claude','codex')}
        plan = generate_development_plan(workers,sources=sources)
        self.assertEqual(len(plan['tasks']),14)
        self.assertNotIn('publish',{task['id'] for task in plan['tasks']})
        with self.assertRaisesRegex(ContractError,'publisher|publish'):
            generate_development_plan(workers,sources=sources,include_publish=True)


class RepairLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'run'
        self.criteria = self.base/'criteria.txt'
        self.criteria.write_text('result must equal good')

    def config(self, coder, verifier, **options):
        return {'repair_version': 1, 'criteria_paths': [str(self.criteria)],
                'coder': {'command': [sys.executable, '-c', coder], 'cwd': str(self.base)},
                'verifier': {'command': [sys.executable, '-c', verifier], 'cwd': str(self.base)},
                **options}

    def run_loop(self, config):
        from switchit.orchestration import run_repair_loop
        return run_repair_loop(config, self.root, poll_interval=.01)

    def test_failed_verification_is_fed_to_next_coder_and_fixed_checks_pass(self):
        coder = '''import json,sys
from pathlib import Path
request=json.load(sys.stdin)['input']
previous=request['previous_verification']
if previous:
    assert previous['phase']=='verifier'
    assert previous['task']['attempts'][-1]['exit_code']==1
Path('result.txt').write_text('good' if previous else 'bad')
'''
        verifier = "from pathlib import Path; import sys; value=Path('result.txt').read_text(); print('expected good, got',value); sys.exit(0 if value=='good' else 1)"
        config = self.config(coder, verifier)
        state = self.run_loop(config)
        self.assertEqual(state['status'], 'succeeded')
        self.assertEqual(len(state['rounds']), 2)
        self.assertEqual(state['rounds'][0]['verifier']['status'], 'failed')
        self.assertEqual(state['criteria_digests'][str(self.criteria.resolve())], __import__('hashlib').sha256(self.criteria.read_bytes()).hexdigest())
        self.assertEqual(self.run_loop(config), state)

    def test_coder_cannot_change_acceptance_criteria(self):
        config = self.config("from pathlib import Path; Path('criteria.txt').write_text('anything passes')",
                             "from pathlib import Path; Path('verifier-ran').touch()")
        state = self.run_loop(config)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('acceptance criteria changed', state['error'])
        self.assertFalse((self.base/'verifier-ran').exists())
        self.assertEqual(len(state['rounds']), 1)

    def test_repair_stops_at_round_budget(self):
        state = self.run_loop(self.config('print("attempt")', 'raise SystemExit(1)', max_rounds=2))
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(len(state['rounds']), 2)
        self.assertIn('round budget', state['error'])

    def test_verifier_script_is_pinned_even_when_not_listed_as_criteria(self):
        verifier = self.base/'verifier.py'
        verifier.write_text('raise SystemExit(1)')
        config = self.config("from pathlib import Path; Path('verifier.py').write_text('print(1)')", 'unused')
        config['verifier']['command'] = [sys.executable, str(verifier)]
        state = self.run_loop(config)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('verifier.py', state['error'])

    def test_repair_resume_rejects_different_config_or_modified_evidence(self):
        config = self.config('print("code")', 'print("verified")')
        state = self.run_loop(config)
        with self.assertRaisesRegex(ContractError, 'immutable'):
            self.run_loop({**config, 'max_rounds': 7})
        artifact = self.root/state['receipts'][0]['path']
        artifact.write_text('changed evidence')
        with self.assertRaisesRegex(ContractError, 'repair artifact changed'):
            self.run_loop(config)

    def test_repair_cancel_then_resume_preserves_round(self):
        coder = "from pathlib import Path; import time; p=Path('started'); old=p.exists(); p.touch(); time.sleep(0 if old else 5)"
        config = self.config(coder, 'print("verified")')
        results = []
        thread = threading.Thread(target=lambda: results.append(self.run_loop(config)))
        thread.start()
        deadline = time.monotonic()+3
        while not (self.base/'started').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        try:
            cancel(self.root)
        finally:
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0]['status'], 'cancelled')
        resumed = self.run_loop(config)
        self.assertEqual(resumed['status'], 'succeeded')
        self.assertEqual(len(resumed['rounds']), 1)
        self.assertEqual(len(resumed['rounds'][0]['coder']['task']['attempts']), 2)

    def test_generated_graph_executes_real_repair_lane_before_integration(self):
        roles = ['investigate_downloads','investigate_claude','investigate_codex','code_source','code_workflow','code_target',
                 'test_source','test_workflow','test_target','verify_integration','package_skill','package_harness','package_platform','verify_release']
        simple = "from pathlib import Path; Path('result.json').write_text('{}')"
        workers = {role:{'command':[sys.executable,'-c',simple],'cwd':str(self.base)} for role in roles}
        workers['code_source']['command'] = [sys.executable, '-c', '''import json,sys
from pathlib import Path
request=json.load(sys.stdin)['input']
assert len(request['request']['dependencies']) == 3
Path('value.txt').write_text('good' if request['previous_verification'] else 'bad')
Path('result.json').write_text('{}')
''']
        workers['test_source']['command'] = [sys.executable, '-c', "from pathlib import Path; import sys; Path('result.json').write_text('{}'); sys.exit(0 if Path('value.txt').read_text()=='good' else 1)"]
        sources = {name:str(self.base/name) for name in ('downloads','claude','codex')}
        plan = generate_development_plan(workers, sources=sources,
                 repairs={'source': {'criteria_paths':[str(self.criteria)], 'max_rounds':2}})
        state = run_plan(plan, self.root, poll_interval=.01)
        self.assertEqual(state['status'], 'succeeded')
        repair = status(self.root/'repairs'/'code-source-adapters')
        self.assertEqual(repair['status'], 'succeeded')
        self.assertEqual(len(repair['rounds']), 2)
        self.assertEqual(state['tasks']['verify-integration']['status'], 'succeeded')
