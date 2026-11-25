"""No timing allocation or confirmation on rejected development evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.follow_answer_mapper import follow


class FollowThroughTests(unittest.TestCase):
    def test_rejected_candidate_never_submits_timing_even_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            training = run/'training'
            training.mkdir()
            (training/'submission.json').write_text(json.dumps({
                'status':'submitted','mode':'answer_train','job_name':'training-test',
                'bundle_sha256':'test-bundle'
            }))
            with (patch('scripts.follow_answer_mapper.terminal', return_value={
                      'TrainingJobStatus':'Completed','BillableTimeInSeconds':1
                  }) as terminal,
                  patch('scripts.follow_answer_mapper.verify_run',
                        return_value={'verdict':'development_rejected'}) as audit,
                  patch('scripts.follow_answer_mapper.subprocess.run') as launch):
                follow(run)
                follow(run)
            launch.assert_not_called()
            terminal.assert_called_once_with('training-test')
            audit.assert_called_once_with(run.resolve())
            result = json.loads((run/'follow-through/result.json').read_text())
            self.assertEqual(result['verdict'], 'development_rejected')
            self.assertFalse(result['timing_submitted'] or result['confirmation_opened'])
