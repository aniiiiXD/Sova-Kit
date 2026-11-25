"""Live metadata reads must never launch jobs or download model artifacts."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cloud import full_collect as collect


class LiveStatusTests(unittest.TestCase):
    def run_status(self, progress=False, missing=False):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'submission.json'
            receipt.write_text(json.dumps({'job_name':'sova-kv-full-test',
                'region':collect.REGION,'profile':collect.PROFILE,
                'request':{'RoleArn':f'arn:aws:iam::{collect.ACCOUNT}:role/sova-sagemaker-exec'},
                'live_results':f's3://{collect.BUCKET}/kv-transfer/sova-kv-full-test/live/',
                'checkpoints':f's3://{collect.BUCKET}/kv-transfer/sova-kv-full-test/checkpoints/',
                'price':{'usd_per_hour':1.515}}))
            calls=[]
            def fake(*args):
                calls.append(args)
                if args[:2] == ('sagemaker','describe-training-job'):
                    return json.dumps({'TrainingJobStatus':'InProgress','SecondaryStatus':'Training'})
                if args[:2] == ('s3','cp') and 'progress.json' in args[2] and args[3]=='-':
                    if missing:
                        raise subprocess.CalledProcessError(1,args,stderr='NoSuchKey')
                    return json.dumps({'phase':'training','step':50})
                raise AssertionError('Unexpected mutation or bulk transfer: '+repr(args))
            argv=['collect','--receipt',str(receipt),'--status-only']
            if progress:argv.append('--progress')
            output=io.StringIO()
            with patch.object(sys,'argv',argv),patch.object(collect,'aws',fake),contextlib.redirect_stdout(output):
                collect.main()
            return json.loads(output.getvalue()),calls

    def test_status_only_has_no_artifact_transfer(self):
        result,calls=self.run_status()
        self.assertEqual(result['status'],'InProgress')
        self.assertEqual(len(calls),1)

    def test_progress_reads_only_small_json(self):
        result,calls=self.run_status(progress=True)
        self.assertEqual(result['live_progress']['step'],50)
        self.assertEqual(len(calls),2)

    def test_missing_progress_does_not_hide_job_status(self):
        result,calls=self.run_status(progress=True,missing=True)
        self.assertEqual(result['secondary_status'],'Training')
        self.assertFalse(result['live_progress_available'])


if __name__=='__main__':
    unittest.main()
