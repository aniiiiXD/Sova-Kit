"""Run explicitly approved tiny CPU tests, offline; freeze all packaged bytes."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        KVVERIFY_TORCH_SMOKE='1',TOKENIZERS_PARALLELISM='false')
    from cloud.state_diagnostic_submit import TESTS,tested_hashes
    args.output.mkdir(parents=True,exist_ok=False)
    before=tested_hashes();log=io.StringIO();started=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames([Path(n).stem for n in TESTS])
    with contextlib.redirect_stdout(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    after=tested_hashes()
    report={'mode':'state_diagnostic','status':'passed'if result.wasSuccessful()and before==after and not result.skipped else'failed',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'seconds':time.perf_counter()-started,'file_sha256':after,'sources_unchanged_during_tests':before==after,
        'gpu':False,'network':False,'model_downloads':False,'tiny_random_models_only':True}
    (args.output/'tests.txt').write_text(log.getvalue())
    (args.output/'test-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='file_sha256'},indent=2))
    if report['status']!='passed':print(log.getvalue()[-10000:]);raise SystemExit(1)


if __name__=='__main__':main()
