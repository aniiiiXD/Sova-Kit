"""Run only the declared tiny offline recovery tests and freeze their source."""
import argparse
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('probe','repair','precision','eval_resume'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',KVVERIFY_TORCH_SMOKE='1',TOKENIZERS_PARALLELISM='false')
    from cloud.recovery_submit import MODE_TESTS,tested_hashes
    args.output.mkdir(parents=True,exist_ok=False)
    before=tested_hashes(args.mode)
    names=[Path(n).stem for n in MODE_TESTS[args.mode]]
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    log=io.StringIO();started=time.perf_counter()
    with contextlib.redirect_stdout(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    after=tested_hashes(args.mode)
    report={'status':'passed'if result.wasSuccessful()and before==after else'failed','mode':args.mode,
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'seconds':time.perf_counter()-started,
        'gpu':False,'network':False,'model_downloads':False,'tiny_random_models_only':True,
        'file_sha256':after,'sources_unchanged_during_tests':before==after}
    (args.output/'tests.txt').write_text(log.getvalue())
    (args.output/'test-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='file_sha256'},indent=2))
    if report['status']!='passed':print(log.getvalue()[-8000:]);raise SystemExit(1)


if __name__=='__main__':main()
