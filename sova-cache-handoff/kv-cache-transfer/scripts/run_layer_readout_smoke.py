"""Offline source-frozen controls for a future acceptance-only job."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',KVVERIFY_TORCH_SMOKE='1',TOKENIZERS_PARALLELISM='false')
    from cloud.layer_readout_submit import TESTS,tested_hashes
    args.output.mkdir(parents=True,exist_ok=False);before=tested_hashes();log=io.StringIO();start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames([Path(n).stem for n in TESTS])
    with contextlib.redirect_stdout(log):r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    after=tested_hashes();result={'mode':'layer_readout','status':'passed'if r.wasSuccessful()and not r.skipped and before==after else'failed',
        'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),
        'seconds':time.perf_counter()-start,'file_sha256':after,'sources_unchanged_during_tests':before==after,'network':False,'gpu':False}
    (args.output/'tests.txt').write_text(log.getvalue());(args.output/'test-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='file_sha256'},indent=2))
    if result['status']!='passed':print(log.getvalue()[-10000:]);raise SystemExit(1)


if __name__=='__main__':main()
