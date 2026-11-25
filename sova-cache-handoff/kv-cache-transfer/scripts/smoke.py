"""Run the scoped offline suite and save an auditable smoke report."""
import argparse
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-torch", action="store_true", help="Opt into tiny random CPU Qwen3 tests; never downloads weights")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KVVERIFY_TORCH_SMOKE"] = "1" if args.with_torch else "0"
    versions = {}
    if args.with_torch:
        for name in ("torch", "transformers", "safetensors"):
            if importlib.util.find_spec(name) is None:
                raise SystemExit(f"{name} is missing; nothing is installed automatically")
            versions[name] = importlib.metadata.version(name)
        if versions["transformers"] != "4.52.4":
            raise SystemExit("The opt-in adapter smoke targets transformers==4.52.4; do not alter another environment silently")
    log = io.StringIO()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    text = log.getvalue()
    (output/"tests.txt").write_text(text)
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "seconds": time.perf_counter()-started, "with_tiny_cpu_models": args.with_torch,
              "versions": versions, "network": False, "gpu": False, "model_downloads": False,
              "thesis_verified": False, "scope": "Software and tiny random-model correctness checks, not pretrained model performance"}
    (output/"test-results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(text, end="")
    if not result.wasSuccessful():
        raise SystemExit(1)
    from kvverify.smoke import run
    run(output/"integration")
    print("Offline smoke reports:", output)


if __name__ == "__main__":
    main()
