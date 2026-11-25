"""Balanced arm order, explicit warmup and durable attempt records."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import traceback

from .data import json_digest
from .evidence import ARMS, write_report, validate_trace


def evaluate(runtime, examples, output, repeats=1, warmups=1, max_new_tokens=4, seed=19):
    if repeats < 1 or warmups < 0 or max_new_tokens < 1:
        raise ValueError("Invalid evaluation limits")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    config = {"source_identity": runtime.identities["source"], "target_identity": runtime.identities["target"],
              "tokenizer_sha256": runtime.tokenizer_sha256, "repeats": repeats, "warmups": warmups,
              "max_new_tokens": max_new_tokens, "seed": seed,
              "mapper": runtime.mapper.manifest,
              "examples_sha256": json_digest([asdict(e) for e in examples]),
              "runtime": "HF reference, both models resident, batch 1, greedy, no prefix-cache reuse across cold calls"}
    config["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")}
    config["source_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    config_hash = json_digest(config)
    (root/"run-config.json").write_text(json.dumps(config, indent=2)+"\n")
    rows = []
    rng = random.Random(seed)
    try:
        # Warmup is explicit and logged separately, never treated as quality data.
        with (root/"warmup.jsonl").open("a") as stream:
            for n in range(warmups):
                for arm in ARMS:
                    row = runtime.answer(examples[0], arm, n, config_hash, max_new_tokens)
                    validate_trace(row)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    stream.flush()
        for repeat in range(repeats):
            for example in examples:
                ids = runtime.tokenize(example)
                order = list(ARMS)
                rng.shuffle(order)
                for arm in order:
                    try:
                        row = runtime.answer(example, arm, repeat, config_hash, max_new_tokens)
                        validate_trace(row)
                    except Exception as exc:
                        row = {"id": example.id, "document_id": example.document_id, "context_hash": example.context_hash,
                               "prompt_hash": json_digest(ids), "run_config_hash": config_hash, "repeat": repeat,
                               "arm": arm, "evidence_kind": "hf_model", "status": "error", "correct": False,
                               "error_type": type(exc).__name__, "error": str(exc), "fallback": False}
                    rows.append(row)
                    with (root/"traces.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                print(f"repeat={repeat} example={example.id} completed_arms={len(rows)}", flush=True)
    except BaseException:
        (root/"interrupted.txt").write_text(traceback.format_exc())
        # Keep the original partial traces; do not invent missing paired arms.
        raise
    return write_report(root, rows, {"run_config": config, "config_sha256": config_hash,
                                    "empirical": True, "cost": "not measured"}, allow_existing=True)
