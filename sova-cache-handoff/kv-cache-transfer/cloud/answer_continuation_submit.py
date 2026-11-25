"""Launch and collect the fixed-budget generated-answer continuation pilot."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from kvverify.continuation_data import load_papers
from kvverify.data import assert_disjoint, check_heldout

from .answer_continuation_worker import summarize
from .answer_train_loop import describe, optional_live
from .harvey_pilot_submit import PAIR, ROOT, dispatch, sha
from .package_integrity import verify_worker_imports
from .submit import BUCKET, REGION, aws, training_prices

DATASETS = {"qasper": ROOT/"datasets/qasper-continuation-pilot-v1",
            "quality": ROOT/"datasets/quality-continuation-pilot-v2"}
INSTANCE = "ml.g6e.2xlarge"


def prepare(dataset="qasper", seconds=1650):
    parent = json.loads((ROOT/"reports/quality-8b-v1/submission.json").read_text())
    if parent["config"]["models"] != PAIR or describe(parent["job_name"])["TrainingJobStatus"] != "Completed":
        raise ValueError("The pilot requires the verified 1.7B-to-8B affine initializer")
    origin = f"s3://{BUCKET}/kv-transfer/{parent['job_name']}/live/ridge/"
    base = optional_live(origin + "manifest.json")
    data = DATASETS[dataset]
    manifest = json.loads((data/"manifest.json").read_text())
    if not base or base["mapper_format"] != "affine-cache-v1" or base["tokenizer_sha256"] != manifest["tokenizer_fingerprint"]:
        raise ValueError("The frozen tokenizer and initializer differ")
    for filename in ("train.jsonl", "dev.jsonl"):
        if sha(data/filename) != manifest["files"][filename]["sha256"]:
            raise ValueError("The dataset freeze differs: " + filename)
    train = load_papers(data/"train.jsonl")
    dev = [row for questions in load_papers(data/"dev.jsonl").values() for row in questions]
    assert_disjoint([row for questions in train.values() for row in questions], dev)
    check_heldout(base, dev)
    files = {path.relative_to(ROOT).as_posix(): path for folder in ("kvverify", "cloud")
             for path in (ROOT/folder).glob("*.py")}
    files["cloud/recovery_bootstrap.sh"] = ROOT/"cloud/recovery_bootstrap.sh"
    inputs = {f"inputs/{name}": data/name for name in ("train.jsonl", "dev.jsonl", "manifest.json")}
    files.update(inputs)
    config = {
        "mode": "answer_continuation", "models": PAIR, "seed": 8127, "rank": 16, "donors": 3,
        "learning_rate": .0006, "training_documents": 64, "max_answer_tokens": 24,
        "dataset": dataset, "decision_weight": .5 if dataset == "quality" else 0.,
        "mapper_origin": origin, "base_weights_sha256": base["weights_sha256"],
        "input_mapper_dir": "/opt/ml/input/data/mapper",
        "max_runtime_seconds": seconds, "worker_timeout_seconds": seconds - 220,
        "input_sha256": {name: sha(path) for name, path in inputs.items()},
        "verified_worker_imports": verify_worker_imports(files, "cloud/answer_continuation_worker.py"),
    }
    return config, files


def collect(output):
    receipt = json.loads((output/"submission.json").read_text())
    if receipt["status"] != "submitted" or receipt["config"]["mode"] != "answer_continuation":
        raise ValueError("Expected a submitted answer-continuation pilot")
    region, live = receipt["region"], receipt["live_results"]
    prior = None
    while True:
        state = describe(receipt["job_name"], region=region)
        status, phase = state["TrainingJobStatus"], state["SecondaryStatus"]
        progress = optional_live(live+"progress.json", region=region) if phase == "Training" else None
        current = (status, phase, progress)
        if current != prior:
            print(json.dumps({"status": status, "phase": phase, "progress": progress}), flush=True)
            prior = current
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(22)
    (output/"aws-final.json").write_text(json.dumps(state, indent=2)+"\n")
    for name in ("progress", "teacher-audit", "verified-teacher", "teacher-manifest", "training-history", "evaluation"):
        payload = optional_live(live+name+".json", region=region)
        if payload is not None:
            (output/(name+".json")).write_text(json.dumps(payload, indent=2)+"\n")
    if status != "Completed":
        try:
            error = aws("s3", "cp", live+"failure.txt", "-", "--only-show-errors", region=region)
        except subprocess.CalledProcessError:
            error = state.get("FailureReason", status)
        (output/"failure.txt").write_text(error+"\n")
        raise RuntimeError(error)
    summary = optional_live(live+"summary.json", region=region)
    m = optional_live(live+"checkpoint-64/manifest.json", region=region)
    cfg = receipt["config"]
    data = DATASETS[cfg.get("dataset", "qasper")]
    for file in ("train.jsonl", "dev.jsonl", "manifest.json"):
        if sha(data/file) != cfg["input_sha256"]["inputs/"+file]:
            raise ValueError("Original data differs from the packaged input: "+file)
    dev = {q.id: q for questions in load_papers(data/"dev.jsonl").values() for q in questions}
    rows = json.loads((output/"evaluation.json").read_text())
    teacher = json.loads((output/"verified-teacher.json").read_text())
    train = {q.id: q for questions in load_papers(data/"train.jsonl").values() for q in questions}
    if ({row["id"] for row in rows} != set(dev) or
            any(item["correct"] != (item["ended"] and dev[row["id"]].correct(item["answer"]))
                for row in rows for item in row["routes"].values())):
        raise ValueError("Held-out accuracy differs from the recorded annotation")
    if (len(teacher) != 64 or len({row["document_id"] for row in teacher}) != 64 or
            any(not train[row["id"]].correct(row["answer"]) or row["document_id"] != train[row["id"]].document_id
                for row in teacher)):
        raise ValueError("Verified teacher corpus differs from the frozen annotation")
    if (not summary or not m or m["steps"] != 64 or m["base_weights_sha256"] != cfg["base_weights_sha256"] or
            summary != summarize(rows, 64)):
        raise ValueError("Checkpoint and recomputed results do not match the worker")
    check_heldout(m, list(dev.values()))
    allocation = state["ResourceConfig"]
    instance = allocation.get("SelectedInstanceType") or allocation["InstanceType"]
    seconds = state["BillableTimeInSeconds"]
    summary.update(job_name=receipt["job_name"], instance_type=instance, billable_seconds=seconds,
                   estimated_compute_usd=seconds * training_prices((instance,), region=region)[instance]["usd_per_hour"]/3600,
                   teacher_artifact=live+"teacher-logprobs.safetensors", mapper_weights_sha256=m["weights_sha256"])
    for name, payload in (("result", summary), ("mapper", m)):
        (output/(name+".json")).write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="qasper")
    parser.add_argument("--max-runtime", type=int, default=1650)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not 900 <= args.max_runtime <= 1650:
        parser.error("The pilot supports a bounded runtime of 900–1650 seconds")
    if args.collect:
        collect(args.output)
    else:
        config, files = prepare(args.dataset, args.max_runtime)
        dispatch(config, files, instance=INSTANCE, output=args.output, submit=args.submit, job_kind="answer-cont",
                 mapper_input=True, preferences=(INSTANCE, "ml.g6e.xlarge"))


if __name__ == "__main__":
    main()
