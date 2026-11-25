"""Package, submit, and verify the frozen 1.7B-to-8B mapper ablation."""

import argparse
import json
from pathlib import Path
import time

from kvverify.data import assert_disjoint, check_heldout, load_examples
from kvverify.paired_replay import NATIVE, select_component_cases
from kvverify.shared_cache_replay import summarize as summarize_shared

from .answer_train_loop import describe, optional_live
from .autonomous_budget import TOTAL_LIMIT_USD
from .harvey_pilot_submit import INSTANCE, PAIR, ROOT, dispatch, sha
from .mapper_ablation_worker import summarize
from .package_integrity import verify_worker_imports

PANEL = ROOT/"datasets/longbench-v2-paired-v1"
RESERVED = ROOT/"datasets/longbench-v2-readout-unseen-v1"
BASELINE = ROOT/"reports/paired-replay-longbench-v1"
TRAINED = ROOT/"reports/answer-continuation-quality-v1"
MAX_RUNTIME = 510
FAST_REPLAY_TOTAL_USD = 30.31
ACCURACY_REPLAY_TOTAL_USD = 31.09
SHARED_REPLAY_TOTAL_USD = 31.20


def read(path):
    return json.loads(path.read_text())


def prepare(*, compare_fast=False):
    prior = read(BASELINE/"submission.json")
    pilot = read(TRAINED/"submission.json")
    result = read(TRAINED/"result.json")
    checkpoint = TRAINED/"checkpoint-64"
    trained = read(checkpoint/"manifest.json")
    mapper_hash = pilot["config"]["base_weights_sha256"]
    if (prior["config"]["models"] != PAIR or pilot["config"]["models"] != PAIR
            or prior["config"]["mapper_weights_sha256"] != mapper_hash
            or prior["config"]["mapper_origin"] != pilot["config"]["mapper_origin"]
            or result["job_name"] != pilot["job_name"] or trained["steps"] != 64
            or trained["base_weights_sha256"] != mapper_hash
            or trained["weights_sha256"] != result["mapper_weights_sha256"]
            or trained["weights_sha256"] != sha(checkpoint/"weights.safetensors")):
        raise ValueError("Replay or checkpoint provenance differs")

    data = PANEL/"evaluation.jsonl"
    if sha(data) != read(PANEL/"manifest.json")["evaluation_sha256"]:
        raise ValueError("Frozen LongBench panel changed")
    examples = load_examples(data, "eval")
    gold = {item.id: item.answer for item in examples}
    prior_rows = read(BASELINE/"rows.json")
    hard = select_component_cases(prior_rows, gold, prior["config"]["repeats"])
    native_correct = {row["id"] for row in prior_rows
                      if row["repeat"] == 0 and row["arm"] == NATIVE and row["answer"] == gold[row["id"]]}
    if len(native_correct) != 13 or len(hard) != 6 or not set(hard) <= native_correct:
        raise ValueError("Unexpected archived native-correct cohort")
    cases = {item.id: {"gold": item.answer, "historical_hard": item.id in hard}
             for item in sorted(examples, key=lambda item: (item.id not in hard, item.id))
             if item.id in native_correct}
    check_heldout(trained, [item for item in examples if item.id in cases])
    inputs = {"inputs/evaluation.jsonl": data, "inputs/manifest.json": PANEL/"manifest.json",
              "inputs/checkpoint-manifest.json": checkpoint/"manifest.json"}
    files = {path.relative_to(ROOT).as_posix(): path for folder in ("kvverify", "cloud")
             for path in (ROOT/folder).glob("*.py")}
    files.update(inputs, **{"cloud/recovery_bootstrap.sh": ROOT/"cloud/recovery_bootstrap.sh"})
    config = {
        "mode": "mapper_ablation", "models": PAIR, "seed": 9814, "repeats": 2, "cases": cases,
        "max_runtime_seconds": MAX_RUNTIME, "worker_timeout_seconds": MAX_RUNTIME - 15,
        "mapper_origin": pilot["config"]["mapper_origin"], "input_mapper_dir": "/opt/ml/input/data/mapper",
        "corrected_mapper_origin": pilot["live_results"]+"checkpoint-64/",
        "base_weights_sha256": mapper_hash, "prior_replay_sha256": sha(BASELINE/"rows.json"),
        "input_sha256": {name: sha(path) for name, path in inputs.items()},
        "verified_worker_imports": verify_worker_imports(files, "cloud/mapper_ablation_worker.py"),
    }
    if compare_fast:
        config["compare_fast"] = True
    return config, files


def prepare_replication(speed_report, *, with_excerpts=False):
    config, files = prepare(compare_fast=True)
    receipt = read(speed_report/"submission.json")
    result = read(speed_report/"result.json")
    native = result["native_correct"]
    optimized = native["arms"]["fast_trained"]
    reference = native["arms"]["trained"]
    if (receipt["status"] != "submitted" or receipt["job_name"] != result["job_name"]
            or receipt["config"].get("compare_fast") is not True
            or receipt["config"].get("reserved_panel") or receipt["config"].get("with_excerpts")
            or native["questions"] != len(config["cases"]) or native["fast_option_agreement"] != native["questions"]
            or optimized["mapper_ms"] > 0.75*reference["mapper_ms"]):
        raise ValueError("The frozen L40S replay has not cleared the speed and agreement gate")
    for name in ("kvverify/fast_affine.py", "kvverify/continuation_pilot.py", "kvverify/native_prefix.py",
                 "kvverify/hf_backend.py", "inputs/checkpoint-manifest.json"):
        if sha(files[name]) != receipt["file_sha256"][name]:
            raise ValueError("The mapper or receiver changed since the speed gate: " + name)
    panel = PANEL if with_excerpts else RESERVED
    data, manifest_path = panel/"evaluation.jsonl", panel/"manifest.json"
    manifest = read(manifest_path)
    examples = load_examples(data, "eval")
    if (sha(data) != manifest["evaluation_sha256"] or len(examples) != manifest["questions"]
            or len({item.document_id for item in examples}) != len(examples)):
        raise ValueError("Evaluation panel freeze differs")
    if not with_excerpts:
        assert_disjoint(load_examples(PANEL/"evaluation.jsonl", "eval"), examples)
    check_heldout(read(files["inputs/checkpoint-manifest.json"]), examples)
    hard = {identifier for identifier, case in config["cases"].items() if case["historical_hard"]}
    seconds = 1100 if with_excerpts else 650
    config.update(
        reserved_panel=not with_excerpts, with_excerpts=with_excerpts,
        cases={item.id: {"gold": item.answer, "historical_hard": item.id in hard} for item in examples},
        max_runtime_seconds=seconds, worker_timeout_seconds=seconds - 15,
        speed_replay_sha256=sha(speed_report/"result.json"),
    )
    inputs = {"inputs/evaluation.jsonl": data, "inputs/manifest.json": manifest_path}
    files.update(inputs)
    config["input_sha256"].update({name: sha(path) for name, path in inputs.items()})
    return config, files


def prepare_shared_cache(accuracy_report):
    speed = ROOT/"reports/mapper-fast-l40s-v1"
    config, files = prepare_replication(speed, with_excerpts=True)
    receipt = read(accuracy_report/"submission.json")
    result = read(accuracy_report/"result.json")
    previous_rows = read(accuracy_report/"rows.json")
    if (receipt["status"] != "submitted" or receipt["job_name"] != result["job_name"]
            or receipt["config"].get("with_excerpts") is not True
            or receipt["config"]["input_sha256"] != config["input_sha256"]
            or receipt["config"]["cases"] != config["cases"]
            or result["development_documents"] != len(config["cases"])):
        raise ValueError("Shared-cache replay requires the completed development accuracy run")
    original = list(read(speed/"submission.json")["config"]["cases"])
    previous_wins = [row["id"] for row in previous_rows if row["id"] in original
                     and row["arms"]["fast_trained"][0]["answer"] == config["cases"][row["id"]]["gold"]]
    if len(previous_wins) != 8 or len(set(previous_wins)) != 8:
        raise ValueError("The archived eight correct answers changed")
    config.update(with_excerpts=False, shared_cache=True, original_native_correct=original,
                  original_fast_correct=previous_wins,
                  max_runtime_seconds=600, worker_timeout_seconds=585,
                  accuracy_replay_sha256=sha(accuracy_report/"result.json"))
    return config, files


def collect(output):
    receipt = read(output/"submission.json")
    cfg = receipt["config"]
    if receipt["status"] != "submitted" or cfg["mode"] != "mapper_ablation":
        raise ValueError("Expected a submitted mapper ablation")
    job, region, live = receipt["job_name"], receipt["region"], receipt["live_results"]
    previous = None
    while True:
        state = describe(job, region=region)
        status, phase = state["TrainingJobStatus"], state["SecondaryStatus"]
        progress = optional_live(live+"progress.json", region=region) if phase == "Training" else None
        if (status, phase, progress) != previous:
            previous = status, phase, progress
            print(json.dumps({"status": status, "phase": phase, "progress": progress}), flush=True)
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(15)
    (output/"aws-final.json").write_text(json.dumps(state, indent=2)+"\n")
    payloads = {}
    for name in ("protocol", "rows", "summary", "progress"):
        payload = optional_live(live+name+".json", region=region)
        if payload is not None:
            payloads[name] = payload
            (output/(name+".json")).write_text(json.dumps(payload, indent=2)+"\n")
    if status != "Completed":
        raise RuntimeError(state.get("FailureReason") or f"SageMaker job {status}")
    if cfg.get("shared_cache"):
        summary = summarize_shared(payloads["rows"], cfg["cases"], cfg["original_native_correct"],
                                   cfg["original_fast_correct"], cfg["repeats"])
    else:
        summary = summarize(payloads["rows"], cfg["cases"], cfg["repeats"],
                            compare_fast=cfg.get("compare_fast", False), reserved_panel=cfg.get("reserved_panel", False),
                            with_excerpts=cfg.get("with_excerpts", False))
    if summary != payloads["summary"] or payloads["protocol"]["cases"] != cfg["cases"]:
        raise ValueError("Saved replay differs from the packaged experiment")
    seconds = state["BillableTimeInSeconds"]
    price = receipt["budget_ledger"]["price"]["usd_per_hour"]
    summary.update(job_name=job, billable_seconds=seconds, estimated_compute_usd=seconds*price/3600)
    (output/"result.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "questions"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--with-fast", action="store_true", help="Compare the optimized mapper with its reference")
    variants.add_argument("--reserved-panel", type=Path, metavar="SPEED_REPORT",
                          help="Prepare the reserved replication using a passing speed replay")
    variants.add_argument("--with-excerpts", type=Path, metavar="SPEED_REPORT",
                          help="Compare the optimized mapper with and without excerpts on all development documents")
    variants.add_argument("--shared-cache", type=Path, metavar="ACCURACY_REPORT",
                          help="Time both suffix passes on a single translated cache")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--approved-fast-replay-overage", action="store_true",
                        help="Apply the approved $30.31 total cap to this optimized replay")
    budget.add_argument("--approved-accuracy-replay-overage", action="store_true",
                        help="Apply the approved $31.09 total cap to this development accuracy replay")
    budget.add_argument("--approved-shared-cache-overage", action="store_true",
                        help="Apply the approved $31.20 total cap to this shared-cache replay")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.approved_fast_replay_overage and not (args.submit and args.with_fast):
        parser.error("The replay overage requires --submit and --with-fast")
    if args.approved_accuracy_replay_overage and not (args.submit and args.with_excerpts):
        parser.error("The accuracy overage requires --submit and --with-excerpts")
    if args.approved_shared_cache_overage and not (args.submit and args.shared_cache):
        parser.error("The shared-cache overage requires --submit and --shared-cache")
    if args.collect:
        collect(args.output)
    else:
        speed_report = args.reserved_panel or args.with_excerpts
        if args.shared_cache:
            config, files = prepare_shared_cache(args.shared_cache)
        elif speed_report:
            config, files = prepare_replication(speed_report, with_excerpts=args.with_excerpts is not None)
        else:
            config, files = prepare(compare_fast=args.with_fast)
        authorized_total = TOTAL_LIMIT_USD
        if args.approved_fast_replay_overage:
            authorized_total = FAST_REPLAY_TOTAL_USD
        if args.approved_accuracy_replay_overage:
            authorized_total = ACCURACY_REPLAY_TOTAL_USD
        if args.approved_shared_cache_overage:
            authorized_total = SHARED_REPLAY_TOTAL_USD
        dispatch(config, files, instance=INSTANCE, output=args.output, submit=args.submit, job_kind="mapper-ablation",
                 mapper_input=True, authorized_total_usd=authorized_total)


if __name__ == "__main__":
    main()
