"""Submit or collect one bounded affine readout correction on SageMaker."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from kvverify.data import load_examples
from kvverify.paired_replay import summarize

from .answer_train_loop import describe, optional_live
from .harvey_pilot_submit import ROOT, dispatch, sha
from .package_integrity import verify_worker_imports
from .paired_replay_submit import CONFIRMATION, DATA, INSTANCE, L40S_PREFERENCES, QWEN_4B, prepare as prepare_replay
from .readout_train_worker import correction_gate
from .submit import REGION, aws, training_prices

TRAIN = ROOT/"datasets/quality-readout-train-v1"
LONG_TRAIN = ROOT/"datasets/quality-readout-long-train-v1"
ARCHIVED_TRAIN = ROOT/"datasets/quality-original-ridge-train-v1"
PREFERENCES = tuple(item for item in L40S_PREFERENCES if item != "ml.g6e.16xlarge")


def prepare(layers, *, student_trajectory=False, ridge_scale=False, resume_ridge=None,
            layer_selection=False, sequence_kl=False, fitted_donor=None):
    config, files = prepare_replay()
    confirm = json.loads((CONFIRMATION/"manifest.json").read_text())
    training_source = (LONG_TRAIN if student_trajectory or sequence_kl else
                       ARCHIVED_TRAIN if ridge_scale or layer_selection else TRAIN)
    train = json.loads((training_source/"manifest.json").read_text())
    extra = {
        "inputs/readout-train.jsonl": training_source/"train.jsonl",
        "inputs/readout-train-manifest.json": training_source/"manifest.json",
        "inputs/replication.jsonl": CONFIRMATION/"evaluation.jsonl",
        "inputs/replication-manifest.json": CONFIRMATION/"manifest.json",
    }
    if layer_selection:
        extra["inputs/layer-selection-pool.jsonl"] = TRAIN/"train.jsonl"
        extra["inputs/layer-selection-manifest.json"] = TRAIN/"manifest.json"
    hashes = {name: sha(path) for name, path in extra.items()}
    if (hashes["inputs/replication.jsonl"] != confirm["evaluation_sha256"]
            or hashes["inputs/readout-train.jsonl"] != train["training_sha256"]):
        raise ValueError("Readout input snapshot differs")
    if layer_selection and hashes["inputs/layer-selection-pool.jsonl"] != json.loads(
            (TRAIN/"manifest.json").read_text())["training_sha256"]:
        raise ValueError("Layer selection source differs from its freeze")
    files.update(extra)
    config["input_sha256"].update(hashes)
    config.update(mode="readout_train", readout_layers=layers, repeats=2, readout_rank=16,
                  readout_queries=8, student_trajectory=student_trajectory, ridge_scale=ridge_scale,
                  layer_selection=layer_selection, sequence_kl=sequence_kl,
                  readout_learning_rate=.0006 if student_trajectory or sequence_kl else .0015,
                  max_runtime_seconds=3000 if sequence_kl else 2700 if student_trajectory or layer_selection else 2100 if ridge_scale else 2400,
                  worker_timeout_seconds=2640 if sequence_kl else 2340 if student_trajectory or layer_selection else 1740 if ridge_scale else 2040,
                  verified_worker_imports=verify_worker_imports(files, "cloud/readout_train_worker.py"))
    if resume_ridge:
        prepare_ridge_resume(config, files, resume_ridge)
    if fitted_donor:
        parent = json.loads((fitted_donor/"submission.json").read_text())
        outcome = json.loads((fitted_donor/"result.json").read_text())
        mapper = json.loads((fitted_donor/"mapper.json").read_text())
        original_panel = config["input_sha256"]["inputs/evaluation.jsonl"]
        if (parent["job_name"] != outcome["job_name"] or parent["region"] != REGION
                or not parent["config"].get("fresh_pair") or parent["config"]["models"][0] != QWEN_4B
                or parent["config"]["models"][1] != config["models"][1]
                or parent["config"]["input_sha256"]["inputs/evaluation.jsonl"] != original_panel
                or outcome["native_warm_agreement"] != outcome["questions"]):
            raise ValueError("Functional donor correction requires a verified matched-pair initializer")
        config.update(models=parent["config"]["models"], parent_replay_job=parent["job_name"],
                      mapper_origin=parent["live_results"]+"mapper/", mapper_weights_sha256=mapper["weights_sha256"],
                      max_runtime_seconds=2400, worker_timeout_seconds=2040)
    return config, files


def prepare_ridge_resume(config, files, prior_folder):
    prior = json.loads((prior_folder/"submission.json").read_text())
    if (prior["status"] != "submitted" or not prior["config"].get("ridge_scale") or prior["config"].get("resume_ridge")
            or prior["region"] != REGION
            or prior["config"]["mapper_weights_sha256"] != config["mapper_weights_sha256"]
            or prior["config"]["input_sha256"] != config["input_sha256"]
            or describe(prior["job_name"], region=prior["region"])["TrainingJobStatus"] != "Failed"
            or "Inference tensors do not track version counter" not in (prior_folder/"failure.txt").read_text()):
        raise ValueError("Ridge resume requires the failed exact fit and the same frozen inputs")
    live = prior["live_results"]
    manifest = optional_live(live+"mapped/manifest.json", region=prior["region"])
    if (not manifest or manifest["comparison_parent_weights_sha256"] != config["mapper_weights_sha256"]
            or "diag(feature variance)" not in manifest["regularization"]):
        raise ValueError("No verified calibrated ridge checkpoint to resume")
    for key, filename in (("history", "training-history.json"), ("audit", "ridge-scale-audit.json")):
        payload = optional_live(live+filename, region=prior["region"])
        if not payload:
            raise ValueError("The saved ridge fit is missing "+key)
        if key == "audit" and (payload["samples"] != manifest["calibration_tokens"] or
                               not all(value <= .001 for value in payload["original_reproduction_relative_error"].values())):
            raise ValueError("The archived fit reproduction audit failed")
        path = prior_folder / filename
        path.write_text(json.dumps(payload, indent=2)+"\n")
        name = "inputs/archived-ridge-"+key+".json"
        files[name] = path
        config["input_sha256"][name] = sha(path)
    config.update(resume_ridge={"job_name": prior["job_name"], "weights_sha256": manifest["weights_sha256"]},
                  corrected_mapper_origin=live+"mapped/")


def collect(output):
    receipt = json.loads((output/"submission.json").read_text())
    config = receipt["config"]
    if receipt["status"] != "submitted" or config["mode"] != "readout_train":
        raise ValueError("Expected submitted readout run")
    live = receipt["live_results"]
    region = receipt["region"]
    previous = None
    while True:
        state = describe(receipt["job_name"], region=region)
        progress = optional_live(live+"progress.json", region=region) if state["SecondaryStatus"] == "Training" else None
        snapshot = {"status": state["TrainingJobStatus"], "phase": state["SecondaryStatus"], "progress": progress}
        if snapshot != previous:
            print(json.dumps(snapshot), flush=True)
            previous = snapshot
        if snapshot["status"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(20)
    if snapshot["status"] != "Completed":
        reason = state.get("FailureReason", snapshot["status"])
        try:
            reason = aws("s3", "cp", live+"failure.txt", "-", "--only-show-errors", region=region)
        except subprocess.CalledProcessError:
            pass
        (output/"failure.txt").write_text(reason+"\n")
        raise RuntimeError(reason)

    result = optional_live(live+"summary.json", region=region)
    history = optional_live(live+"training-history.json", region=region)
    protocol = optional_live(live+"protocol.json", region=region)
    mapper = optional_live(live+"mapped/manifest.json", region=region)
    if (not result or not history or not protocol or not mapper
            or mapper["weights_sha256"] != result["weights_sha256"]
            or mapper["comparison_parent_weights_sha256"] != config["mapper_weights_sha256"]
            or protocol["base_weights_sha256"] != config["mapper_weights_sha256"]
            or len(history) != result["steps"]):
        raise ValueError("Incomplete training evidence or weights identity mismatch")

    for name, directory, filename in (("development", DATA, "evaluation"), ("replication", CONFIRMATION, "replication")):
        if result[name] is None:
            continue
        path = directory/"evaluation.jsonl"
        if sha(path) != config["input_sha256"][f"inputs/{filename}.jsonl"]:
            raise ValueError("Collection input snapshot differs")
        rows = optional_live(live+name+"-rows.json", region=region)
        gold = {item.id: item.answer for item in load_examples(path, "eval")}
        if summarize(rows, gold, config["repeats"], corrected=True) != result[name]:
            raise ValueError("Local paired metrics differ from worker report")
        (output/(name+"-rows.json")).write_text(json.dumps(rows, indent=2)+"\n")

    if correction_gate(result["development"]) != result["status"]:
        raise ValueError("Collection did not reproduce the correction gate")
    allocation = state["ResourceConfig"]
    instance = allocation.get("SelectedInstanceType") or allocation["InstanceType"]
    seconds = state["BillableTimeInSeconds"]
    result.update(job_name=receipt["job_name"], billable_seconds=seconds, instance_type=instance,
                  estimated_compute_usd=seconds * training_prices((instance,), region=region)[instance]["usd_per_hour"] / 3600)
    for name, data in (("result", result), ("training-history", history), ("protocol", protocol), ("mapper", mapper)):
        (output/(name+".json")).write_text(json.dumps(data, indent=2)+"\n")
    if config.get("ridge_scale") or config.get("layer_selection"):
        audit = optional_live(live+"ridge-scale-audit.json", region=region)
        if not audit or not all(value <= .001 for value in audit["original_reproduction_relative_error"].values()):
            raise ValueError("Missing archived ridge reproduction audit")
        (output/"ridge-scale-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    if config.get("layer_selection"):
        audit = optional_live(live+"layer-selection-audit.json", region=region)
        if not audit or audit["source_layers"] != mapper["source_layers"] or len(audit["validation_ids"]) != 16:
            raise ValueError("Layer selection evidence or mapper identity is missing")
        (output/"layer-selection-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(36)))
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--sequence-kl", action="store_true", help="Distill several query positions through the full receiver")
    variant.add_argument("--select-source-layers", action="store_true", help="Select one donor layer per receiver layer")
    variant.add_argument("--resume-ridge", type=Path, help="Score the frozen fit from a failed ridge smoke")
    variant.add_argument("--ridge-scale", action="store_true", help="Change only ridge feature-scale regularization")
    variant.add_argument("--student-trajectory", action="store_true", help="Fit actual mapped-path queries on composed long histories")
    parser.add_argument("--init-4b", type=Path, help="Use a verified depth-matched donor as initializer for the student loss")
    parser.add_argument("--concurrent-sequence", action="store_true", help="Reserve this sequence run alongside one active job")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.init_4b and not args.student_trajectory:
        parser.error("--init-4b requires --student-trajectory")
    if args.concurrent_sequence and not (args.sequence_kl and args.submit):
        parser.error("--concurrent-sequence applies to submitting the sequence loss")
    if (args.ridge_scale or args.resume_ridge or args.select_source_layers or args.sequence_kl) and args.layers != list(range(36)):
        parser.error("This control applies to the full affine map")
    if args.collect:
        collect(args.output)
    else:
        if args.layers != sorted(set(args.layers)) or not 0 <= args.layers[0] <= args.layers[-1] < 36:
            parser.error("--layers must be unique, ascending indices between 0 and 35")
        config, files = prepare(args.layers, student_trajectory=args.student_trajectory,
                                ridge_scale=args.ridge_scale or bool(args.resume_ridge), resume_ridge=args.resume_ridge,
                                layer_selection=args.select_source_layers, sequence_kl=args.sequence_kl,
                                fitted_donor=args.init_4b)
        fixed_instance = args.sequence_kl or args.init_4b
        dispatch(config, files, instance="ml.g6e.2xlarge" if fixed_instance else INSTANCE,
                 output=args.output, submit=args.submit,
                 job_kind="4b-readout" if args.init_4b else "sequence-kl" if args.sequence_kl
                 else "student-readout" if args.student_trajectory else "ridge-replay" if args.resume_ridge
                 else "ridge-scale" if args.ridge_scale else "source-layer-choice" if args.select_source_layers else "joint-readout",
                 mapper_input=True, preferences=() if fixed_instance else PREFERENCES,
                 allow_active_workers=int(args.concurrent_sequence))


if __name__ == "__main__":
    main()
