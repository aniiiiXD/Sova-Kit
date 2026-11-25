"""Launch or collect a single frozen inference-only SageMaker replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import tempfile
import time

from kvverify.data import assert_disjoint, check_heldout, load_examples
from kvverify.paired_replay import TAIL_REPAIR, select_component_cases, summarize, summarize_components

from .answer_train_loop import describe, optional_live
from .autonomous_budget import TOTAL_LIMIT_USD, reconcile
from .harvey_pilot_submit import PAIR, ROOT, dispatch, sha
from .package_integrity import verify_worker_imports
from .submit import BUCKET, REGION, aws, training_prices

DATA = ROOT / "datasets/longbench-v2-paired-v1"
CONFIRMATION = ROOT / "datasets/longbench-v2-tail-confirmation-v1"
UNSEEN = ROOT / "datasets/longbench-v2-readout-unseen-v1"
FRESH_CALIBRATION = ROOT / "datasets/quality-original-ridge-train-v1"
QWEN_4B = {"repo": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"}
INSTANCE = "ml.g6e.8xlarge"
MUMBAI = "ap-south-1"
MUMBAI_BUCKET = "sova-finance1-artifacts-169970912263"
L40S_PREFERENCES = ("ml.g6e.2xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.xlarge", "ml.g6e.16xlarge")


def prepare(component_source=None, *, tail_repair=False, confirm=None, causal_layers=False,
            evidence=False, corrected_from=None, evidence_from=None, fresh_pair=False, donor_from=None):
    parent = json.loads((ROOT/"reports/quality-8b-v1/submission.json").read_text())
    if parent["config"]["models"] != PAIR or describe(parent["job_name"])["TrainingJobStatus"] != "Completed":
        raise ValueError("Replay requires the completed historical Qwen3 pair")
    origin = f"s3://{BUCKET}/kv-transfer/{parent['job_name']}/live/ridge/"
    mapper = json.loads(aws("s3", "cp", origin+"manifest.json", "-", "--only-show-errors"))
    dataset = UNSEEN if corrected_from else CONFIRMATION if confirm or evidence_from or donor_from else DATA
    manifest = json.loads((dataset/"manifest.json").read_text())
    data = {"inputs/evaluation.jsonl": dataset/"evaluation.jsonl", "inputs/manifest.json": dataset/"manifest.json"}
    if (sha(data["inputs/evaluation.jsonl"]) != manifest["evaluation_sha256"]
            or mapper.get("mapper_format") != "affine-cache-v1" or mapper.get("head_mixing") is not True
            or mapper["tokenizer_sha256"] != manifest["tokenizer_fingerprint"]):
        raise ValueError("Candidate or dataset does not match the frozen replay")
    examples = load_examples(data["inputs/evaluation.jsonl"], "eval")
    check_heldout(mapper, examples)
    files = {path.relative_to(ROOT).as_posix(): path for folder in ("kvverify", "cloud")
             for path in (ROOT/folder).glob("*.py")}
    files["cloud/recovery_bootstrap.sh"] = ROOT/"cloud/recovery_bootstrap.sh"
    files.update(data)
    config = {"mode": "paired_replay", "models": PAIR, "questions": manifest["questions"],
              "repeats": 3, "seed": 1029, "max_prompt_tokens": manifest["filter"]["max_prompt_tokens"],
              "max_runtime_seconds": 2400, "worker_timeout_seconds": 2040,
              "mapper_origin": origin, "input_mapper_dir": "/opt/ml/input/data/mapper",
              "mapper_weights_sha256": mapper["weights_sha256"],
              "input_sha256": {name: sha(path) for name, path in data.items()},
              "minimum_incremental_mean_saving": .10,
              "verified_worker_imports": verify_worker_imports(files, "cloud/paired_replay_worker.py")}
    if component_source:
        prior = json.loads((component_source/"submission.json").read_text())
        prior_config = prior["config"]
        parent_result = json.loads((component_source/"result.json").read_text())
        if (prior["job_name"] != parent_result["job_name"] or prior_config.get("component_selection") or prior_config.get("tail_repair_cases")
                or prior_config["mapper_weights_sha256"] != config["mapper_weights_sha256"]
                or prior_config["input_sha256"] != config["input_sha256"]
                or parent_result["decision"] != "investigate_mapper_before_router"):
            raise ValueError("Component probe requires the verified replay with this mapper and dataset")
        rows_file = component_source/"rows.json"
        gold = {item.id: item.answer for item in examples}
        selection = select_component_cases(json.loads(rows_file.read_text()), gold, prior_config["repeats"])
        if len(selection) != 6:
            raise ValueError("Expected six replay questions requiring full native 8B")
        config.update({"parent_replay_job": prior["job_name"], "parent_rows_sha256": sha(rows_file)})
        if tail_repair:
            config.update(tail_repair_cases=selection, repeats=2, max_runtime_seconds=1800, worker_timeout_seconds=1440)
        else:
            config.update(component_selection=selection, repeats=1, max_runtime_seconds=1200, worker_timeout_seconds=840)
            if causal_layers:
                config["layer_groups"] = {"largest_readout_error": [2, 9, 22, 31], "matched_four": [5, 12, 25, 34],
                                          "early_twelve": list(range(12)), "middle_twelve": list(range(12,24)),
                                          "late_twelve": list(range(24,36))}
    if confirm:
        prior = json.loads((confirm/"submission.json").read_text())
        result = json.loads((confirm/"result.json").read_text())
        if (prior["job_name"] != result["job_name"] or not prior["config"].get("tail_repair_cases")
                or prior["config"]["mapper_weights_sha256"] != config["mapper_weights_sha256"]
                or result["decision"] != "test_fixed_router_before_continual_learning"
                or manifest["excluded_prior_cohort_sha256"] != sha(DATA/"manifest.json")):
            raise ValueError("Fresh replay requires a passing development variant and disjoint manifest")
        assert_disjoint(load_examples(DATA/"evaluation.jsonl", "eval"), examples)
        config.update(confirmation=True, tail_arms=["sink1_tail31"], parent_replay_job=prior["job_name"],
                      parent_rows_sha256=sha(confirm/"rows.json"), repeats=2,
                      max_runtime_seconds=1800, worker_timeout_seconds=1440)
    if evidence or evidence_from:
        config.update(evidence_replay=True, repeats=2, max_runtime_seconds=1500, worker_timeout_seconds=1140)
    if evidence_from:
        prior = json.loads((evidence_from/"submission.json").read_text())
        result = json.loads((evidence_from/"result.json").read_text())
        if (prior["job_name"] != result["job_name"] or not prior["config"].get("evidence_replay")
                or prior["config"].get("confirmation")
                or prior["config"]["mapper_weights_sha256"] != config["mapper_weights_sha256"]
                or prior["file_sha256"]["kvverify/evidence_replay.py"] != sha(files["kvverify/evidence_replay.py"])
                or result["decision"] != "replicate_query_aware_transfer"
                or manifest["excluded_prior_cohort_sha256"] != sha(DATA/"manifest.json")):
            raise ValueError("Evidence replication requires a verified first-panel gain")
        assert_disjoint(load_examples(DATA/"evaluation.jsonl", "eval"), examples)
        config.update(confirmation=True, parent_replay_job=prior["job_name"])
    if corrected_from:
        from .readout_train_worker import correction_gate

        parent = json.loads((corrected_from/"submission.json").read_text())
        result = json.loads((corrected_from/"result.json").read_text())
        final = json.loads((corrected_from/"mapper.json").read_text())
        if parent["config"]["models"] == [QWEN_4B, PAIR[1]]:
            original = optional_live(parent["config"]["mapper_origin"]+"manifest.json", region=REGION)
            if not original or original["weights_sha256"] != parent["config"]["mapper_weights_sha256"]:
                raise ValueError("Depth-matched comparison mapper is missing or changed")
            check_heldout(original, examples)
            config.update(models=parent["config"]["models"], mapper_origin=parent["config"]["mapper_origin"],
                          mapper_weights_sha256=original["weights_sha256"])
        if (parent["job_name"] != result["job_name"] or parent["region"] != REGION
                or parent["config"]["models"] != config["models"]
                or result["status"] != "replicate" or not result["replication"]
                or correction_gate(result["replication"]) != "replicate"
                or final["weights_sha256"] != result["weights_sha256"]
                or final["comparison_parent_weights_sha256"] != config["mapper_weights_sha256"]
                or manifest["excluded_prior_cohort_sha256"] != [sha(panel/"manifest.json")
                                                               for panel in (DATA, CONFIRMATION)]):
            raise ValueError("Unseen evaluation requires both passing panels and one verified frozen checkpoint")
        for panel in (DATA, CONFIRMATION):
            assert_disjoint(load_examples(panel/"evaluation.jsonl", "eval"), examples)
        check_heldout(final, examples)
        config.update(unseen_confirmation=True, corrected_mapper_origin=parent["live_results"]+"mapped/",
                      corrected_mapper_sha256=final["weights_sha256"], parent_replay_job=parent["job_name"],
                      evidence_replay=True, repeats=2, max_runtime_seconds=1500, worker_timeout_seconds=1140)
    if fresh_pair:
        train = FRESH_CALIBRATION/"train.jsonl"
        calibration_manifest = FRESH_CALIBRATION/"manifest.json"
        calibration = json.loads(calibration_manifest.read_text())
        if sha(train) != calibration["training_sha256"] or len(calibration["selected"]) != 32:
            raise ValueError("Fresh source pair requires the original 32 frozen train documents")
        assert_disjoint(load_examples(train, "calibration"), examples)
        extra = {"inputs/pair-calibration.jsonl": train, "inputs/pair-calibration-manifest.json": calibration_manifest}
        files.update(extra)
        config["input_sha256"].update({name: sha(path) for name, path in extra.items()})
        config.update(models=[QWEN_4B, PAIR[1]], fresh_pair=True, mapper_weights_sha256=None, repeats=2,
                      max_runtime_seconds=1500, worker_timeout_seconds=1140)
    if donor_from:
        prior = json.loads((donor_from/"submission.json").read_text())
        outcome = json.loads((donor_from/"result.json").read_text())
        fitted = json.loads((donor_from/"mapper.json").read_text())
        if (prior["job_name"] != outcome["job_name"] or prior["region"] != REGION
                or not prior["config"].get("fresh_pair") or prior["config"]["models"] != [QWEN_4B, PAIR[1]]
                or outcome["decision"] != "test_fixed_router_before_continual_learning"
                or manifest["excluded_prior_cohort_sha256"] != sha(DATA/"manifest.json")):
            raise ValueError("Donor confirmation requires the fixed, successful 4B-to-8B development checkpoint")
        assert_disjoint(load_examples(DATA/"evaluation.jsonl", "eval"), examples)
        check_heldout(fitted, examples)
        config.update(models=prior["config"]["models"], confirmation=True, parent_replay_job=prior["job_name"],
                      mapper_origin=prior["live_results"]+"mapper/", mapper_weights_sha256=fitted["weights_sha256"],
                      repeats=2, max_runtime_seconds=1500, worker_timeout_seconds=1140)
    return config, files


def evidence_decision(summary):
    if summary["native_warm_agreement"] != summary["questions"]:
        return "check_native_cache_parity"
    oracles = summary["oracles"]["preserve_native_correct"]
    base = oracles["with_evidence_and_original_transfer"]["ttft_ms"]
    candidate = oracles["with_evidence_transfer"]["ttft_ms"]
    if (summary["evidence"]["native_correct_unique_to_transfer_and_excerpts"] < 1
            or candidate["mean"] > .95 * base["mean"] or candidate["p95"] > 1.02 * base["p95"]):
        return "no_incremental_cache_transfer_gain"
    return "replicate_query_aware_transfer"


def unseen_correction_decision(summary):
    """Require incremental improvement over both excerpt routes on the unseen panel."""
    if summary["native_warm_agreement"] != summary["questions"]:
        return "check_native_cache_parity"
    oracle = summary["oracles"]["preserve_native_correct"]
    base = oracle["with_evidence_transfer"]["ttft_ms"]
    candidate = oracle["with_evidence_and_corrected_mapper"]["ttft_ms"]
    if (summary["correction"]["native_correct_unique_beyond_evidence"] >= 1
            and candidate["mean"] <= .95 * base["mean"] and candidate["p95"] <= 1.02 * base["p95"]):
        return "unseen_gain_supports_router_test"
    return "unseen_transfer_gain_unconfirmed"


def decision(summary, min_saving):
    if summary["native_warm_agreement"] != summary["questions"]:
        return "check_native_cache_parity"
    oracles = summary["oracles"]["preserve_native_correct"]
    base = oracles["without_translation"]["ttft_ms"]
    transfer = oracles.get("with_tail_repair", oracles["with_translation"])["ttft_ms"]
    if transfer["mean"] > (1-min_saving) * base["mean"]:
        return "investigate_mapper_before_router"
    if transfer["p95"] > base["p95"]:
        return "inspect_tail_latency_before_router"
    return "test_fixed_router_before_continual_learning"


def retry_mumbai(source, output):
    receipt = json.loads((source/"submission.json").read_text())
    old = describe(receipt["job_name"])
    if (receipt["region"] != REGION or not receipt["config"].get("component_selection")
            or old["TrainingJobStatus"] != "Stopped" or old.get("TrainingStartTime")):
        raise ValueError("Regional fallback requires the stopped, unallocated component probe")
    if json.loads(aws("sagemaker", "list-training-jobs", "--name-contains", "sova-kv",
                      region=MUMBAI))["TrainingJobSummaries"]:
        raise ValueError("Reconcile existing Mumbai cache jobs before opening another worker")
    instance = "ml.g6e.2xlarge"
    ledger = reconcile(instance_type=instance)
    pricing = training_prices((instance,), region=MUMBAI)[instance]
    runtime = receipt["config"]["max_runtime_seconds"]
    cap = runtime * pricing["usd_per_hour"]/3600
    if not 0 < pricing["usd_per_hour"] <= 3.50 or ledger["estimated_with_reserve_and_next_job_usd"]+cap > TOTAL_LIMIT_USD:
        raise ValueError("Mumbai worker exceeds the price or project budget limit")
    job = "sova-kv-component-swap-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    prefix = "kv-transfer/" + job
    base = f"s3://{MUMBAI_BUCKET}/{prefix}/"
    request = receipt["request"]
    code = request["InputDataConfig"][0]["DataSource"]["S3DataSource"]
    mapper = request["InputDataConfig"][1]["DataSource"]["S3DataSource"]
    old_code, old_mapper = code["S3Uri"], mapper["S3Uri"]
    code["S3Uri"], mapper["S3Uri"] = base+"input/", base+"mapper/"
    request["TrainingJobName"] = job
    request["ResourceConfig"]["InstanceType"] = instance
    request["AlgorithmSpecification"]["TrainingImage"] = request["AlgorithmSpecification"]["TrainingImage"].replace(REGION, MUMBAI)
    request["Environment"].update(KV_BUCKET=MUMBAI_BUCKET, KV_PREFIX=prefix)
    request["OutputDataConfig"]["S3OutputPath"] = base+"output/"
    output.mkdir(parents=True, exist_ok=False)
    destination = output/"submission.json"
    receipt.update({"status": "prepared", "job_name": job, "region": MUMBAI, "live_results": base+"live/",
                    "regional_retry_of": old["TrainingJobName"], "estimated_max_worker_compute_usd": cap,
                    "budget_ledger": {**ledger, "price": pricing, "proposed_max_runtime_compute_usd": cap,
                                      "estimated_with_reserve_and_next_job_usd": ledger["estimated_with_reserve_and_next_job_usd"]+cap}})
    receipt.pop("job_arn", None)
    try:
        with tempfile.TemporaryDirectory(prefix="sova-component-retry-") as directory:
            bundle = Path(directory)/"bundle.tar.gz"
            aws("s3", "cp", old_code+"bundle.tar.gz", str(bundle), "--only-show-errors")
            if sha(bundle) != receipt["bundle_sha256"]:
                raise ValueError("The relocated worker bundle is not byte-identical")
            aws("s3", "cp", str(bundle), code["S3Uri"]+"bundle.tar.gz", "--only-show-errors", region=MUMBAI)
            aws("s3", "cp", old_mapper, mapper["S3Uri"], "--recursive", "--source-region", REGION,
                "--only-show-errors", region=MUMBAI)
            spec = Path(directory)/"request.json"
            spec.write_text(json.dumps(request))
            destination.write_text(json.dumps(receipt, indent=2)+"\n")
            result = json.loads(aws("sagemaker", "create-training-job", "--cli-input-json", "file://"+str(spec), region=MUMBAI))
            receipt.update(status="submitted", job_arn=result["TrainingJobArn"])
    except Exception as error:
        receipt.update(status="submission_failed", error=getattr(error, "stderr", None) or str(error))
        raise
    finally:
        destination.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({key: receipt[key] for key in ("status", "job_name", "region", "estimated_max_worker_compute_usd")}))


def collect(output):
    receipt = json.loads((output/"submission.json").read_text())
    config = receipt["config"]
    if receipt["status"] != "submitted" or config["mode"] != "paired_replay":
        raise ValueError("Expected an existing inference replay")
    dataset = (UNSEEN if config.get("unseen_confirmation") else
               CONFIRMATION if config.get("confirmation") else DATA)
    if sha(dataset/"evaluation.jsonl") != config["input_sha256"]["inputs/evaluation.jsonl"]:
        raise ValueError("Scoring dataset differs from the submitted file")
    live = receipt["live_results"]
    region = receipt["region"]
    previous = None
    while True:
        state = describe(receipt["job_name"], region=region)
        status = state["TrainingJobStatus"]
        progress = optional_live(live+"progress.json", region=region) if state["SecondaryStatus"] == "Training" else None
        snapshot = {"status": status, "phase": state["SecondaryStatus"], "progress": progress}
        if snapshot != previous:
            print(json.dumps(snapshot), flush=True)
            previous = snapshot
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(30)
    if status != "Completed":
        failure = state.get("FailureReason")
        try:
            failure = aws("s3", "cp", live+"failure.txt", "-", "--only-show-errors", region=region)
        except Exception:
            pass
        (output/"failure.txt").write_text((failure or status)+"\n")
        raise RuntimeError(f"Replay worker {status}: {failure}")
    rows = optional_live(live+"rows.json", region=region)
    saved = optional_live(live+"summary.json", region=region)
    protocol = optional_live(live+"protocol.json", region=region)
    weights_sha256 = config["mapper_weights_sha256"]
    if config.get("fresh_pair"):
        fitted = optional_live(live+"mapper/manifest.json", region=region)
        calibration = optional_live(live+"mapper-calibration.json", region=region)
        expected = json.loads((FRESH_CALIBRATION/"manifest.json").read_text())["selected"]
        if (not fitted or not calibration or fitted["calibration_tokens"] != 16384
                or fitted["calibration_ids"] != [item["id"] for item in expected]
                or [item["id"] for item in calibration] != fitted["calibration_ids"]):
            raise ValueError("Fresh-pair calibration artifacts do not match the submitted protocol")
        weights_sha256 = fitted["weights_sha256"]
        (output/"mapper.json").write_text(json.dumps(fitted, indent=2)+"\n")
        (output/"mapper-calibration.json").write_text(json.dumps(calibration, indent=2)+"\n")
    if (rows is None or saved is None or protocol is None or protocol["weights_sha256"] != weights_sha256
            or protocol.get("corrected_weights_sha256") != config.get("corrected_mapper_sha256")):
        raise ValueError("Missing or mismatched replay evidence")
    examples = load_examples(dataset/"evaluation.jsonl", "eval")
    if config.get("fresh_pair"):
        check_heldout(fitted, examples)
    gold = {item.id: item.answer for item in examples}
    components = config.get("component_selection")
    tail_cases = config.get("tail_repair_cases")
    tail_arms = config.get("tail_arms", tuple(TAIL_REPAIR) if tail_cases else ())
    summary = (summarize_components(rows, gold, components, layer_groups=config.get("layer_groups")) if components
               else summarize(rows, gold, config["repeats"], tail_cases=tail_cases, tail_arms=tail_arms,
                              evidence=config.get("evidence_replay", False),
                              corrected=bool(config.get("corrected_mapper_sha256"))))
    if summary != saved:
        raise ValueError("Worker results differ from locally recomputed paired results")
    if not components:
        if config.get("corrected_mapper_sha256"):
            summary["decision"] = unseen_correction_decision(summary)
        else:
            summary["decision"] = (evidence_decision(summary) if config.get("evidence_replay")
                                   else decision(summary, config["minimum_incremental_mean_saving"]))
    selected = state["ResourceConfig"].get("SelectedInstanceType") or state["ResourceConfig"].get("InstanceType")
    rate = receipt["budget_ledger"]["price"]
    if receipt["request"]["ResourceConfig"].get("InstancePreferences"):
        rate = training_prices((selected,), region=region)[selected]
    summary.update({"job_name": receipt["job_name"], "billable_seconds": state["BillableTimeInSeconds"],
                    "instance_type": selected, "region": region,
                    "estimated_compute_usd": state["BillableTimeInSeconds"] * rate["usd_per_hour"]/3600,
                    "scope": protocol["endpoint"]})
    for name, payload in (("rows", rows), ("protocol", protocol), ("result", summary)):
        (output/(name+".json")).write_text(json.dumps(payload, indent=2)+"\n")
    if config.get("evidence_replay"):
        selection = optional_live(live+"evidence-selection.json", region=region)
        if not selection or set(selection) != set(gold):
            raise ValueError("Evidence selection audit is incomplete")
        (output/"evidence-selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    if tail_cases:
        traces = optional_live(live+"attention-readout.json", region=region)
        if not traces or {row["id"] for row in traces} != set(tail_cases) or any(len(row["layers"]) != 36 for row in traces):
            raise ValueError("Incomplete 8B layer traces for the six failures")
        (output/"attention-readout.json").write_text(json.dumps(traces, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-source", type=Path, help="Select six rescue cases from this completed replay")
    parser.add_argument("--tail-repair", action="store_true", help="Replay native-first-token and native-tail repair on all 32 records")
    parser.add_argument("--confirm-tail", type=Path, help="Freeze the 31-token variant for a disjoint second panel")
    parser.add_argument("--confirm-readout", type=Path, help="Run a passing fixed correction on the sealed third panel")
    parser.add_argument("--source-4b", action="store_true", help="Fit and screen the depth-matched Qwen3-4B to 8B pair")
    parser.add_argument("--confirm-4b", type=Path, help="Replicate a passing 4B-to-8B mapper without refitting")
    parser.add_argument("--confirm-evidence", type=Path, help="Replicate the same evidence policy on the separate 20-document panel")
    parser.add_argument("--concurrent-control", action="store_true", help="Reserve a scoped control alongside one active worker")
    parser.add_argument("--evidence", action="store_true", help="Replay native question-aware excerpts with and without transfer")
    parser.add_argument("--causal-layers", action="store_true", help="Restore paired native K/V state at fixed layer groups")
    parser.add_argument("--instance", choices=(INSTANCE, "ml.g6e.2xlarge", "ml.g6e.xlarge"), default=INSTANCE)
    parser.add_argument("--flexible-l40s", action="store_true", help="Use one worker across the approved single-L40S preferences")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--collect", action="store_true")
    actions.add_argument("--retry-mumbai", type=Path, help="Relocate a stopped, unallocated Ohio component probe")
    args = parser.parse_args()
    if args.concurrent_control and not (args.submit and (args.evidence or args.source_4b)):
        parser.error("--concurrent-control is scoped to submitting evidence or the depth-matched donor control")
    if args.collect:
        collect(args.output)
    elif args.retry_mumbai:
        retry_mumbai(args.retry_mumbai, args.output)
    else:
        if (args.tail_repair and not args.component_source) or (args.confirm_tail and args.component_source):
            parser.error("Use one of --component-source with --tail-repair, or --confirm-tail")
        protocols = (args.component_source, args.confirm_tail, args.confirm_readout, args.evidence,
                     args.confirm_evidence, args.source_4b, args.confirm_4b)
        if sum(bool(value) for value in protocols) > 1:
            parser.error("Select one paired replay input protocol")
        if args.flexible_l40s and not any(protocols):
            parser.error("--flexible-l40s requires a scoped diagnostic")
        if args.causal_layers and (not args.component_source or args.tail_repair):
            parser.error("--causal-layers requires the component replay")
        config, files = prepare(args.component_source, tail_repair=args.tail_repair, confirm=args.confirm_tail,
                                causal_layers=args.causal_layers, evidence=args.evidence, corrected_from=args.confirm_readout,
                                evidence_from=args.confirm_evidence, fresh_pair=args.source_4b, donor_from=args.confirm_4b)
        scoped = (args.causal_layers or args.evidence or args.confirm_readout or args.confirm_evidence
                  or args.source_4b or args.confirm_4b)
        preferences = (L40S_PREFERENCES[:-1] if scoped else L40S_PREFERENCES) if args.flexible_l40s else ()
        if args.source_4b and args.concurrent_control:
            preferences = tuple(item for item in preferences if item != "ml.g6e.2xlarge")
        instance = INSTANCE if scoped and preferences else L40S_PREFERENCES[-1] if preferences else args.instance
        job_kind = ("causal-layers" if args.causal_layers else "tail-confirm" if args.confirm_tail else "tail-repair" if args.tail_repair
                    else "component-swap" if args.component_source else "evidence-replay" if args.evidence
                    else "readout-unseen" if args.confirm_readout else "evidence-confirm" if args.confirm_evidence
                    else "4b-confirm" if args.confirm_4b else "4b-8b" if args.source_4b else "paired-replay")
        dispatch(config, files, instance=instance, output=args.output, submit=args.submit,
                 job_kind=job_kind,
                 mapper_input=not args.source_4b, preferences=preferences, allow_active_workers=int(args.concurrent_control))


if __name__ == "__main__":
    main()
