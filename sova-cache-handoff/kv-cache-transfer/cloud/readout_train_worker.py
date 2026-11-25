"""Train a frozen-model cache readout mapper and run its paired dev screen."""

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.probe_worker import OUT, STARTED, notice, upload, write
from kvverify.data import assert_disjoint, check_heldout, load_examples
from kvverify.paired_replay import benchmark_question, summarize


def main():
    import torch
    from huggingface_hub import snapshot_download

    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.activation_offload import selective_saved_tensors_cpu
    from kvverify.answer_mapper import AnswerMapper, sequence_kl_loss
    from kvverify.hf_backend import DenseRuntime
    from kvverify.qa_training import split_document_prompt
    from kvverify.quality import option_token_ids
    from kvverify.readout_mapper import fit_document

    cfg = json.loads(Path(os.environ["KV_CONFIG"]).read_text())
    if cfg["mode"] != "readout_train" or not torch.cuda.is_bf16_supported():
        raise ValueError("Readout training requires BF16 CUDA")
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected
           for name, expected in cfg["input_sha256"].items()):
        raise ValueError("Frozen input hash differs")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    development = load_examples(ROOT/"inputs/evaluation.jsonl", "eval")
    confirmation = load_examples(ROOT/"inputs/replication.jsonl", "eval")
    train = load_examples(ROOT/"inputs/readout-train.jsonl", "calibration")
    if cfg.get("sequence_kl"):
        train = train[:32]
    assert_disjoint(train, development)
    assert_disjoint(train, confirmation)
    assert_disjoint(development, confirmation)
    evaluation = development + confirmation
    manifest = json.loads((ROOT/"inputs/manifest.json").read_text())
    confirm_manifest = json.loads((ROOT/"inputs/replication-manifest.json").read_text())
    frozen_train = json.loads((ROOT/"inputs/readout-train-manifest.json").read_text())
    selection_pool = []
    if cfg.get("layer_selection"):
        pool = load_examples(ROOT/"inputs/layer-selection-pool.jsonl", "calibration")
        baseline_docs = {row.document_id for row in train}
        baseline_contexts = {row.context_hash for row in train}
        selection_pool = [row for row in pool if row.document_id not in baseline_docs
                          and row.context_hash not in baseline_contexts][:16]
        if len(selection_pool) != 16:
            raise ValueError("Need 16 independent source-layer selection articles")
        assert_disjoint(selection_pool, train)
        assert_disjoint(selection_pool, evaluation)
    origins = frozen_train.get("constituents", {})
    if set(origins).intersection(item.document_id for item in evaluation) or set(origins.values()).intersection(
            item.context_hash for item in evaluation):
        raise ValueError("A composed training constituent overlaps the evaluation set")
    student_trajectory = cfg.get("student_trajectory", False)
    if student_trajectory and not origins:
        raise ValueError("Composed histories require source document provenance")
    expected_train = frozen_train["selected"][:32] if cfg.get("sequence_kl") else frozen_train["selected"]
    if [row.id for row in train] != [row["id"] for row in expected_train]:
        raise ValueError("Readout optimization order differs from the freeze")

    paths = []
    for model in cfg["models"]:
        notice({"phase": "downloading_model", "model": model["repo"]})
        paths.append(snapshot_download(model["repo"], revision=model["revision"], token=False,
                     allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], max_workers=4))
    notice({"phase": "loading_models"})
    runtime = DenseRuntime.from_local(*paths, dtype="bfloat16")
    for model in (runtime.source, runtime.target):
        model.requires_grad_(False)
    if any(runtime.tokenizer_sha256 != file["tokenizer_fingerprint"]
           for file in (manifest, confirm_manifest, frozen_train)):
        raise ValueError("Tokenizer differs from a frozen cohort")
    base = AffineCacheMapper.load(cfg["input_mapper_dir"], runtime.identities["source"]["sha256"],
                                 runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
    if base.manifest["weights_sha256"] != cfg["mapper_weights_sha256"]:
        raise ValueError("Unexpected base mapper")
    check_heldout(base.manifest, evaluation)
    base.to("cuda:0").requires_grad_(False)
    layers = cfg["readout_layers"]
    if sorted(set(layers)) != layers or not layers or not 0 <= layers[0] <= layers[-1] < runtime.target_geometry[2]:
        raise ValueError("Invalid trainable layers")

    def prompt(item, frozen, maximum):
        full, prefix, suffix = split_document_prompt(runtime, item)
        if (item.context_hash != frozen["context_sha256"] or len(full) != frozen["prompt_tokens"]
                or not 256 < len(prefix) < len(full) <= maximum):
            raise ValueError("Token boundary differs from the freeze: " + item.id)
        return prefix, suffix

    deadline = STARTED + cfg["worker_timeout_seconds"] - 60

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Readout run reached its collection deadline")

    prompts = {
        item.id: prompt(item, select[item.id], panel["filter"]["max_prompt_tokens"])
        for items, panel in ((development, manifest), (confirmation, confirm_manifest))
        for select in ({row["id"]: row for row in panel["selected"]},)
        for item in items
    }
    training_prompts = [prompt(item, frozen, frozen_train["max_prompt_tokens"])
                        for item, frozen in zip(train, frozen_train["selected"])]
    selection_prompts = []
    if selection_pool:
        frozen_selection = json.loads((ROOT/"inputs/layer-selection-manifest.json").read_text())
        frozen_rows = {row["id"]: row for row in frozen_selection["selected"]}
        selection_prompts = [prompt(item, frozen_rows[item.id], frozen_selection["max_prompt_tokens"])
                             for item in selection_pool]
    write("protocol.json", {
        "models": runtime.identities,
        "gpu": torch.cuda.get_device_name(0),
        "base_weights_sha256": cfg["mapper_weights_sha256"],
        "layers": layers,
        "rank": cfg["readout_rank"],
        "query_samples": cfg["readout_queries"],
        "optimization": ("Backpropagate through frozen 8B on eight query positions with final-option KL"
                         if cfg.get("sequence_kl") else
                         "Closed-form ridge on exact original calibration positions with relative-depth and nearby donors; "
                         "detached native readout selection on 16 excluded QuALITY train articles."
                         if selection_pool else
                         "Closed-form ridge on exact original calibration positions. Change only regularizer scaling."
                         if cfg.get("ridge_scale") else
                         "One update per unique full QuALITY train document, without gold. "
                         "Normalized attention output MSE plus 0.02 attention KL."),
        "trajectory": ("Complete differentiable mapped-cache question forward"
                       if cfg.get("sequence_kl") else None if cfg.get("ridge_scale") or selection_pool else
                       "Detached mapped Q and suffix states, with target output-projection MSE"
                       if student_trajectory else "Detached native Q and suffix states"),
        "training_corpus": frozen_train.get("composition", "single_article"),
        "evaluation": "Fixed final weights. Randomized synchronized BF16 first-option timings on frozen LongBench panels.",
        "training_documents": len(train),
        "replication_gate": "Positive staged development gate only. Both panels are known development benchmarks.",
    })
    metadata = {"comparison_parent_weights_sha256": base.manifest["weights_sha256"]}
    if cfg.get("resume_ridge"):
        corrected = AffineCacheMapper.load("/opt/ml/input/data/corrected", runtime.identities["source"]["sha256"],
                                           runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
        if (corrected.manifest["weights_sha256"] != cfg["resume_ridge"]["weights_sha256"]
                or corrected.manifest["comparison_parent_weights_sha256"] != base.manifest["weights_sha256"]):
            raise ValueError("Recovered ridge checkpoint does not match the original job")
        corrected.to("cuda:0").requires_grad_(False)
        history = json.loads((ROOT/"inputs/archived-ridge-history.json").read_text())
        audit = json.loads((ROOT/"inputs/archived-ridge-audit.json").read_text())
        if len(history) != len(train) or audit["samples"] != corrected.manifest["calibration_tokens"]:
            raise ValueError("Recovered ridge calibration does not match the saved checkpoint")
        write("training-history.json", history)
        write("ridge-scale-audit.json", audit)
        notice({"phase": "loaded_recovered_ridge", "calibration_documents": len(history)})
    elif cfg.get("ridge_scale") or selection_pool:
        corrected, history = fit_ridge_control(runtime, base, train, training_prompts, check_time, metadata,
                                               selection=list(zip(selection_pool, selection_prompts)))
    else:
        mapper = AnswerMapper(base, cfg["readout_rank"])
        optimizer = torch.optim.AdamW((mapper.down, mapper.up), lr=cfg["readout_learning_rate"], weight_decay=0)
        history = []
        scheduled = list(zip(train, training_prompts))
        if cfg.get("sequence_kl"):
            largest = max(range(len(scheduled)), key=lambda i: sum(map(len, scheduled[i][1])))
            scheduled.insert(0, scheduled.pop(largest))
        for i, (item, (prefix, suffix)) in enumerate(scheduled, 1):
            check_time()
            if cfg.get("sequence_kl"):
                optimizer.zero_grad(set_to_none=True)
                with selective_saved_tensors_cpu([runtime.source, runtime.target, base]):
                    loss, metrics = sequence_kl_loss(runtime, mapper, prefix, suffix,
                                                     option_token_ids(runtime.tokenizer, item.choices),
                                                     native_prefix_tokens=256, positions=cfg["readout_queries"])
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_((mapper.down, mapper.up), 1.0, error_if_nonfinite=True)
                optimizer.step()
                del loss
                metrics.update(gradient_norm=float(norm), peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            else:
                metrics = fit_document(runtime, mapper, optimizer, prefix, suffix, layers,
                                       native_head_tokens=256, query_count=cfg["readout_queries"],
                                       student_trajectory=student_trajectory)
            history.append({"step": i, "id": item.id, "document_id": item.document_id, **metrics})
            if i == 1 or i % 4 == 0 or i == len(train):
                write("training-history.json", history)
                notice({"phase": "fitting_joint_readout", "step": i, "total": len(train), "loss": metrics["loss"]})
        metadata.update(
            method=("query_sequence_kl_residual" if cfg.get("sequence_kl") else
                    "mapped_trajectory_readout_residual" if student_trajectory else "joint_key_value_readout_residual"),
            trained_steps=len(history), readout_layers=layers,
            calibration_ids=sorted(set(base.manifest["calibration_ids"]) | {item.id for item in train}),
            calibration_documents=sorted(set(base.manifest["calibration_documents"]) | set(origins)
                                         | {item.document_id for item in train}),
            calibration_context_hashes=sorted(set(base.manifest["calibration_context_hashes"]) | set(origins.values())
                                              | {item.context_hash for item in train}),
        )
        corrected = mapper.folded(metadata)
        del mapper, optimizer
    check_heldout(corrected.manifest, evaluation)
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.get("resume_ridge"):
        exported = OUT/"mapped"
        exported.mkdir()
        for filename in ("manifest.json", "weights.safetensors"):
            shutil.copyfile(Path("/opt/ml/input/data/corrected")/filename, exported/filename)
        saved_mapper = corrected.manifest
    else:
        saved_mapper = corrected.save(OUT/"mapped")

    def evaluate(examples, name):
        rows = []
        rng = random.Random(cfg["seed"])
        with torch.inference_mode():
            for i, item in enumerate(examples, 1):
                check_time()
                rows.extend(benchmark_question(runtime, base, item, *prompts[item.id], cfg["repeats"], rng,
                                               corrected_mapper=corrected))
                if i == 1 or i % 4 == 0 or i == len(examples):
                    write(name + "-rows.json", rows)
                    notice({"phase": name, "ready": i, "total": len(examples)})
        summary = summarize(rows, {item.id: item.answer for item in examples}, cfg["repeats"], corrected=True)
        write(name+"-summary.json", summary)
        return summary

    result = evaluate(development, "development")
    status = correction_gate(result)
    replication = evaluate(confirmation, "replication") if status == "replicate" else None
    write("summary.json", {"status": status, "development": result, "replication": replication,
                           "weights_sha256": saved_mapper["weights_sha256"], "steps": len(history)})
    notice({"phase": "completed", "status": status})



def fit_ridge_control(runtime, base, train, prompts, check_time, metadata, *, selection=()):
    """Verify original sampled fit and then change only the penalty geometry."""
    import torch
    from kvverify.affine_cache import CrossHeadMoments
    from kvverify.layer_alignment import assemble_selected_layers, functional_layer_errors, nearby_layers

    original = base.manifest
    if [item.id for item in train] != original["calibration_ids"]:
        raise ValueError("Scale control requires the original calibration document order")
    supports = (nearby_layers(original["source_layers"], runtime.source_geometry[2])
                if selection else [original["source_layers"]])
    banks = [CrossHeadMoments(runtime.target_geometry, layers, "cuda:0", source_geometry=runtime.source_geometry)
             for layers in supports]
    moments = banks[0]
    rng = random.Random(19)
    history = []
    with torch.inference_mode():
        for i, (item, (prefix, _)) in enumerate(zip(train, prompts), 1):
            check_time()
            prefix = prefix[:original["calibration_prefix_limit"]]
            positions = sorted(rng.sample(range(256, len(prefix)), min(512, len(prefix) - 256)))
            source = runtime.forward(runtime.source, prefix).past_key_values
            target = runtime.forward(runtime.target, prefix).past_key_values
            for bank in banks:
                bank.update(source, target, runtime.source, runtime.target, token_indices=positions)
            del source, target
            history.append({"document": i, "id": item.id, "prefix_tokens": len(prefix), "positions": positions})
            if i == 1 or i % 4 == 0 or i == len(train):
                write("training-history.json", history)
                notice({"phase": "recreating_archived_ridge_moments", "ready": i, "total": len(train)})
        if moments.n != original["calibration_tokens"]:
            raise ValueError("Original ridge sample count changed")
    comparable = {**original, **metadata}
    control = moments.fit(original["alpha"], True, comparable).to("cuda:0")
    errors = {name: float(torch.linalg.vector_norm(new - prior) / torch.linalg.vector_norm(prior))
              for name, new, prior in (("weight", control.weight, base.weight),
                                       ("bias", control.bias, base.bias))}
    centered_variance = moments.xx.diagonal(dim1=-2, dim2=-1) / moments.n - (moments.sx/moments.n).square()
    median_variance = centered_variance.flatten(1).median(-1).values
    write("ridge-scale-audit.json", {
        "original_reproduction_relative_error": errors,
        "samples": moments.n,
        "median_source_feature_variance": dict(zip(("key", "value"), median_variance.tolist())),
        "original_penalty_to_median_variance": dict(zip(("key", "value"),
                                                       (original["alpha"] / median_variance).tolist())),
    })
    if not all(value <= .001 for value in errors.values()):
        raise ValueError("Matched ridge refit drifted from the original checkpoint")
    if not selection:
        calibrated = moments.fit(original["alpha"], True, comparable, relative_regularization=True)
        return calibrated.to("cuda:0"), history

    candidates = [control] + [bank.fit(original["alpha"], True, comparable).to("cuda:0") for bank in banks[1:]]
    del banks, moments
    per_document = []
    for index, (item, (prefix, suffix)) in enumerate(selection, 1):
        check_time()
        per_document.append(functional_layer_errors(runtime, candidates, prefix, suffix))
        if index == 1 or index % 4 == 0 or index == len(selection):
            notice({"phase": "selecting_source_layers", "ready": index, "total": len(selection)})
    errors = torch.stack(per_document).mean(0)
    provenance = {}
    for field, attribute in (("calibration_ids", "id"), ("calibration_documents", "document_id"),
                             ("calibration_context_hashes", "context_hash")):
        provenance[field] = sorted(set(original[field]) | {getattr(item, attribute) for item, _ in selection})
    selected, choices = assemble_selected_layers(candidates, errors, provenance)
    write("layer-selection-audit.json", {"supports": supports, "mean_readout_loss": errors.tolist(),
                                        "source_layers": selected.manifest["source_layers"], "choices": choices,
                                        "validation_ids": [item.id for item, _ in selection]})
    return selected, history


def correction_gate(summary):
    evidence = summary["correction"]
    oracle = summary["oracles"]["preserve_native_correct"]
    current = oracle["with_translation"]["ttft_ms"]
    corrected = oracle["with_corrected_mapper"]["ttft_ms"]
    if (summary["native_warm_agreement"] == summary["questions"] and evidence["rescued_known_failures"] >= 1
            and evidence["recovered_native_correct"] > evidence["old_hybrid_native_correct"]
            and corrected["mean"] <= .95 * current["mean"] and corrected["p95"] <= 1.02 * current["p95"]):
        return "replicate"
    return "rework"


if __name__ == "__main__":
    def interrupt(*_):
        raise TimeoutError("SageMaker worker deadline reached")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc())
        try:
            upload()
        except Exception:
            traceback.print_exc()
        raise
