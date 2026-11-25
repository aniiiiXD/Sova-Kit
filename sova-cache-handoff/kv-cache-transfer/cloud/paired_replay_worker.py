"""Run an inference-only paired cache replay on one GPU."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.probe_worker import OUT, notice, upload, write
from kvverify.paired_replay import (ARMS, COMPONENTS, EVIDENCE_ARMS, TAIL_REPAIR, benchmark_question,
                                    probe_components, summarize, summarize_components, trace_readout)


def main():
    import torch
    from huggingface_hub import snapshot_download

    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.data import check_heldout, load_examples
    from kvverify.hf_backend import DenseRuntime
    from kvverify.evidence_replay import prepare_evidence_replay
    from kvverify.qa_training import split_document_prompt

    cfg = json.loads(Path(os.environ["KV_CONFIG"]).read_text())
    if cfg["mode"] != "paired_replay" or not torch.cuda.is_bf16_supported():
        raise ValueError("Paired replay requires BF16 CUDA")
    for name, expected in cfg["input_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input changed: " + name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((ROOT/"inputs/manifest.json").read_text())
    examples = load_examples(ROOT/"inputs/evaluation.jsonl", "eval")
    if len(examples) != cfg["questions"] or len({item.document_id for item in examples}) != len(examples):
        raise ValueError("Evaluation denominator changed")
    components = cfg.get("component_selection")
    layer_groups = cfg.get("layer_groups")
    tail_cases = cfg.get("tail_repair_cases")
    tail_arms = cfg.get("tail_arms", tuple(TAIL_REPAIR) if tail_cases else ())
    if components:
        examples = [item for item in examples if item.id in components]
        if {item.id for item in examples} != set(components):
            raise ValueError("Component questions differ from the parent replay")
    paths = []
    for model in cfg["models"]:
        notice({"phase": "downloading_model", "model": model["repo"]})
        paths.append(snapshot_download(model["repo"], revision=model["revision"], token=False,
                     allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], max_workers=4))
    notice({"phase": "loading_models"})
    runtime = DenseRuntime.from_local(*paths, dtype="bfloat16")
    for model in (runtime.source, runtime.target):
        model.requires_grad_(False)
    if runtime.tokenizer_sha256 != manifest["tokenizer_fingerprint"]:
        raise ValueError("Tokenizer differs from the frozen cohort")
    if cfg.get("fresh_pair"):
        mapper = fit_fresh_pair(runtime, examples)
    else:
        mapper = AffineCacheMapper.load(cfg["input_mapper_dir"], runtime.identities["source"]["sha256"],
                                       runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
        if mapper.manifest["weights_sha256"] != cfg["mapper_weights_sha256"]:
            raise ValueError("Archived adapter does not match the frozen hash")
    check_heldout(mapper.manifest, examples)
    mapper = mapper.to("cuda:0").requires_grad_(False)
    corrected = None
    if cfg.get("corrected_mapper_sha256"):
        corrected = AffineCacheMapper.load("/opt/ml/input/data/corrected", runtime.identities["source"]["sha256"],
                                           runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
        if (corrected.manifest["weights_sha256"] != cfg["corrected_mapper_sha256"]
                or corrected.manifest["comparison_parent_weights_sha256"] != mapper.manifest["weights_sha256"]):
            raise ValueError("Corrected checkpoint does not match the frozen replay")
        check_heldout(corrected.manifest, examples)
        corrected.to("cuda:0").requires_grad_(False)
    selection = {row["id"]: row for row in manifest["selected"]}
    prompts = {}
    evidence = {}
    audit = {}
    for item in examples:
        full, prefix, suffix = split_document_prompt(runtime, item)
        frozen = selection[item.id]
        if (item.context_hash != frozen["context_sha256"] or len(full) != frozen["prompt_tokens"]
                or not 256 < len(prefix) < len(full) <= cfg["max_prompt_tokens"]):
            raise ValueError("Prompt differs from the frozen cohort: " + item.id)
        prompts[item.id] = (prefix, suffix)
        if cfg.get("evidence_replay"):
            started = time.perf_counter()
            selected = prepare_evidence_replay(runtime, item, prefix)
            evidence[item.id] = selected
            audit[item.id] = {"source_token_spans": selected.source_spans,
                              "retained_source_tokens": selected.cache_tokens,
                              "augmented_suffix_tokens": len(selected.augmented_suffix),
                              "short_prompt_tokens": len(selected.short_ids),
                              "retrieval_and_tokenization_ms": (time.perf_counter() - started)*1000}
    if evidence:
        write("evidence-selection.json", audit)
    endpoint = ("Gold margin and forward KL after separately restoring native keys and values; offline diagnostic"
                if components else "synchronized prompt-to-constrained-first-option compute, excluding tokenizer, HTTP and network; resident warm caches")
    arms = (*COMPONENTS, *(layer_groups or {})) if components else (*ARMS, *tail_arms, *(EVIDENCE_ARMS if evidence else ()))
    if corrected is not None:
        arms = (*arms, "corrected_transfer")
    if evidence:
        endpoint += "; evidence arms also charged the measured cold lexical selection and prompt construction CPU time"
    write("protocol.json", {"job_mode": "inference-only", "arms": arms,
                            "questions": len(examples),
                            "repeats": cfg["repeats"], "cuda_device": torch.cuda.get_device_name(0),
                            "models": runtime.identities, "weights_sha256": mapper.manifest["weights_sha256"],
                            "corrected_weights_sha256": cfg.get("corrected_mapper_sha256"),
                            "endpoint": endpoint})
    rows = []
    rng = random.Random(cfg["seed"])
    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            phase = "component_probe" if components else "tail_repair" if tail_arms else "paired_replay"
            notice({"phase": phase, "ready": index-1, "total": len(examples)})
            if components:
                rows.append(probe_components(runtime, mapper, example, *prompts[example.id], layer_groups=layer_groups))
            else:
                question_rows = benchmark_question(runtime, mapper, example, *prompts[example.id], cfg["repeats"], rng,
                                                   tail_arms=tail_arms, evidence=evidence.get(example.id),
                                                   corrected_mapper=corrected)
                for row in question_rows:
                    if row["arm"] in EVIDENCE_ARMS:
                        cpu_ms = audit[example.id]["retrieval_and_tokenization_ms"]
                        row["model_ttft_ms"] = row["ttft_ms"]
                        row["retrieval_and_tokenization_ms"] = cpu_ms
                        row["ttft_ms"] += cpu_ms
                rows.extend(question_rows)
            write("rows.json", rows)
        if tail_cases:
            readouts = []
            for example in examples:
                if example.id in tail_cases:
                    notice({"phase": "attention_readout", "ready": len(readouts), "total": len(tail_cases)})
                    readouts.append(trace_readout(runtime, mapper, example, *prompts[example.id]))
                    write("attention-readout.json", readouts)
    gold = {example.id: example.answer for example in examples}
    summary = (summarize_components(rows, gold, components, layer_groups=layer_groups) if components
               else summarize(rows, gold, cfg["repeats"], tail_cases=tail_cases, tail_arms=tail_arms,
                               evidence=bool(evidence), corrected=corrected is not None))
    write("summary.json", summary)
    notice({"phase": "completed", "ready": len(examples), "total": len(examples)})


def fit_fresh_pair(runtime, examples):
    """Build the same affine reference for the different pinned donor width."""
    import torch

    from kvverify.affine_cache import CrossHeadMoments
    from kvverify.data import assert_disjoint, load_examples
    from kvverify.qa_training import split_document_prompt

    if runtime.source_geometry != (8, 128, 36) or runtime.target_geometry != (8, 128, 36):
        raise ValueError("Expected matching 36-layer dense Qwen cache geometry")
    calibration = load_examples(ROOT/"inputs/pair-calibration.jsonl", "calibration")
    frozen = json.loads((ROOT/"inputs/pair-calibration-manifest.json").read_text())
    if runtime.tokenizer_sha256 != frozen["tokenizer_fingerprint"] or len(calibration) != 32:
        raise ValueError("Fresh pair does not match the frozen calibration")
    assert_disjoint(calibration, examples)
    moments = CrossHeadMoments(runtime.target_geometry, list(range(36)), "cuda:0",
                               source_geometry=runtime.source_geometry)
    sample = random.Random(19)
    rows = []
    with torch.inference_mode():
        for index, (example, expected) in enumerate(zip(calibration, frozen["selected"]), 1):
            full, prefix, _ = split_document_prompt(runtime, example)
            if (example.id != expected["id"] or example.context_hash != expected["context_sha256"]
                    or len(full) != expected["prompt_tokens"]):
                raise ValueError("Fresh pair calibration prompt differs")
            prefix = prefix[:8192]
            positions = sorted(sample.sample(range(256, len(prefix)), 512))
            source = runtime.forward(runtime.source, prefix).past_key_values
            target = runtime.forward(runtime.target, prefix).past_key_values
            moments.update(source, target, runtime.source, runtime.target, token_indices=positions)
            del source, target
            rows.append({"id": example.id, "prefix_tokens": len(prefix), "positions": positions})
            if index == 1 or index % 4 == 0 or index == len(calibration):
                write("mapper-calibration.json", rows)
                notice({"phase": "fresh_pair_calibration", "ready": index, "total": len(calibration)})
    fitted = moments.fit(.01, True, {
        "source_sha256": runtime.identities["source"]["sha256"],
        "target_sha256": runtime.identities["target"]["sha256"],
        "tokenizer_sha256": runtime.tokenizer_sha256,
        "calibration_ids": [item.id for item in calibration],
        "calibration_documents": sorted({item.document_id for item in calibration}),
        "calibration_context_hashes": sorted({item.context_hash for item in calibration}),
        "calibration_prefix_limit": 8192,
    })
    fitted.manifest.update(fitted.save(OUT/"mapper"))
    return fitted


if __name__ == "__main__":
    def interrupt(*_):
        raise TimeoutError("SageMaker replay deadline reached")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc())
        upload()
        raise
