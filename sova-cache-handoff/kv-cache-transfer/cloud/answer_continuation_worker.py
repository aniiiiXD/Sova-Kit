"""Verify native teacher continuations, train the mapper, then score fresh papers."""

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

from cloud.probe_worker import OUT, STARTED, notice, upload, write


def main():
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    from kvverify.activation_offload import selective_saved_tensors_cpu
    from kvverify.affine_cache import AffineCacheMapper
    from kvverify.continuation_data import load_papers
    from kvverify.continuation_pilot import MultiDonorMapper, answer_token_loss, decode
    from kvverify.data import assert_disjoint, check_heldout
    from kvverify.hf_backend import DenseRuntime, legacy
    from kvverify.qa_training import split_document_prompt

    cfg = json.loads(Path(os.environ["KV_CONFIG"]).read_text())
    if cfg["mode"] != "answer_continuation" or not torch.cuda.is_bf16_supported():
        raise ValueError("The managed pilot requires BF16 CUDA")
    for name, expected in cfg["input_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input hash differs: " + name)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    training = load_papers(ROOT/"inputs/train.jsonl")
    development = [rows[0] for rows in load_papers(ROOT/"inputs/dev.jsonl").values()]
    pilot_manifest = json.loads((ROOT/"inputs/manifest.json").read_text())
    assert_disjoint([item for group in training.values() for item in group], development)
    if len(development) != pilot_manifest["files"]["dev.jsonl"]["papers"]:
        raise ValueError("Development paper count changed")
    paths = []
    for item in cfg["models"]:
        notice({"phase": "downloading_model", "model": item["repo"]})
        paths.append(snapshot_download(item["repo"], revision=item["revision"], token=False,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], max_workers=4))
    notice({"phase": "loading_models"})
    runtime = DenseRuntime.from_local(*paths, dtype="bfloat16")
    if runtime.tokenizer_sha256 != pilot_manifest["tokenizer_fingerprint"]:
        raise ValueError("The worker tokenizer differs from the dataset freeze")
    for model in (runtime.source, runtime.target):
        model.requires_grad_(False)
    base = AffineCacheMapper.load(cfg["input_mapper_dir"], runtime.identities["source"]["sha256"],
                                  runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
    if base.manifest["weights_sha256"] != cfg["base_weights_sha256"]:
        raise ValueError("The affine initializer changed")
    check_heldout(base.manifest, development)
    base.to("cuda:0").requires_grad_(False)
    count = cfg["training_documents"]
    limit = cfg["max_answer_tokens"]
    deadline = STARTED + cfg["worker_timeout_seconds"] - 35

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Continuation pilot reached its bounded execution window")

    def split(question):
        full, prefix, suffix = split_document_prompt(runtime, question)
        if not 256 < len(prefix) < len(full) <= pilot_manifest["max_prompt_tokens"]:
            raise ValueError("Frozen prompt boundary changed: " + question.id)
        return prefix, suffix

    def view(cache):
        return runtime.DynamicCache.from_legacy_cache(legacy(cache))

    audit, accepted = [], []
    decoded_papers = 0
    for questions in training.values():
        check_time()
        prefix, first_suffix = split(questions[0])
        native = runtime.forward(runtime.target, prefix).past_key_values
        decoded_papers += 1
        for index, item in enumerate(questions):
            question_prefix, suffix = split(item) if index else (prefix, first_suffix)
            if question_prefix != prefix:
                raise ValueError("One paper produced different reusable token prefixes")
            # Only the unlabeled question enters generation. The annotation is
            # consulted after the frozen teacher has produced its answer.
            trace = decode(runtime, runtime.target, suffix, cache=view(native), maximum=limit,
                           store_distributions=True)
            valid = trace.ended and len(trace.tokens) >= 4 and item.correct(trace.text)
            audit.append({"id": item.id, "document_id": item.document_id, "finished": trace.ended, "verified": valid})
            if valid:
                accepted.append((item, prefix, suffix, trace))
                break
        del native
        if decoded_papers % 8 == 0 or len(accepted) == count:
            write("teacher-audit.json", audit)
            notice({"phase": "verify_teacher", "candidate_papers": decoded_papers,
                    "verified_papers": len(accepted), "required": count})
        if len(accepted) == count:
            break
    write("teacher-audit.json", audit)
    teacher_rows = [{
        "id": item.id, "document_id": item.document_id, "context_sha256": item.context_hash,
        "answer": trace.text, "tokens": trace.tokens, "annotated_answers": item.answers,
        "prefix_tokens": len(prefix), "suffix_tokens": len(suffix),
    } for item, prefix, suffix, trace in accepted]
    write("verified-teacher.json", teacher_rows)
    if len(accepted) != count:
        raise ValueError(f"Expected {count} verified teacher papers, obtained {len(accepted)}")
    scores_file = OUT/"teacher-logprobs.safetensors"
    save_file({item.id: trace.log_probs for item, _, _, trace in accepted}, str(scores_file))
    provenance = {"teacher_scores_sha256": hashlib.sha256(scores_file.read_bytes()).hexdigest(),
                  "teacher_rows_sha256": hashlib.sha256((OUT/"verified-teacher.json").read_bytes()).hexdigest(),
                  "selection": "First independently verified native 8B answer per frozen paper",
                  "distribution_dtype": "float16_log_softmax",
                  "first_answer_token_loss_weight": cfg["decision_weight"],
                  "calibration_ids": sorted(set(base.manifest["calibration_ids"]) | {row["id"] for row in teacher_rows}),
                  "calibration_documents": sorted(set(base.manifest["calibration_documents"])
                                                  | {row["document_id"] for row in teacher_rows}),
                  "calibration_context_hashes": sorted(set(base.manifest["calibration_context_hashes"])
                                                       | {row["context_sha256"] for row in teacher_rows}),
                  "models": runtime.identities, "tokenizer_sha256": runtime.tokenizer_sha256}
    check_heldout(provenance, development)
    write("teacher-manifest.json", provenance)
    notice({"phase": "teacher_verified", "verified_papers": count})

    mapper = MultiDonorMapper(base, rank=cfg["rank"], donors=cfg["donors"])
    optimizer = torch.optim.AdamW(mapper.residual_parameters(), lr=cfg["learning_rate"], weight_decay=0.)
    history = []
    longest = max(range(count), key=lambda i: len(accepted[i][1]) + len(accepted[i][2]))
    accepted.insert(0, accepted.pop(longest))
    for step, (item, prefix, suffix, trace) in enumerate(accepted, 1):
        check_time()
        optimizer.zero_grad(set_to_none=True)
        with selective_saved_tensors_cpu((runtime.source, runtime.target, base)):
            loss = answer_token_loss(runtime, mapper, prefix, suffix, trace, decision_weight=cfg["decision_weight"])
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(mapper.residual_parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "id": item.id, "answer_tokens": len(trace.tokens),
                        "loss": float(loss.detach()), "gradient_norm": float(norm)})
        del loss
        if step % 8 == 0 or step == 1:
            write("training-history.json", history)
            notice({"phase": "training", "completed": step, "total": count})
        if step % 16 == 0:
            mapper.save(OUT/f"checkpoint-{step:02d}", {**provenance, "steps": step})
    mapper.eval()
    del optimizer
    results = []
    rng = random.Random(cfg["seed"])
    for item in development:
        check_time()
        prefix, suffix = split(item)
        results.append(evaluate_question(runtime, base, mapper, item, prefix, suffix, rng, limit))
        write("evaluation.json", results)
        notice({"phase": "evaluation", "completed": len(results), "total": len(development)})
    write("summary.json", summarize(results, count))
    notice({"phase": "completed", "trained_papers": count, "evaluation_papers": len(results)})


def evaluate_question(runtime, base, mapper, item, prefix, suffix, rng, limit):
    import torch

    from kvverify.continuation_pilot import decode
    from kvverify.fast_affine import map_affine_fast
    from kvverify.hf_backend import legacy
    from kvverify.native_prefix import replace_prefix

    source = runtime.forward(runtime.source, prefix).past_key_values
    target = runtime.forward(runtime.target, prefix).past_key_values
    routes = ["native_cold", "native_hit", "warm_source", "short_native", "single_donor", "multi_donor"]
    rng.shuffle(routes)
    result = {"id": item.id, "document_id": item.document_id, "prefix_tokens": len(prefix), "routes": {}}

    def first(route):
        if route == "native_cold":
            return runtime.forward(runtime.target, prefix+suffix)
        if route == "native_hit":
            return runtime.forward(runtime.target, suffix, runtime.DynamicCache.from_legacy_cache(legacy(target)))
        if route == "warm_source":
            return runtime.forward(runtime.source, suffix, runtime.DynamicCache.from_legacy_cache(legacy(source)))
        head = runtime.forward(runtime.target, prefix[:256]).past_key_values
        if route == "short_native":
            return runtime.forward(runtime.target, suffix, head)
        mapped = (map_affine_fast(base, source, runtime.source, runtime.target, block_tokens=256)
                  if route == "single_donor" else mapper.apply(source, runtime.source, runtime.target))
        return runtime.forward(runtime.target, suffix, replace_prefix(mapped, head, 256))

    with torch.inference_mode():
        for route in routes:
            initial, ttft = runtime.timed(lambda: first(route))
            model = runtime.source if route == "warm_source" else runtime.target
            trace = decode(runtime, model, [], maximum=limit, first=initial)
            result["routes"][route] = {"answer": trace.text, "ended": trace.ended,
                                       "correct": trace.ended and item.correct(trace.text), "ttft_ms": ttft}
            del initial
    return result


def summarize(rows, count):
    from statistics import median

    required = {"native_cold", "native_hit", "warm_source", "short_native", "single_donor", "multi_donor"}
    if not rows or len({row["document_id"] for row in rows}) != len(rows) or any(set(row["routes"]) != required for row in rows):
        raise ValueError("Incomplete paired replay")
    routes = {route: {"correct": sum(row["routes"][route]["correct"] for row in rows),
                      "median_ttft_ms": median(row["routes"][route]["ttft_ms"] for row in rows)}
              for route in sorted(required)}
    misses = [row["routes"] for row in rows if row["routes"]["native_cold"]["correct"]
              and not any(row["routes"][arm]["correct"] for arm in ("warm_source", "short_native", "single_donor"))]
    return {"trained_papers": count, "evaluation_papers": len(rows), "routes": routes,
            "native_hit_exact_match": sum(row["routes"]["native_cold"]["answer"] == row["routes"]["native_hit"]["answer"]
                                          and row["routes"]["native_cold"]["ended"] == row["routes"]["native_hit"]["ended"]
                                          for row in rows),
            "native_correct_missed_by_cheaper_controls": len(misses),
            "rescued_by_multi_donor": sum(case["multi_donor"]["correct"] for case in misses),
            "timing_scope": "Synchronized first-token compute in one process with models resident; one timing per route/paper."}


if __name__ == "__main__":
    def interrupted(*_):
        raise TimeoutError("Managed worker was interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc())
        upload()
        raise
