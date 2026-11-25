"""Freeze a model-output-blind LongBench v2 subset for one paired replay."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.harvey_pilot_submit import sha
from kvverify.data import Example, digest
from kvverify.hf_backend import DenseRuntime
from kvverify.qa_training import split_document_prompt

REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
SOURCE_SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
CAP = 20000
COUNT = 32
SEED = 1029
REFERENCE = ROOT / "datasets/reference-benchmarks-v1/processed/quality"


class PromptTokenizer:
    tokenize = DenseRuntime.tokenize

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def historical_contexts():
    result = set()
    for path in (REFERENCE / "train.jsonl", REFERENCE / "dev-all.jsonl",
                 ROOT / "datasets/harvey-contract-facts-v1/pilot.jsonl"):
        seen = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["document_id"] not in seen:
                    result.add(digest(" ".join(row["context"].split())))
                    seen.add(row["document_id"])
    return result


def freeze(raw, tokenizer, output, *, prior=None, unseen=False):
    if sha(raw) != SOURCE_SHA:
        raise ValueError("LongBench v2 snapshot drifted")
    records = json.loads(raw.read_text())
    historical = historical_contexts()
    cap, count, seed = CAP, COUNT, SEED
    prior_cohorts = (() if prior is None else (prior,))
    if unseen:
        if prior is not None:
            raise ValueError("Final replay excludes the two recorded development panels")
        prior_cohorts = tuple(ROOT/"datasets"/name/"manifest.json" for name in
                              ("longbench-v2-paired-v1", "longbench-v2-tail-confirmation-v1"))
        cap, count, seed = 32768, None, 1907
    elif prior:
        cap, count, seed = 26500, 20, 2039
    for cohort in prior_cohorts:
        historical.update(row["context_sha256"] for row in json.loads(cohort.read_text())["selected"])
    candidates = {}
    excluded = Counter()
    token_lengths = {}
    for row in sorted(records, key=lambda item: item["_id"]):
        if row["domain"] != "Single-Document QA" or row["length"] != "short":
            continue
        example = Example.parse({"id": "longbench-v2:" + row["_id"],
                                 "document_id": "longbench-v2:" + row["_id"],
                                 "context": row["context"], "question": row["question"],
                                 "choices": {key: row["choice_" + key] for key in "ABCD"},
                                 "answer": row["answer"], "split": "eval"})
        fingerprint = example.context_hash
        if fingerprint in historical:
            excluded["historical_normalized_exact_overlap"] += 1
            continue
        if fingerprint in candidates:
            excluded["repeated_normalized_document"] += 1
            continue
        full, prefix, _ = split_document_prompt(tokenizer, example)
        if len(full) > cap or len(prefix) <= 256:
            excluded["outside_prompt_cap"] += 1
            continue
        candidates[fingerprint] = example
        token_lengths[example.id] = len(full)
    if count is None:
        count = len(candidates)
        if count < 8:
            raise ValueError("Too few distinct remaining documents for an unseen replay")
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} distinct questions fit the frozen protocol")
    picked = random.Random(seed).sample(list(candidates.values()), count)
    output.mkdir(parents=True, exist_ok=True)
    data = output / "evaluation.jsonl"
    with data.open("x") as stream:
        for example in picked:
            stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    manifest = {"dataset": "zai-org/LongBench-v2", "source_revision": REVISION,
                "source_url": f"https://huggingface.co/datasets/zai-org/LongBench-v2/tree/{REVISION}",
                "source_sha256": SOURCE_SHA, "dataset_card_license": "apache-2.0",
                "purpose": "Independent-data screening replay for frozen historical mapper, not a product acceptance set",
                "filter": {"domain": "Single-Document QA", "length": "short", "max_prompt_tokens": cap},
                "short_single_document_rows": sum(row["domain"] == "Single-Document QA" and row["length"] == "short"
                                                  for row in records),
                "historical_overlap_check": "Whitespace-normalized exact document fingerprints against QuALITY train/all dev and Harvey pilot. Training-corpus and near-duplicate exposure unknown.",
                "seed": seed, "candidates": len(candidates), "exclusions": dict(excluded), "questions": count,
                "tokenizer_fingerprint": DenseRuntime.tokenizer_fingerprint(tokenizer.tokenizer),
                "selected": [{"id": item.id, "context_sha256": item.context_hash,
                              "prompt_tokens": token_lengths[item.id]} for item in picked],
                "evaluation_sha256": sha(data)}
    if unseen:
        manifest["excluded_prior_cohort_sha256"] = [sha(cohort) for cohort in prior_cohorts]
        manifest["purpose"] = ("All remaining eligible long documents, frozen before corrected mapper results. "
                               "No model-output filtering. Same source benchmark as development.")
    elif prior:
        manifest["excluded_prior_cohort_sha256"] = sha(prior)
        manifest["purpose"] = "Frozen, document-disjoint second panel for the selected 31-token tail; longer-length validation"
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: manifest[key] for key in ("questions", "candidates", "exclusions", "evaluation_sha256")}))


if __name__ == "__main__":
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--exclude-prior-cohort", type=Path)
    selection.add_argument("--unseen", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False, local_files_only=True)
    freeze(args.raw, PromptTokenizer(tokenizer), args.output, prior=args.exclude_prior_cohort, unseen=args.unseen)
