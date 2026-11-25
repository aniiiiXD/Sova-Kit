"""Freeze the established QuALITY splits for full-option answer generation."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.harvey_pilot_submit import sha
from freeze_paired_replay import PromptTokenizer, REVISION, SOURCE_SHA
from kvverify.continuation_data import ExtractiveQA
from kvverify.data import load_examples
from kvverify.hf_backend import DenseRuntime
from kvverify.qa_training import split_document_prompt

REF = ROOT/"datasets/reference-benchmarks-v1"
SEED = 8128
MAX_TOKENS = 8900


def freeze(tokenizer, raw, output):
    runtime = PromptTokenizer(tokenizer)
    train_name = "processed/quality/train.jsonl"
    reference = json.loads((REF/"manifest.json").read_text())
    train = REF/train_name
    if sha(train) != reference["processed"][train_name]["sha256"] or sha(raw) != SOURCE_SHA:
        raise ValueError("A pinned data source changed")

    excluded_papers = set()
    for name in ("longbench-v2-paired-v1", "longbench-v2-tail-confirmation-v1", "longbench-v2-readout-unseen-v1"):
        older = json.loads((ROOT/"datasets"/name/"manifest.json").read_text())
        excluded_papers.update(item["context_sha256"] for item in older["selected"])
    groups = defaultdict(list)
    for row in load_examples(train, "calibration"):
        groups[row.document_id].append(row)
    training, blocked = [], Counter()
    for identifier, rows in sorted(groups.items()):
        shortlisted = []
        for row in rows:
            answer = f"{row.answer}. {row.choices[row.answer]}"
            count = len(tokenizer.encode(answer, add_special_tokens=False))
            if 6 <= count <= 16:
                shortlisted.append((count, row.id, row, answer))
        usable = []
        for _, _, row, answer in sorted(shortlisted)[:4]:
            item = ExtractiveQA(row.id, identifier, row.context, row.question, (answer,), 0,
                                tuple(sorted(row.choices.items())))
            full, _, _ = split_document_prompt(runtime, item)
            if len(full) <= MAX_TOKENS:
                usable.append({"id": item.id, "question": item.question, "answers": [answer],
                               "evidence_offset": 0, "choices": row.choices, "prompt_tokens": len(full)})
        if usable:
            training.append({"document_id": identifier, "context": rows[0].context, "questions": usable})
            excluded_papers.add(rows[0].context_hash)
        else:
            blocked["no_multi_token_question_in_bounds"] += 1

    development = []
    domains = {"Multi-Document QA", "Long-dialogue History Understanding", "Long Structured Data Understanding"}
    for row in sorted(json.loads(raw.read_text()), key=lambda item: item["_id"]):
        if row["domain"] not in domains or row["length"] != "short" or len(row["context"]) > 150000:
            continue
        choices = {label: row["choice_"+label] for label in "ABCD"}
        answer = f"{row['answer']}. {choices[row['answer']]}"
        if not 6 <= len(tokenizer.encode(answer, add_special_tokens=False)) <= 19:
            continue
        identifier = "longbench-v2:"+row["_id"]
        item = ExtractiveQA(identifier, identifier, row["context"], row["question"], (answer,), 0,
                            tuple(choices.items()))
        if item.context_hash in excluded_papers:
            continue
        full, _, _ = split_document_prompt(runtime, item)
        if not 256 < len(full) <= 16500:
            continue
        excluded_papers.add(item.context_hash)
        development.append({"document_id": identifier, "context": item.context, "domain": row["domain"],
                            "questions": [{"id": item.id, "question": item.question, "answers": [answer],
                                           "evidence_offset": 0, "choices": choices,
                                           "prompt_tokens": len(full)}]})
    rng = random.Random(SEED)
    rng.shuffle(training)
    rng.shuffle(development)
    development = development[:12]
    supplemental = None
    if len(development) < 12:
        qasper = ROOT/"datasets/qasper-continuation-pilot-v1"
        frozen = json.loads((qasper/"manifest.json").read_text())
        supplemental = qasper/"dev.jsonl"
        if sha(supplemental) != frozen["files"]["dev.jsonl"]["sha256"]:
            raise ValueError("The supplemental extractive evaluation changed")
        for line in supplemental.read_text().splitlines():
            record = json.loads(line)
            # The original QASPER run stopped in teacher filtering, before
            # loading any held-out paper through either model.
            item = record["questions"][0]
            example = ExtractiveQA(item["id"], record["document_id"], record["context"], item["question"],
                                   tuple(item["answers"]), item["evidence_offset"])
            if example.context_hash not in excluded_papers:
                development.append({**record, "domain": "QASPER extractive QA"})
                excluded_papers.add(example.context_hash)
            if len(development) == 12:
                break
    if len(training) < 100 or len(development) != 12:
        raise ValueError(f"Insufficient frozen inputs, {len(training)} training and {len(development)} held-out papers")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": "QuALITY v1.0.1 official train with a separate LongBench v2 evaluation",
                "source_url": "https://github.com/nyu-mll/quality", "training_sha256": sha(train),
                "holdout_source_url": f"https://huggingface.co/datasets/zai-org/LongBench-v2/tree/{REVISION}",
                "holdout_source_sha256": SOURCE_SHA,
                "tokenizer_fingerprint": DenseRuntime.tokenizer_fingerprint(tokenizer),
                "seed": SEED, "max_prompt_tokens": 16500, "training_max_prompt_tokens": MAX_TOKENS,
                "format": "correct option letter and complete official option text",
                "holdout_domains": dict(Counter(row["domain"] for row in development)),
                "supplemental_qasper_sha256": sha(supplemental) if supplemental else None,
                "training_exclusions": dict(blocked), "files": {}}
    for split, selected in (("train", training), ("dev", development)):
        path = output/f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        lengths = [row["prompt_tokens"] for paper in selected for row in paper["questions"]]
        manifest["files"][path.name] = {"sha256": sha(path), "papers": len(selected), "questions": len(lengths),
                                       "prompt_tokens": [min(lengths), max(lengths)]}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest["files"]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(freeze(tokenizer, args.raw, args.output), indent=2))
