"""Freeze full QASPER papers and annotated extractive answers for the pilot."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freeze_paired_replay import PromptTokenizer
from kvverify.continuation_data import ExtractiveQA
from kvverify.hf_backend import DenseRuntime
from kvverify.qa_training import split_document_prompt

ARCHIVE_SHA = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
SOURCE = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
MIN_TOKENS, MAX_TOKENS = 4096, 10240
SEED = 8127


def render(paper):
    sections = [paper["title"], paper["abstract"]]
    for part in paper["full_text"]:
        sections.extend([part["section_name"] or "", *part["paragraphs"]])
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def questions(paper, text, tokenizer, reliable_min):
    # Use the same source text for all questions about a paper. Store the exact
    # character offset solely for auditing. The prompt never gets the span.
    result = []
    compact = " ".join(text.split())
    for row in paper["qas"]:
        if not row["question"].strip():
            continue
        spans = {}
        for annotation in row["answers"]:
            ans = annotation["answer"]
            raw_spans = ans["extractive_spans"]
            if ans["unanswerable"] or len(raw_spans) != 1:
                continue
            span = " ".join(raw_spans[0].split())
            if not 3 <= len(tokenizer.encode(span, add_special_tokens=False)) <= 18:
                continue
            for evidence in ans["evidence"]:
                paragraph = " ".join(evidence.split())
                offset = compact.find(paragraph) if paragraph else -1
                if offset >= reliable_min and span in paragraph:
                    spans[span] = offset
                    break
        if spans:
            # Retain compatible aliases, as independently recorded by the dataset.
            result.append((row, tuple(sorted(spans)), min(spans.values())))
    return result


def freeze(archive, tokenizer, output):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Unexpected QASPER source archive bytes")
    output.mkdir(parents=True, exist_ok=True)
    runtime = PromptTokenizer(tokenizer)
    manifest = {"source_url": SOURCE, "source_archive_sha256": ARCHIVE_SHA, "license": "CC BY 4.0",
                "tokenizer_fingerprint": DenseRuntime.tokenizer_fingerprint(tokenizer), "seed": SEED,
                "min_prompt_tokens": MIN_TOKENS, "max_prompt_tokens": MAX_TOKENS, "files": {}}
    seen_hashes = set()
    with tarfile.open(archive, "r:gz") as tar:
        train = json.load(tar.extractfile("qasper-train-v0.3.json"))
        dev = json.load(tar.extractfile("qasper-dev-v0.3.json"))
    if set(train) & set(dev):
        raise ValueError("The published splits share paper IDs")
    for split, corpus in (("dev", dev), ("train", train)):
        candidates, excluded = [], Counter()
        for paper_id, paper in sorted(corpus.items()):
            context = render(paper)
            if len(context) < MIN_TOKENS:
                continue
            encoded = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            if not MIN_TOKENS <= len(offsets) <= MAX_TOKENS:
                excluded["context_length"] += 1
                continue
            cutoff = len(" ".join(context[:offsets[512][0]].split()))
            rows = []
            for row, answers, offset in questions(paper, context, tokenizer, cutoff):
                identifier = "qasper:" + paper_id
                item = ExtractiveQA("qasper:" + row["question_id"], identifier, context,
                                    row["question"], answers, offset)
                full, _, _ = split_document_prompt(runtime, item)
                if len(full) <= MAX_TOKENS:
                    rows.append({"id": item.id, "question": item.question, "answers": answers,
                                 "evidence_offset": offset, "prompt_tokens": len(full)})
            if not rows or item.context_hash in seen_hashes:
                excluded["no_supported_span_or_duplicate"] += 1
                continue
            seen_hashes.add(item.context_hash)
            # Use short annotated answers first to bound teacher generation cost.
            rows.sort(key=lambda row: (min(len(a) for a in row["answers"]), row["id"]))
            candidates.append({"document_id": identifier, "context": context,
                               "questions": rows[:1 if split == "dev" else 3]})
        random.Random(SEED).shuffle(candidates)
        selected = candidates[:12 if split == "dev" else 170]
        if len(selected) < (12 if split == "dev" else 80):
            raise ValueError(f"Only {len(selected)} {split} papers meet the natural-context criteria")
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False)+"\n")
        lengths = [qa["prompt_tokens"] for paper in selected for qa in paper["questions"]]
        manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "papers": len(selected), "questions": len(lengths),
                                       "eligible_papers": len(candidates),
                                       "prompt_tokens": [min(lengths), max(lengths)],
                                       "excluded": dict(excluded)}
    with (output/"manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest["files"]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(freeze(args.archive, tokenizer, args.output), indent=2))
