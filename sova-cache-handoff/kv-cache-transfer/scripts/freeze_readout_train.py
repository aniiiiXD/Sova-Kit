"""Freeze one QuALITY train question per complete optimization document."""

from collections import defaultdict
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.harvey_pilot_submit import sha
from cloud.harvey_pilot_worker import calibration_examples
from kvverify.data import assert_disjoint, load_examples
from kvverify.hf_backend import DenseRuntime
from kvverify.qa_training import split_document_prompt
from freeze_paired_replay import PromptTokenizer

REFERENCE = ROOT/"datasets/reference-benchmarks-v1"
LIMIT = 14336
COUNT = 64
SEED = 2211


def freeze(tokenizer, output, *, composed=False, archived=False):
    origin = REFERENCE/"processed/quality/train.jsonl"
    source_manifest = json.loads((REFERENCE/"manifest.json").read_text())
    if sha(origin) != source_manifest["processed"]["processed/quality/train.jsonl"]["sha256"]:
        raise ValueError("QuALITY source changed")
    examples = load_examples(origin, "calibration")
    for panel in ("longbench-v2-paired-v1", "longbench-v2-tail-confirmation-v1"):
        assert_disjoint(examples, load_examples(ROOT/"datasets"/panel/"evaluation.jsonl", "eval"))
    groups = defaultdict(list)
    for item in examples:
        groups[item.document_id].append(item)
    rng = random.Random(SEED)
    docs = sorted(groups)
    rng.shuffle(docs)
    runtime = PromptTokenizer(tokenizer)
    expected_count = 32 if archived else COUNT
    pool = calibration_examples(examples, expected_count, 19) if archived else (
        rng.choice(sorted(groups[doc], key=lambda example: example.id)) for doc in docs
    )
    selected = []
    limit = 20480 if composed else LIMIT
    constituent_hashes = {}
    for item in pool:
        doc = item.document_id
        components = [item]
        if composed:
            donor_ids = [identifier for identifier in docs if identifier != doc]
            shuffled = rng.sample(donor_ids, 5)
            components.extend(groups[other][0] for other in shuffled)
            texts = [component.context for component in components]
            lengths = [len(tokenizer.encode(part, add_special_tokens=False)) for part in texts]
            while sum(lengths) > (limit - 512) and len(components) > 2:
                components.pop()
                lengths.pop()
            rng.shuffle(components)
            item = replace(item, context="\n\n---\n\n".join(component.context for component in components))
        prompt, prefix, _ = split_document_prompt(runtime, item)
        if (12288 if composed else 1024) <= len(prefix) < len(prompt) <= limit:
            selected.append((item, len(prompt)))
            constituent_hashes.update({component.document_id: component.context_hash for component in components})
        if len(selected) == expected_count:
            break
    if len(selected) != expected_count:
        raise ValueError("Insufficient eligible full QuALITY training documents")
    output.mkdir(parents=True, exist_ok=False)
    data = output/"train.jsonl"
    with data.open("x") as stream:
        for item, _ in selected:
            stream.write(json.dumps(asdict(item), ensure_ascii=False)+"\n")
    manifest = {
        "purpose": "Complete document histories and native teacher queries. No answer labels used by the optimizer.",
        "source_file": "processed/quality/train.jsonl",
        "source_sha256": sha(origin),
        "training_sha256": sha(data),
        "tokenizer_fingerprint": DenseRuntime.tokenizer_fingerprint(tokenizer),
        "max_prompt_tokens": limit,
        "composition": "One questioned article and unrelated complete QuALITY train texts, shuffled" if composed else "single_article",
        "constituents": constituent_hashes,
        "seed": SEED,
        "selected": [{"id": item.id, "context_sha256": item.context_hash, "prompt_tokens": length}
                     for item, length in selected],
    }
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"documents": len(selected), "prompt_tokens": [min(n for _, n in selected),
                     max(n for _, n in selected)], "training_sha256": manifest["training_sha256"]}))


if __name__ == "__main__":
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--composed", action="store_true")
    mode.add_argument("--archived", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False, local_files_only=True)
    freeze(tokenizer, args.output, composed=args.composed, archived=args.archived)
