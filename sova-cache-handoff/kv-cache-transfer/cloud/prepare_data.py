"""Prepare a frozen train-only mapper split and complete development evaluation.

Standard-library data plumbing only: no tokenizer, model, GPU, or network.
Official test documents and labels are never opened or included in outputs.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvverify.data import Example, assert_disjoint, load_examples
from kvverify.quality import SCORING_VERSION

CHOICES = {'A': 'Entailment', 'B': 'Contradiction', 'C': 'NotMentioned'}
LABELS = {value: key for key, value in CHOICES.items()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def examples_for(documents, decisions, record_split, official_split):
    by_id = {str(d['document_id']): d for d in documents}
    if len(by_id) != len(documents):
        raise ValueError('Duplicate document IDs')
    result = []
    for decision in sorted(decisions, key=lambda d: d['decision_id']):
        document = by_id.get(str(decision['document_id']))
        if document is None:
            continue
        if decision['official_split'] != official_split or document['official_split'] != official_split:
            raise ValueError('Official source split mismatch')
        label = decision['gold_label']
        if label not in LABELS:
            raise ValueError('Unknown ContractNLI label')
        result.append(Example(
            decision['decision_id'], 'contractnli:'+str(document['document_id']), document['text'],
            'Classify whether the document entails, contradicts, or does not mention this statement: '+decision['statement'],
            dict(CHOICES), LABELS[label], record_split))
    if set(by_id) != {e.document_id.split(':', 1)[1] for e in result}:
        raise ValueError('A selected document lacks decisions')
    if len({e.id for e in result}) != len(result):
        raise ValueError('Duplicate decisions')
    return result


def prepare(source, output, train_documents=64, validation_documents=16, seed=20261118):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen prepared dataset')
    if train_documents < 1 or validation_documents < 1:
        raise ValueError('Positive train/validation document counts required')
    input_names = ('documents/train_clean.jsonl', 'decisions/train_clean.jsonl',
                   'documents/dev.jsonl', 'decisions/dev.jsonl', 'README.md', 'LICENSE', 'TERMS')
    if not all((source/name).is_file() for name in input_names):
        raise FileNotFoundError('Required train/dev source or attribution file missing')
    train = read_rows(source/'documents/train_clean.jsonl')
    dev = read_rows(source/'documents/dev.jsonl')
    train_decisions = read_rows(source/'decisions/train_clean.jsonl')
    dev_decisions = read_rows(source/'decisions/dev.jsonl')
    if len(train) < train_documents+validation_documents:
        raise ValueError('Not enough train-only documents for disjoint mapper splits')
    ordered = sorted(train, key=lambda d: str(d['document_id']))
    random.Random(seed).shuffle(ordered)
    specs = {
        'train': (ordered[:train_documents], train_decisions, 'calibration', 'train', 'mapper optimization'),
        'mapper_validation': (ordered[train_documents:train_documents+validation_documents],
                              train_decisions, 'calibration', 'train', 'mapper selection/early stopping; not final evaluation'),
        'dev': (dev, dev_decisions, 'eval', 'dev', 'complete official development evaluation'),
    }
    records = {name: examples_for(*spec[:4]) for name, spec in specs.items()}
    for a, b in (('train', 'mapper_validation'), ('train', 'dev'), ('mapper_validation', 'dev')):
        assert_disjoint(records[a], records[b])
    output.mkdir(parents=True, exist_ok=False)
    files, splits = {}, {}
    for name, examples in records.items():
        path = output/(name+'.jsonl')
        path.write_text(''.join(json.dumps(asdict(e), ensure_ascii=False)+'\n' for e in examples))
        load_examples(path, specs[name][2])
        entry = {'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                 'examples': len(examples), 'documents': len({e.document_id for e in examples}),
                 'record_split': specs[name][2], 'official_split': specs[name][3], 'role': specs[name][4],
                 'label_distribution': dict(sorted(Counter(e.answer for e in examples).items())),
                 'document_ids': sorted({e.document_id for e in examples}),
                 'context_hashes': sorted({e.context_hash for e in examples})}
        splits[name] = entry
        files[path.name] = {key: entry[key] for key in ('sha256', 'bytes', 'examples', 'documents', 'role')}
    for name in ('README.md', 'LICENSE', 'TERMS'):
        destination = output/('SOURCE_README.md' if name == 'README.md' else name)
        shutil.copyfile(source/name, destination)
        files[destination.name] = {'sha256': sha(destination), 'bytes': destination.stat().st_size}
    manifest = {
        'schema_version': 1, 'dataset': 'ContractNLI', 'preparation_version': 'mapper-v2-data-1',
        'source_directory': str(source),
        'source_files': {name: {'sha256': sha(source/name), 'bytes': (source/name).stat().st_size} for name in input_names},
        'seed': seed, 'files': files, 'splits': splits, 'choices': CHOICES,
        'attribution': 'Yuta Koreeda and Christopher D. Manning, ContractNLI, Findings of EMNLP 2021.',
        'source_url': 'https://github.com/stanfordnlp/contract-nli',
        'license': 'CC-BY-4.0 per bundled dataset README, LICENSE and TERMS; original document URLs remain in upstream source',
        'modifications': 'Reformatted document-level NLI into QA/options records; deterministic train-only document partition; no text truncation or label alteration.',
        'selection': 'Seeded shuffled train_clean documents without gold-label balancing; every QA for each selected document; all official dev documents/questions.',
        'scoring_version': SCORING_VERSION,
        'evaluation_contract': {
            'decision_primary': 'argmax over next-token logits for single-token A/B/C options; report option mass and conditioned probabilities',
            'generation_separate': 'exact one-letter response contract, invalid format, length truncation and diagnostic prefix scores kept separate',
            'maximum_prompt_tokens': 8192,
            'over_limit': 'Keep all dev examples in the denominator as over_limit/incorrect; never truncate document text for labeled QA',
            'confidence_intervals': 'Paired document-cluster resampling; 17 hypotheses on one contract are not independent documents',
        },
        'test_policy': 'Official test files were not opened, hashed, selected, copied or used for mapper tuning; reserve for a later locked protocol.',
        'leakage_checks': ['example ID', 'document ID', 'normalized exact context hash'],
        'limitations': ['Exact and whitespace duplicate checks do not establish absence of near-duplicates or pretraining contamination.',
                       'ContractNLI has fixed hypotheses; this measures transfer on one legal task, not general generation quality.',
                       'Dev data were used in pilot v1 and are development evidence, not an untouched final test.'],
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'sova-sys1/datasets/processed/contractnli')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/prepared-v1')
    parser.add_argument('--train-documents', type=int, default=64)
    parser.add_argument('--validation-documents', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.train_documents, args.validation_documents, args.seed)
    print(json.dumps({'output': str(args.output.resolve()),
                      'splits': {name: {key: spec[key] for key in ('examples', 'documents', 'sha256')}
                                 for name, spec in manifest['splits'].items()}}, indent=2))


if __name__ == '__main__':
    main()
