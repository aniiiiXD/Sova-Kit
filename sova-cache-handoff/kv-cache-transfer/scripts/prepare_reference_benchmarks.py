"""Freeze official public QuALITY train/dev and HellaSwag validation only.

No model loading, LLM call, GPU use, official-test read or dependency install.
Requires an existing pyarrow installation to decode the pinned HellaSwag file.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
QUALITY_REVISION = 'f84977c40dbfef70c9cab48037b7becfc8e45f73'
HELLASWAG_REVISION = '218ec52e09a7e7462a5400043bb9a69a41d06b76'
HARNESS_REVISION = 'd6de81643928d653435c431bae19945d41d32520'
QUALITY_HASHES = {
    'train': '4011e9952d5395beb8ff7637b963481a400630c1bbe2f40dc0d83f5d59f926ed',
    'dev': '99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={'User-Agent': 'SOVA-public-reference-data/1.0'})
    with urlopen(request, timeout=60) as response, path.open('wb') as stream:
        while chunk := response.read(1024*1024):
            stream.write(chunk)
    checksum = sha(path)
    if expected and checksum != expected:
        raise ValueError('Pinned upstream bytes changed: '+url)
    return {'url': url, 'sha256': checksum, 'bytes': path.stat().st_size}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            # Some articles contain literal Unicode line-separator characters.
            # Escape them in JSON so downstream str.splitlines() stays safe;
            # decoding still reproduces the exact source context string.
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'examples': len(rows)}


def context_hash(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def quality_rows(raw, split):
    articles, flattened = {}, []
    ids = set()
    for source in raw:
        article = source['article_id']
        text = source['article']
        if article in articles and articles[article] != text:
            raise ValueError('Two writer records disagree on an article text')
        articles[article] = text
        for q in source['questions']:
            question_id = str(q['question_unique_id'])
            if question_id in ids or len(q['options']) != 4 or not 1 <= q['gold_label'] <= 4:
                raise ValueError('Duplicate question ID or invalid QuALITY options/label')
            ids.add(question_id)
            flattened.append({'id': 'quality:'+split+':'+question_id,
                'document_id': 'quality:'+article, 'context': text, 'question': q['question'],
                'choices': {chr(65+i): choice for i, choice in enumerate(q['options'])},
                'answer': chr(64+q['gold_label']),
                'split': 'calibration' if split == 'train' else 'eval',
                'official_split': split, 'source_dataset': 'QuALITY-v1.0.1-htmlstripped',
                'article_id': article, 'question_unique_id': question_id,
                'set_unique_id': source['set_unique_id'], 'difficult': int(q['difficult']),
                'article_url': source.get('url'), 'article_license': source.get('license'),
                'context_sha256_normalized': context_hash(text)})
    return flattened, articles


def select_quality(rows, seed=19, document_count=100):
    groups = defaultdict(list)
    for row in rows:
        groups[row['document_id']].append(row)
    if len(groups) < document_count:
        raise ValueError('Not enough distinct development articles')
    rng = random.Random(seed)
    documents = sorted(groups)
    rng.shuffle(documents)
    result = [rng.choice(sorted(groups[d], key=lambda q:q['question_unique_id']))
              for d in documents[:document_count]]
    if len({q['document_id'] for q in result}) != document_count:
        raise AssertionError('QuALITY selection must have exactly one question per document')
    return result


def hellaswag_preprocess(text):
    # Exact text operations in the pinned EleutherAI HellaSwag utils.py.
    text = text.strip().replace(' [title]', '. ')
    text = re.sub(r'\[.*?\]', '', text)
    return text.replace('  ', ' ')


def hellaswag_rows(raw):
    output, identifiers = [], set()
    for row_index, row in enumerate(raw):
        # The source's `ind` is not unique in official validation. Preserve it,
        # but key frozen examples by the pinned parquet's immutable row index.
        identifier = f'hellaswag:validation:row-{row_index:05d}'
        if identifier in identifiers or len(row['endings']) != 4 or int(row['label']) not in range(4):
            raise ValueError('Invalid HellaSwag validation record')
        identifiers.add(identifier)
        choices = [hellaswag_preprocess(ending) for ending in row['endings']]
        if any(not choice for choice in choices):
            raise ValueError('Empty HellaSwag continuation')
        output.append({**row, 'benchmark_id': identifier, 'official_row_index': row_index, 'official_split': 'validation',
            'query': hellaswag_preprocess(row['activity_label']+': '+row['ctx_a']+' '+row['ctx_b'].capitalize()),
            'choices': choices, 'gold_index': int(row['label']),
            'completion_char_lengths': [len(choice) for choice in choices],
            'evaluation_protocol': 'lm-eval-hellaswag-zero-shot-continuation-likelihood-v1'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'datasets/reference-benchmarks-v1'))
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq
    final = Path(args.output).resolve()
    if final.exists():
        raise ValueError('Refusing to overwrite an existing frozen reference dataset')
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=final.name+'.preparing-', dir=final.parent))
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': 19, 'python': sys.version, 'pyarrow': pyarrow.__version__,
        'purpose': 'Frozen reference data only; no model scores or same-vendor-item claim',
        'official_test_accessed': False,
        'upstream_revisions': {'quality_github': QUALITY_REVISION,
                              'hellaswag_huggingface': HELLASWAG_REVISION,
                              'lm_evaluation_harness_github': HARNESS_REVISION},
        'sources': {}, 'processed': {}}
    try:
        quality = {}
        for split in ('train','dev'):
            name = 'QuALITY.v1.0.1.htmlstripped.'+split
            path = staging/'raw/quality'/name
            url = f'https://raw.githubusercontent.com/nyu-mll/quality/{QUALITY_REVISION}/data/v1.0.1/{name}'
            manifest['sources'][str(path.relative_to(staging))] = download(url,path,QUALITY_HASHES[split])
            rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
            quality[split] = quality_rows(rows,split)
        if set(quality['train'][1]) & set(quality['dev'][1]):
            raise ValueError('QuALITY training/development article IDs overlap')
        if {context_hash(t) for t in quality['train'][1].values()} & {context_hash(t) for t in quality['dev'][1].values()}:
            raise ValueError('QuALITY normalized contexts overlap')
        for name, rows in [('train',quality['train'][0]),('dev-all',quality['dev'][0]),
                           ('dev-100',select_quality(quality['dev'][0]))]:
            filename = 'processed/quality/'+name+'.jsonl'
            spec = write_jsonl(staging/filename,rows)
            spec.update(documents=len({r['document_id'] for r in rows}),
                        difficult_questions=sum(r['difficult'] for r in rows),
                        label_distribution=dict(Counter(r['answer'] for r in rows)),
                        article_license_values=dict(Counter(str(r['article_license']) for r in rows)))
            manifest['processed'][filename] = spec
        readme = staging/'raw/quality/SOURCE_README.md'
        manifest['sources'][str(readme.relative_to(staging))] = download(
            f'https://raw.githubusercontent.com/nyu-mll/quality/{QUALITY_REVISION}/README.md',readme)
        parquet = staging/'raw/hellaswag/validation.parquet'
        manifest['sources'][str(parquet.relative_to(staging))] = download(
            f'https://huggingface.co/datasets/Rowan/hellaswag/resolve/{HELLASWAG_REVISION}/data/validation-00000-of-00001.parquet',parquet)
        raw_hellaswag = pq.read_table(parquet).to_pylist()
        converted = hellaswag_rows(raw_hellaswag)
        if len(converted) != 10042:
            raise ValueError('Unexpected official HellaSwag validation population')
        manifest['processed']['raw/hellaswag/validation.jsonl'] = write_jsonl(staging/'raw/hellaswag/validation.jsonl',raw_hellaswag)
        rng = random.Random(19)
        selected = rng.sample(converted,1000)
        for name, rows in [('validation-all',converted),('validation-1000',selected)]:
            filename = 'processed/hellaswag/'+name+'.jsonl'
            spec = write_jsonl(staging/filename,rows)
            spec.update(source_groups=len({r['source_id'] for r in rows}),
                        label_distribution=dict(Counter(str(r['gold_index']) for r in rows)))
            manifest['processed'][filename] = spec
        readme = staging/'raw/hellaswag/SOURCE_README.md'
        manifest['sources'][str(readme.relative_to(staging))] = download(
            f'https://huggingface.co/datasets/Rowan/hellaswag/raw/{HELLASWAG_REVISION}/README.md',readme)
        for name in ('hellaswag.yaml','utils.py'):
            path = staging/'raw/protocol'/name
            manifest['sources'][str(path.relative_to(staging))] = download(
                f'https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/{HARNESS_REVISION}/lm_eval/tasks/hellaswag/{name}',path)
        path = staging/'raw/protocol/task.py'
        manifest['sources'][str(path.relative_to(staging))] = download(
            f'https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/{HARNESS_REVISION}/lm_eval/api/task.py',path)
        selection = {'quality_dev_100': [{'id':r['id'],'article_id':r['article_id'],
                                        'question_unique_id':r['question_unique_id'],'difficult':r['difficult']}
                                       for r in select_quality(quality['dev'][0])],
                     'hellaswag_validation_1000': [r['benchmark_id'] for r in selected]}
        (staging/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        manifest['selection_sha256'] = sha(staging/'selection.json')
        manifest['licensing'] = {
            'quality': 'No single repository-wide dataset license was verified. Preserve per-article license values and source URLs; confirm redistribution/use rights before commercial distribution.',
            'hellaswag': 'Pinned author-HF dataset card states MIT and links original GitHub LICENSE. Original GitHub API returned HTTP451 and raw LICENSE HTTP404 in this audit; original license bytes were not verified. This is attribution, not a commercial-use clearance.',
            'harness': 'Pinned author-maintained task sources are stored as protocol reference only; no downloaded source is executed.'}
        manifest['protocol'] = {
            'quality': 'Official htmlstripped train2523QA/150articles; all2086devQA/115articles preserved. Seed19 shuffles sorted article IDs, selects100distinctdevarticles, then samples one question perarticle. Gold labels1-based→A-D; hard flag retained. No exactvendorIDs claim.',
            'hellaswag': 'Official validation only; all10042rows preserved; seed19 random.sample selects1000rows. Preserve ctx,endings,source_id,label. Pinned lm-eval query/choice preprocessing. Score full ending loglikelihood; acc_norm divides by processed ending character count, not token count. Never use A/B/C/D generation as a substitute.',
            'scoring_not_run': True,
            'split_checks': 'QuALITY article IDs and normalized context hashes disjoint between train/dev; all question and benchmark IDs unique; all option labels in range.',
            'test_policy': 'No official test file was opened, downloaded, hashed, copied, normalized or scored.'}
        manifest['preparation_script_sha256'] = sha(Path(__file__))
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        (staging/'README.md').write_text(
            '# Frozen reference benchmarks v1\n\n'
            'Prepared data only; no models were trained or evaluated. See manifest.json for pinned sources, SHA256, exact counts, split checks, licensing caveats and scoring definitions.\n\n'
            '- QuALITY: all official train and dev preserved; dev-100 has one question from each of100distinct devarticles, seed19. This is an independent frozen slice, not the vendor\'s unpublished exact item set.\n'
            '- HellaSwag: official validation only, with all rows and a seed19sample of1000; continuation likelihood scoring rather than lettergeneration.\n'
            '- selection.json freezes sampleIDs. raw/protocol stores pinned source reference; it was not executed.\n'
            '- Official test files were not read. Licensing is documented rather than assumed; review manifest before redistribution.\n')
        staging.rename(final)
        print(json.dumps({'status':'prepared','output':str(final),'processed':manifest['processed'],
                          'selection_sha256':manifest['selection_sha256'],'official_test_accessed':False},indent=2))
    except BaseException:
        print('Preparation incomplete; partial artifacts preserved at '+str(staging),file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
