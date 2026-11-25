"""Build a reproducible draft-fact diagnostic from public Harvey LAB contracts."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

REVISION = '7f9b1ee19f9224576a5a5143c4e6f75842ac7020'
BASE = f'https://raw.githubusercontent.com/harveyai/harvey-labs/{REVISION}/'
TASK = 'tasks/contracts/commercial-vendor-customer/master-services-agreement-counterparty-paper-review'
WORD = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

DRAFTS = {
    2: ('pinnacle-draft-msa.docx', [
        ('C-001', 'How many calendar days does the draft allow to pay undisputed invoices?', 'fifteen (15) calendar days', '15 days', ['10 days', '30 days', '45 days']),
        ('C-004', 'What annual escalation does the draft specify for fixed fees starting in month 13?', 'four and one-half percent (4.5%)', '4.5%', ['2.5%', '3.5%', '5.0%']),
        (None, 'How long must the CRO retain study records after completion or termination, subject to applicable law?', 'not less than fifteen (15) years', 'At least 15 years', ['At least 5 years', 'At least 10 years', 'At least 20 years']),
    ]),
    3: ('axionex-msa-draft-v1.docx', [
        ('C-001', 'How many days from invoice date does the draft give Verdantis to pay?', 'due and payable Net fifteen (15) days', '15 days', ['10 days', '30 days', '45 days']),
        ('C-003', 'By what percentage do fees increase on each anniversary under the draft?', 'shall increase by five percent (5%)', '5%', ['2%', '3%', '7.5%']),
        ('C-009', 'What share of remaining fees is charged for early termination under the draft, outside the stated uncured-breach exception?', 'early termination fee equal to seventy-five percent (75%)', '75%', ['25%', '50%', '100%']),
    ]),
    4: ('proposed-gg-pa.docx', [
        ('C001', 'How long is the primary term under the proposed agreement?', 'fifteen (15) years from the Effective Date', '15 years', ['5 years', '10 years', '12 years']),
        ('C004', 'What percentage of NGL net proceeds does the proposed agreement allocate to the Producer?', 'receive seventy-two percent (72%) of Net Proceeds attributable to all NGL Products', '72%', ['70%', '75%', '78%']),
        ('C007', 'What average daily raw gas volume does the draft minimum volume commitment require?', 'forty-five thousand (45,000) MMBtu per day', '45,000 MMBtu', ['25,000 MMBtu', '35,000 MMBtu', '55,000 MMBtu']),
    ]),
    6: ('triton-draft-subcontract-agreement.docx', [
        ('C001', 'How many days after receipt of a complete invoice does draft Section 4.2 allow for payment?', 'due and payable Net forty-five (45) days', '45 days', ['15 days', '30 days', '60 days']),
        ('C004', 'What flat percentage of each progress payment does draft Section 5.1 withhold as retainage?', 'shall retain three percent (3%)', '3%', ['2%', '5%', '10%']),
        ('C009', 'What share of the Subcontract Price is the aggregate liability cap in draft Section 8.1?', 'shall not exceed fifty percent (50%) of the Subcontract Price', '50%', ['25%', '75%', '100%']),
    ]),
}


def fetch(path):
    with urlopen(BASE + path, timeout=25) as response:
        return response.read()


def draft_paragraphs(payload):
    with ZipFile(BytesIO(payload)) as archive:
        document = ElementTree.fromstring(archive.read('word/document.xml'))
    return [line for paragraph in document.iter(WORD + 'p')
            if (line := ''.join(node.text or '' for node in paragraph.iter(WORD + 't')).strip())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    license_bytes = fetch('LICENSE')
    rows, sources = [], []
    for number, (name, questions) in DRAFTS.items():
        scenario = f'scenario-{number:02d}'
        folder = f'{TASK}/{scenario}'
        doc_path = f'{folder}/documents/{name}'
        document = fetch(doc_path)
        paragraphs = draft_paragraphs(document)
        context = name + '\n\n' + '\n'.join(paragraphs)
        sources.append({'scenario': scenario, 'url': BASE + doc_path,
                        'sha256': hashlib.sha256(document).hexdigest()})
        for criterion, question, evidence, correct, distractors in questions:
            evidence_paragraph = next(line for line in paragraphs if evidence in line)
            # Latin rotation balances gold letters and never uses model output.
            position = len(rows) % 4
            options = list(distractors)
            options.insert(position, correct)
            rows.append({'id': f'harvey-msa-{number:02d}-{len(rows)%3+1}',
                         'document_id': f'harvey-msa-{number:02d}', 'context': context,
                         'question': question, 'choices': dict(zip('ABCD', options)),
                         'answer': 'ABCD'[position], 'split': 'eval',
                         'related_public_criterion': criterion,
                         'task_url': BASE + folder + '/task.json', 'evidence': evidence_paragraph})
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'pilot.jsonl').write_bytes(payload)
    (args.output / 'LICENSE.harvey.txt').write_bytes(license_bytes)
    manifest = {'benchmark_catalog': 'https://benchmarkrouter.newmeasure.ai/benchmarks/harvey_lab',
                'upstream_revision': REVISION, 'documents': len(sources), 'questions': len(rows),
                'pilot_sha256': hashlib.sha256(payload).hexdigest(), 'source_documents': sources,
                'scope': 'Draft-fact MCQ adaptation. Full main-draft DOCX body text in order, no companion artifacts, layout or official agent scoring.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Prepared {len(rows)} evidence-checked questions across {len(sources)} draft contracts.")


if __name__ == '__main__':
    main()
