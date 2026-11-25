"""Frozen extractive QA documents for generated-answer distillation."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from .data import digest


def normalize_answer(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split()).removesuffix(".").strip()


@dataclass(frozen=True)
class ExtractiveQA:
    id: str
    document_id: str
    context: str
    question: str
    answers: tuple
    evidence_offset: int
    choices: tuple = ()

    @property
    def context_hash(self):
        return digest(" ".join(self.context.split()))

    def prompt(self):
        if self.choices:
            options = "\n".join(f"{label}. {text}" for label, text in self.choices)
            return (f"Document:\n{self.context}\n\nQuestion: {self.question}\n{options}\n"
                    "Copy the entire line of the correct option, including its letter and its text. "
                    "Write only that line.\nAnswer:")
        return (f"Document:\n{self.context}\n\nQuestion: {self.question}\n"
                "Copy the shortest complete answer span verbatim from the document. "
                "Write only that span.\nAnswer:")

    def correct(self, text):
        def normalized(value):
            if self.choices:
                match = re.fullmatch(r"(?:Answer:\s*)?([A-Z])\s*[.):\-]\s*(.+)", value.strip(), re.IGNORECASE | re.DOTALL)
                if not match:
                    return None
                return match.group(1).upper(), normalize_answer(match.group(2))
            return normalize_answer(value)

        actual = normalized(text)
        return actual is not None and actual in {normalized(answer) for answer in self.answers}


def load_papers(path):
    papers = {}
    hashes = set()
    question_ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            identifier, context = record["document_id"], record["context"]
            questions = tuple(ExtractiveQA(row["id"], identifier, context, row["question"],
                                           tuple(row["answers"]), row["evidence_offset"],
                                           tuple(sorted(row.get("choices", {}).items())))
                              for row in record["questions"])
            if not questions or not context or identifier in papers or questions[0].context_hash in hashes:
                raise ValueError("Empty or duplicate extractive paper")
            for q in questions:
                if (q.id in question_ids or not q.question.strip() or not q.answers
                        or any(not normalize_answer(a) for a in q.answers)
                        or not 0 <= q.evidence_offset < len(context)):
                    raise ValueError("Invalid extractive question: " + q.id)
                question_ids.add(q.id)
            papers[identifier] = questions
            hashes.add(questions[0].context_hash)
    if not papers:
        raise ValueError("No frozen extractive papers")
    return papers
