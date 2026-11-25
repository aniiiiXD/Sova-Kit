"""Strict records, document-disjoint splits and reproducible local inputs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


@dataclass(frozen=True)
class Example:
    id: str
    document_id: str
    context: str
    question: str
    choices: dict
    answer: str
    split: str

    @classmethod
    def parse(cls, row):
        for key in ("id", "document_id", "context", "question", "answer", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or invalid {key}")
        choices = row.get("choices")
        if not isinstance(choices, dict) or not 2 <= len(choices) <= 26:
            raise ValueError("choices must contain 2-26 labeled options")
        if any(not isinstance(k, str) or len(k) != 1 or k not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or not isinstance(v, str) or not v for k, v in choices.items()):
            raise ValueError("Choice keys must be uppercase single letters with nonempty text")
        if row["answer"] not in choices or row["split"] not in ("calibration", "eval"):
            raise ValueError("Invalid answer or split")
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})

    @property
    def context_hash(self):
        # Normalization catches whitespace-only copies; no near-duplicate claim.
        return digest(" ".join(self.context.split()))

    def prompt(self):
        options = "\n".join(f"{k}. {v}" for k, v in sorted(self.choices.items()))
        return (f"Document:\n{self.context}\n\nQuestion: {self.question}\n{options}\n"
                "Answer with exactly one option letter and no other text.\nAnswer:")


def load_examples(path, split):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = Example.parse(json.loads(line))
                if row.split != split:
                    raise ValueError(f"{row.id}: expected split {split}")
                rows.append(row)
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty with unique example IDs")
    documents, contexts = {}, {}
    for row in rows:
        if row.document_id in documents and documents[row.document_id] != row.context_hash:
            raise ValueError("One document_id must refer to one exact context")
        if row.context_hash in contexts and contexts[row.context_hash] != row.document_id:
            raise ValueError("Repeated exact contexts must share a document_id for clustered evaluation")
        documents[row.document_id] = row.context_hash
        contexts[row.context_hash] = row.document_id
    return rows


def assert_disjoint(calibration, evaluation):
    for key in ("id", "document_id", "context_hash"):
        overlap = {getattr(r, key) for r in calibration} & {getattr(r, key) for r in evaluation}
        if overlap:
            raise ValueError(f"Calibration/evaluation overlap in {key}: {len(overlap)}")


def check_heldout(mapper_manifest, evaluation):
    for field, key in (("calibration_ids", "id"), ("calibration_documents", "document_id"), ("calibration_context_hashes", "context_hash")):
        if field not in mapper_manifest:
            raise ValueError(f"Mapper lacks leakage registry: {field}")
        if set(mapper_manifest[field]) & {getattr(r, key) for r in evaluation}:
            raise ValueError(f"Evaluation leaks into mapper {field}")


def validate_geometry(config):
    if config.get("model_type") != "qwen3":
        raise ValueError("Only dense full-attention Qwen3 is supported; hybrid/MoE states are not implemented")
    if config.get("use_sliding_window") or config.get("rope_scaling"):
        raise ValueError("Sliding attention and scaled/dynamic RoPE are not supported")
    if any(t != "full_attention" for t in config.get("layer_types", [])):
        raise ValueError("Non-full-attention layer detected")
    heads, dim, layers = (config.get(k) for k in ("num_key_value_heads", "head_dim", "num_hidden_layers"))
    if not all(isinstance(x, int) and x > 0 for x in (heads, dim, layers)) or dim % 2:
        raise ValueError("Invalid KV geometry")
    return heads, dim, layers


def validate_pair(source, target):
    s, t = validate_geometry(source), validate_geometry(target)
    if s[:2] != t[:2]:
        raise ValueError("Source and target must have matched KV heads/head dimension")
    return s, t


def checkpoint_identity(directory):
    """Hash local checkpoint bytes once, before the timed region. No remote code."""
    root = Path(directory).resolve()
    weights = sorted(root.glob("*.safetensors"))
    if not (root / "config.json").is_file() or not weights:
        raise ValueError("Use a local safetensors checkpoint containing config.json")
    files = [root / "config.json", *weights]
    for name in ("tokenizer.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"):
        if (root / name).is_file():
            files.append(root / name)
    manifest = {}
    for path in files:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        manifest[path.name] = h.hexdigest()
    return {"directory": str(root), "files": manifest, "sha256": json_digest(manifest)}
