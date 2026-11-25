"""Private, single-host cache artifacts shared by vLLM and the mapper."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import time

from .data import json_digest


VALID_DTYPES = {"float16": "F16", "bfloat16": "BF16"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _validate_identifier(value):
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Scope and handoff ID must be 1–128 letters, digits, dots, underscores, or hyphens")


def _validate_digest(value):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CacheKey:
    scope: str
    handoff_id: str
    prefix_sha256: str
    tokens: int
    model_sha256: str
    tokenizer_sha256: str
    mapper_sha256: str = ""

    def __post_init__(self):
        _validate_identifier(self.scope)
        _validate_identifier(self.handoff_id)
        for value in (self.prefix_sha256, self.model_sha256, self.tokenizer_sha256):
            _validate_digest(value)
        if self.mapper_sha256 != "":
            _validate_digest(self.mapper_sha256)
        if type(self.tokens) is not int or self.tokens <= 0:
            raise ValueError("Cache token count must be positive")

    @classmethod
    def from_tokens(cls, scope, handoff_id, tokens, model_sha256, tokenizer_sha256, mapper_sha256=""):
        tokens = tuple(tokens)
        if not tokens or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens):
            raise ValueError("Cache prefixes require unsigned 32-bit token IDs")
        digest = hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()
        return cls(scope, handoff_id, digest, len(tokens), model_sha256, tokenizer_sha256, mapper_sha256)


@dataclass(frozen=True)
class CacheGeometry:
    heads: int
    head_dim: int
    layers: int
    dtype: str

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.heads, self.head_dim, self.layers)):
            raise ValueError("Cache geometry must be positive")
        if self.dtype not in VALID_DTYPES:
            raise ValueError("The connector supports FP16 and BF16 KV caches")

    def tensor_shape(self, tokens):
        return (2, tokens, self.heads, self.head_dim)

    def names(self):
        return tuple(f"layer_{index}" for index in range(self.layers))


class CacheStore:
    """Atomic safetensors files in a directory shared by trusted local processes."""

    def __init__(self, root, *, ttl_seconds=900):
        self.root = Path(root).expanduser().resolve()
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("Cache lifetime must be a positive number of seconds")
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError("Cache directory must be private to the serving user (chmod 700)")

    def path(self, key):
        return self.root / (json_digest(asdict(key)) + ".safetensors")

    @staticmethod
    def _metadata(key, geometry):
        return {"format": "sova-kv-v1", "key": json.dumps(asdict(key), sort_keys=True),
                "geometry": json.dumps(asdict(geometry), sort_keys=True)}

    def put(self, key, geometry, rows):
        from safetensors.torch import save_file

        rows = tuple(rows)
        names = geometry.names()
        if len(rows) != len(names) or any(
            tuple(row.shape) != geometry.tensor_shape(key.tokens)
            or str(row.dtype) != f"torch.{geometry.dtype}"
            for row in rows
        ):
            raise ValueError("Cache layers do not match the declared geometry")
        path = self.path(key)
        handle = tempfile.NamedTemporaryFile(dir=self.root, prefix=".kv-", delete=False)
        handle.close()
        temp_path = Path(handle.name)
        try:
            save_file({name: row.detach().cpu().contiguous() for name, row in zip(names, rows)},
                      str(temp_path), metadata=self._metadata(key, geometry))
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        return path

    def _inspect(self, file, key, geometry):
        names = geometry.names()
        if file.metadata() != self._metadata(key, geometry) or set(file.keys()) != set(names):
            raise ValueError("Cache artifact identity or layer list differs")
        for name in names:
            header = file.get_slice(name)
            if (tuple(header.get_shape()) != geometry.tensor_shape(key.tokens)
                    or header.get_dtype() != VALID_DTYPES[geometry.dtype]):
                raise ValueError("Cache tensor shape or dtype differs")
        return names

    def _fresh_path(self, key):
        path = self.path(key)
        if time.time() - path.stat().st_mtime >= self.ttl_seconds:
            raise FileNotFoundError("Cache artifact expired")
        return path

    def available(self, key, geometry):
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(self._fresh_path(key), framework="pt", device="cpu") as file:
                self._inspect(file, key, geometry)
        except (OSError, ValueError, SafetensorError):
            return False
        return True

    def read(self, key, geometry):
        from safetensors import safe_open

        with safe_open(self._fresh_path(key), framework="pt", device="cpu") as file:
            return tuple(file.get_tensor(name) for name in self._inspect(file, key, geometry))


def _block_indices(layer, block_ids):
    import torch

    if (layer.ndim != 5 or layer.shape[1] != 2 or not block_ids
            or len(set(block_ids)) != len(block_ids)
            or any(not 0 <= value < layer.shape[0] for value in block_ids)):
        raise ValueError("Expected distinct block IDs in a block-major NHD KV layer")
    return torch.tensor(block_ids, dtype=torch.long, device=layer.device)


def read_pages(layer, block_ids):
    """Gather [blocks, 2, block_size, heads, dim] as [2, tokens, heads, dim]."""
    indices = _block_indices(layer, block_ids)
    return layer.index_select(0, indices).transpose(0, 1).flatten(1, 2).contiguous()


def write_pages(layer, block_ids, row):
    """Scatter whole blocks into one paged vLLM layer."""
    indices = _block_indices(layer, block_ids)
    if (tuple(row.shape) != (2, len(block_ids) * layer.shape[2], *layer.shape[3:])
            or row.dtype != layer.dtype):
        raise ValueError("Cannot import a different block shape or dtype")
    pages = row.to(device=layer.device).reshape(2, len(block_ids), *layer.shape[2:])
    layer.index_copy_(0, indices, pages.transpose(0, 1).contiguous())
