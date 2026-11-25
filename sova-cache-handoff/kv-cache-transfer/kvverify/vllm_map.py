"""Translate an exported vLLM cache with an existing SOVA mapper checkpoint."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from .affine_cache import AffineCacheMapper
from .continuation_pilot import MultiDonorMapper
from .data import checkpoint_identity, json_digest, validate_geometry
from .fast_affine import map_affine_fast, map_multi_donor_fast
from .hf_backend import DenseRuntime, legacy
from .native_prefix import replace_prefix
from .vllm_handoff import CacheGeometry, CacheKey


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str
    tokenizer_sha256: str
    geometry: CacheGeometry
    config: Qwen3Config

    @classmethod
    def from_local(cls, path, dtype):
        path = Path(path).expanduser().resolve()
        config = json.loads((path / "config.json").read_text())
        geometry = CacheGeometry(*validate_geometry(config), dtype)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        return cls(path, checkpoint_identity(path)["sha256"], DenseRuntime.tokenizer_fingerprint(tokenizer),
                   geometry, Qwen3Config.from_dict(config))

    def connector_config(self, cache_directory, scope, *, role, mapper_sha256=""):
        """Return the dict consumed by vLLM's --kv-transfer-config flag."""
        if role not in ("kv_producer", "kv_consumer", "kv_both"):
            raise ValueError("Invalid vLLM KV role")
        return {
            "kv_connector": "SovaKVConnector",
            "kv_connector_module_path": "kvverify.vllm_connector",
            "kv_role": role,
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "cache_directory": str(Path(cache_directory).expanduser().resolve()),
                "scope": scope,
                "checkpoint": str(self.path),
                "model_sha256": self.sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "mapper_sha256": mapper_sha256,
            },
        }


class _RotaryModel:
    """The rotary interface used by the mapper, without loading LM weights."""

    def __init__(self, checkpoint, device):
        self._placement = torch.empty(0, device=device, dtype=getattr(torch, checkpoint.geometry.dtype))
        self.model = SimpleNamespace(rotary_emb=Qwen3RotaryEmbedding(config=checkpoint.config).to(device))

    def parameters(self):
        return iter((self._placement,))


def _dense_cache(rows, device):
    return DynamicCache.from_legacy_cache(tuple(
        (row[0].transpose(0, 1).unsqueeze(0).to(device).contiguous(),
         row[1].transpose(0, 1).unsqueeze(0).to(device).contiguous())
        for row in rows
    ))


class VllmCacheMapper:
    def __init__(self, source, target, affine_dir, *, residual_dir=None, native_head_tokens=0, device="cpu"):
        if source.tokenizer_sha256 != target.tokenizer_sha256:
            raise ValueError("Both model checkpoints must have the same tokenizer")
        if type(native_head_tokens) is not int or native_head_tokens < 0:
            raise ValueError("Native head token count must be nonnegative")
        base = AffineCacheMapper.load(affine_dir, source.sha256, target.sha256, source.tokenizer_sha256)
        manifest = base.manifest
        for key, checkpoint in (("source_geometry", source), ("target_geometry", target)):
            g = checkpoint.geometry
            if tuple(manifest[key]) != (g.heads, g.head_dim, g.layers):
                raise ValueError("Mapper cache geometry disagrees with the local checkpoint")
        if not manifest["head_mixing"]:
            raise ValueError("The vLLM bridge uses the all-head affine mapper")
        self.source = source
        self.target = target
        self.native_head_tokens = native_head_tokens
        self._device = torch.device(device)
        self._source_rope = _RotaryModel(source, self._device)
        self._target_rope = _RotaryModel(target, self._device)
        self._affine = base.to(self._device).requires_grad_(False)
        self._residual = None
        residual_manifest = None
        if residual_dir is not None:
            self._residual = MultiDonorMapper.load(residual_dir, self._affine).requires_grad_(False)
            residual_manifest = json.loads((Path(residual_dir) / "manifest.json").read_text())
        self.sha256 = json_digest({"format": "vllm-qwen3-sova-v1", "affine": manifest,
                                   "residual": residual_manifest, "native_head_tokens": native_head_tokens})

    @torch.inference_mode()
    def prepare(self, store, scope, handoff_id, prefix):
        """Map ready native artifacts and publish the receiver's exact prefix."""
        prefix = tuple(prefix)
        if self.native_head_tokens >= len(prefix):
            raise ValueError("The mapped prefix must extend beyond the native head")
        source_key = CacheKey.from_tokens(scope, handoff_id, prefix, self.source.sha256,
                                          self.source.tokenizer_sha256)
        source = _dense_cache(store.read(source_key, self.source.geometry), self._device)
        if self._residual is None:
            mapped = map_affine_fast(self._affine, source, self._source_rope, self._target_rope)
        else:
            mapped = map_multi_donor_fast(self._residual, source, self._source_rope, self._target_rope)
        if self.native_head_tokens:
            head_key = CacheKey.from_tokens(scope, handoff_id, prefix[:self.native_head_tokens],
                                            self.target.sha256, self.target.tokenizer_sha256)
            native_head = _dense_cache(store.read(head_key, self.target.geometry), self._device)
            mapped = replace_prefix(mapped, native_head, self.native_head_tokens)
        rows = tuple(torch.stack((key[0].transpose(0, 1), value[0].transpose(0, 1))).cpu()
                     for key, value in legacy(mapped))
        if any(not bool(torch.isfinite(row).all()) for row in rows):
            raise ValueError("Mapped KV contains nonfinite values")
        target_key = CacheKey.from_tokens(scope, handoff_id, prefix, self.target.sha256,
                                          self.target.tokenizer_sha256, self.sha256)
        store.put(target_key, self.target.geometry, rows)
        return target_key
