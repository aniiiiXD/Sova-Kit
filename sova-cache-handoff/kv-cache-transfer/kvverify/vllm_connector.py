"""vLLM 0.34.0 V1 connector for single-host SOVA cache handoffs."""
from dataclasses import dataclass, field
from importlib.metadata import version
import logging
from pathlib import Path
import re

import torch
from safetensors import SafetensorError
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .vllm_handoff import CacheGeometry, CacheKey, CacheStore, read_pages, write_pages


logger = logging.getLogger("vllm.sova_cache_handoff")
_VLLM_VERSION = "0.34.0"
_ARTIFACT_ERRORS = (OSError, ValueError, SafetensorError, RuntimeError)


@dataclass(frozen=True)
class BlockTransfer:
    key: CacheKey
    blocks: tuple[int, ...]


@dataclass
class SovaMetadata(KVConnectorMetadata):
    loads: list[BlockTransfer] = field(default_factory=list)
    exports: list[BlockTransfer] = field(default_factory=list)


class SovaKVConnector(KVConnectorBase_V1):
    """Load ready prefixes and export complete native blocks after a forward.

    The mapper prepares translated artifacts separately from the GPU scheduler.
    Disable vLLM APC on receivers so translated state cannot populate its
    native prefix cache.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        if version("vllm").split("+", 1)[0] != _VLLM_VERSION:
            raise ValueError(f"SovaKVConnector requires vLLM {_VLLM_VERSION}")
        super().__init__(vllm_config, role, kv_cache_config)
        config = self._kv_transfer_config
        extra = config.kv_connector_extra_config
        self._can_export = config.is_kv_producer
        self._can_load = config.is_kv_consumer
        if self._can_load and (vllm_config.cache_config.enable_prefix_caching
                               or config.kv_load_failure_policy != "recompute"):
            raise ValueError("SOVA receivers require disabled prefix caching and recompute on load failure")
        parallel = vllm_config.parallel_config
        if (any(getattr(parallel, field, 1) != 1 for field in (
                "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size"))
                or vllm_config.speculative_config is not None
                or vllm_config.scheduler_config.async_scheduling):
            raise ValueError("SOVA expects one worker per model with synchronous scheduling")

        model = vllm_config.model_config
        groups = kv_cache_config.kv_cache_groups
        if (model.hf_text_config.model_type != "qwen3"
                or len(groups) != 1 or not groups[0].enable_kv_transfer
                or type(groups[0].kv_cache_spec) is not FullAttentionSpec):
            raise ValueError("SOVA expects one dense Qwen3 full-attention KV group")
        if Path(model.model).resolve() != Path(extra["checkpoint"]).resolve():
            raise ValueError("vLLM must serve the exact local checkpoint in the connector config")
        spec = groups[0].kv_cache_spec
        self._block_size = spec.block_size
        self._num_blocks = kv_cache_config.num_blocks
        layer_numbers = {}
        for name in groups[0].layer_names:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match is None or int(match[1]) in layer_numbers:
                raise ValueError("Could not identify unique Qwen3 layer indices")
            layer_numbers[int(match[1])] = name
        if sorted(layer_numbers) != list(range(len(layer_numbers))):
            raise ValueError("Qwen3 KV layers must start at zero and be contiguous")
        self._layers = tuple(layer_numbers[index] for index in range(len(layer_numbers)))
        self._geometry = CacheGeometry(spec.num_kv_heads, spec.head_size, len(self._layers),
                                       str(spec.dtype).removeprefix("torch."))
        if spec.head_size_v != spec.head_size:
            raise ValueError("SOVA expects matching key and value head sizes")
        self._scope = extra["scope"]
        self._model_sha = extra["model_sha256"]
        self._tokenizer_sha = extra["tokenizer_sha256"]
        self._mapper_sha = extra.get("mapper_sha256", "")
        CacheKey.from_tokens(self._scope, "config-check", (0,), self._model_sha,
                             self._tokenizer_sha, self._mapper_sha)
        self._max_tokens = extra.get("max_prefix_tokens", 8192)
        if type(self._max_tokens) is not int or self._max_tokens < self._block_size:
            raise ValueError("max_prefix_tokens must allow at least one KV block")
        self._store = CacheStore(extra["cache_directory"], ttl_seconds=extra.get("ttl_seconds", 900))

        self._loads = {}
        self._exports = {}
        self._attempted = set()
        self._invalid_blocks = set()
        self._cache_layers = {}

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        return "NHD"

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config):
        return False

    @property
    def requires_kv_delivery(self):
        return False

    def _key(self, request, op):
        transfer = request.kv_transfer_params
        params = transfer.get("sova") if isinstance(transfer, dict) else None
        if (not isinstance(params, dict) or params.get("op") != op
                or request.cache_salt != self._scope or request.mm_features
                or request.lora_request is not None or request.prompt_embeds is not None
                or request.sampling_params is None
                or request.sampling_params.prompt_logprobs is not None):
            return None
        tokens = request.prompt_token_ids or []
        count = params.get("prefix_tokens")
        if (type(count) is not int or not 0 < count < len(tokens)
                or count > self._max_tokens or count % self._block_size):
            return None
        try:
            return CacheKey.from_tokens(self._scope, params.get("id"), tokens[:count],
                                        self._model_sha, self._tokenizer_sha,
                                        self._mapper_sha if op == "load" else "")
        except ValueError:
            return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self._can_load or num_computed_tokens or request.request_id in self._attempted:
            return 0, False
        key = self._key(request, "load")
        if key is None or not self._store.available(key, self._geometry):
            return 0, False
        return key.tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        request_id = request.request_id
        if num_external_tokens:
            key = self._key(request, "load")
            if key is None or key.tokens != num_external_tokens:
                raise ValueError("vLLM allocated a different external prefix length")
            self._loads[request_id] = BlockTransfer(key, self._prefix_blocks(blocks.get_block_ids(), key))
            self._attempted.add(request_id)
        elif self._can_export:
            key = self._key(request, "export")
            if key is not None:
                self._exports[request_id] = key

    def _prefix_blocks(self, groups, key):
        count = key.tokens // self._block_size
        if len(groups) != 1 or len(groups[0]) < count:
            raise ValueError("vLLM did not allocate enough blocks for the requested prefix")
        return tuple(groups[0][:count])

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        starts = dict(zip(cached.req_ids, cached.num_computed_tokens))
        starts.update((row.req_id, row.num_computed_tokens) for row in scheduler_output.scheduled_new_reqs)
        block_state = scheduler_output.kv_connector_block_state
        meta = SovaMetadata()
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            load = self._loads.pop(request_id, None)
            if load is not None:
                meta.loads.append(load)
            elif (key := self._exports.get(request_id)) is not None:
                if starts[request_id] + scheduled < key.tokens:
                    continue
                groups = block_state.get_block_ids(request_id) if block_state is not None else None
                if groups is not None:
                    meta.exports.append(BlockTransfer(key, self._prefix_blocks(groups, key)))
                    self._exports.pop(request_id)
        return meta

    def request_finished(self, request, block_ids):
        self._loads.pop(request.request_id, None)
        self._exports.pop(request.request_id, None)
        self._attempted.discard(request.request_id)
        return False, None

    def register_kv_caches(self, kv_caches):
        shape = (self._num_blocks, 2, self._block_size, self._geometry.heads, self._geometry.head_dim)
        for name in self._layers:
            layer = kv_caches[name]
            if tuple(layer.shape) != shape or str(layer.dtype) != f"torch.{self._geometry.dtype}":
                raise ValueError(f"Unexpected vLLM KV layout for {name}, expected block-major NHD")
        self._cache_layers = {name: kv_caches[name] for name in self._layers}

    @torch.inference_mode()
    def start_load_kv(self, forward_context, **kwargs):
        if not self.has_connector_metadata():
            return
        for transfer in self._get_connector_metadata().loads:
            try:
                rows = self._store.read(transfer.key, self._geometry)
                for name, row in zip(self._layers, rows):
                    write_pages(self._cache_layers[name], transfer.blocks, row)
                logger.info("SOVA loaded %d external KV tokens", transfer.key.tokens)
            except _ARTIFACT_ERRORS as error:
                self._invalid_blocks.update(transfer.blocks)
                logger.warning("SOVA cache load failed (%s); vLLM will recompute", type(error).__name__)

    def get_block_ids_with_load_errors(self):
        errors, self._invalid_blocks = self._invalid_blocks, set()
        return errors

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    @torch.inference_mode()
    def wait_for_save(self):
        if not self.has_connector_metadata():
            return
        for transfer in self._get_connector_metadata().exports:
            try:
                rows = tuple(read_pages(self._cache_layers[name], transfer.blocks).cpu()
                             for name in self._layers)
                self._store.put(transfer.key, self._geometry, rows)
                logger.info("SOVA exported %d native KV tokens", transfer.key.tokens)
            except _ARTIFACT_ERRORS as error:
                logger.warning("SOVA cache export failed (%s)", type(error).__name__)
