"""Connector lifecycle and mapper tests with CPU pages and a vLLM interface double."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kvverify.vllm_handoff import CacheGeometry, CacheKey, CacheStore, read_pages


class ConnectorBase:
    def __init__(self, vllm_config, role, kv_cache_config):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self.metadata = None

    def has_connector_metadata(self):
        return self.metadata is not None

    def _get_connector_metadata(self):
        return self.metadata


class AttentionSpec(SimpleNamespace):
    pass


def connector_type():
    name = "kvverify._vllm_connector_fixture"
    path = Path(__file__).resolve().parents[1] / "kvverify" / "vllm_connector.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        name: module,
        "vllm.distributed.kv_transfer.kv_connector.v1.base": SimpleNamespace(
            KVConnectorBase_V1=ConnectorBase, KVConnectorMetadata=object),
        "vllm.v1.kv_cache_interface": SimpleNamespace(FullAttentionSpec=AttentionSpec),
    }):
        spec.loader.exec_module(module)
    module.version = lambda package: "0.34.0"
    return module.SovaKVConnector


@unittest.skipUnless(os.environ.get("KVVERIFY_TORCH_SMOKE") == "1", "Explicit tiny Torch fixtures")
class VllmHandoffTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.store = CacheStore(self.root.name)
        self.layer_names = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
        self.geometry = CacheGeometry(1, 4, 2, "float16")

    def engine(self, *, producer=False, consumer=False):
        import torch

        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={"scope": "tenant-a", "checkpoint": "/models/test",
                                           "model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
                                           "cache_directory": self.root.name},
                is_kv_producer=producer, is_kv_consumer=consumer, kv_load_failure_policy="recompute",
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            parallel_config=SimpleNamespace(),
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            model_config=SimpleNamespace(model="/models/test", hf_text_config=SimpleNamespace(model_type="qwen3")),
        )
        spec = AttentionSpec(block_size=2, num_kv_heads=1, head_size=4, head_size_v=4, dtype=torch.float16)
        groups = [SimpleNamespace(kv_cache_spec=spec, layer_names=self.layer_names[::-1], enable_kv_transfer=True)]
        return connector_type()(config, None, SimpleNamespace(kv_cache_groups=groups, num_blocks=8))

    @staticmethod
    def request(op, *, scope="tenant-a", tokens=(1, 2, 3, 4, 5), request_id="req"):
        return SimpleNamespace(
            request_id=request_id, cache_salt=scope, mm_features=[], lora_request=None, prompt_embeds=None,
            sampling_params=SimpleNamespace(prompt_logprobs=None), prompt_token_ids=tokens,
            kv_transfer_params={"sova": {"op": op, "id": "handoff", "prefix_tokens": 4}},
        )

    @staticmethod
    def step(request_id, start, count, blocks, *, new):
        cached = SimpleNamespace(req_ids=[] if new else [request_id], num_computed_tokens=[] if new else [start])
        added = [SimpleNamespace(req_id=request_id, num_computed_tokens=start)] if new else []
        return SimpleNamespace(scheduled_new_reqs=added, scheduled_cached_reqs=cached,
                               num_scheduled_tokens={request_id: count},
                               kv_connector_block_state=SimpleNamespace(get_block_ids=lambda _: (blocks,)))

    def test_chunked_native_export_receiving_blocks_and_recompute(self):
        import torch

        shape = (8, 2, 2, 1, 4)
        source_layers = {name: torch.arange(128, dtype=torch.float16).reshape(shape) + i
                         for i, name in enumerate(self.layer_names)}
        target_layers = {name: torch.zeros(shape, dtype=torch.float16) for name in self.layer_names}
        source, target = self.engine(producer=True), self.engine(consumer=True)
        source.register_kv_caches(source_layers)
        target.register_kv_caches(target_layers)

        export = self.request("export")
        source.update_state_after_alloc(export, None, 0)
        source.metadata = source.build_connector_meta(self.step("req", 0, 2, [2], new=True))
        self.assertEqual(source.metadata.exports, [])
        source.metadata = source.build_connector_meta(self.step("req", 2, 3, [2, 5, 6], new=False))
        source.wait_for_save()

        receive = self.request("load")
        self.assertEqual(target.get_num_new_matched_tokens(receive, 0), (4, False))
        self.assertEqual(target.get_num_new_matched_tokens(self.request("load", scope="tenant-b"), 0), (0, False))
        self.assertEqual(target.get_num_new_matched_tokens(self.request("load", tokens=(1, 99, 3, 4, 5)), 0),
                         (0, False))
        blocks = SimpleNamespace(get_block_ids=lambda: ([1, 4, 7],))
        target.update_state_after_alloc(receive, blocks, 4)
        target.metadata = target.build_connector_meta(self.step("req", 4, 1, [1, 4, 7], new=True))
        target.start_load_kv(None)
        self.assertEqual(target.get_block_ids_with_load_errors(), set())
        for name in self.layer_names:
            torch.testing.assert_close(read_pages(target_layers[name], [1, 4]), read_pages(source_layers[name], [2, 5]))
            self.assertEqual(int(torch.count_nonzero(target_layers[name][7])), 0)

        retry = self.request("load", request_id="retry")
        target.update_state_after_alloc(retry, blocks, 4)
        target.metadata = target.build_connector_meta(self.step("retry", 4, 1, [1, 4, 7], new=True))
        target._store.path(target.metadata.loads[0].key).unlink()
        with self.assertLogs("vllm.sova_cache_handoff", level="WARNING"):
            target.start_load_kv(None)
        self.assertEqual(target.get_block_ids_with_load_errors(), {1, 4})
        self.assertEqual(target.get_block_ids_with_load_errors(), set())
        self.assertEqual(target.get_num_new_matched_tokens(retry, 0), (0, False))
        target.request_finished(retry, [])
        self.assertNotIn("retry", target._attempted)

    def test_mapped_artifact_has_correct_target_geometry_and_native_head(self):
        import torch
        from transformers import Qwen3Config

        from kvverify.affine_cache import AffineCacheMapper
        from kvverify.continuation_pilot import MultiDonorMapper
        from kvverify.vllm_map import Checkpoint, VllmCacheMapper

        directory = Path(self.root.name)
        cfg = Qwen3Config(hidden_size=16, num_attention_heads=4, num_key_value_heads=1, head_dim=4)
        source = Checkpoint(directory, "a" * 64, "b" * 64, CacheGeometry(1, 4, 1, "float16"), cfg)
        target = Checkpoint(directory, "c" * 64, "b" * 64, self.geometry, cfg)
        manifest = {"source_geometry": [1, 4, 1], "target_geometry": [1, 4, 2], "source_layers": [0, 0],
                    "head_mixing": True, "source_sha256": source.sha256, "target_sha256": target.sha256,
                    "tokenizer_sha256": source.tokenizer_sha256}
        affine = AffineCacheMapper(manifest)
        with torch.no_grad():
            affine.weight.copy_(torch.eye(4).expand_as(affine.weight))
            affine.bias.zero_()
        affine_dir, residual_dir = directory / "affine", directory / "residual"
        affine.manifest = affine.save(affine_dir)
        MultiDonorMapper(affine, rank=2, donors=1).save(residual_dir, {})

        tokens = (1, 2, 3, 4)
        src_key = CacheKey.from_tokens("tenant-a", "handoff", tokens, source.sha256, source.tokenizer_sha256)
        row = torch.arange(32, dtype=torch.float16).reshape(2, 4, 1, 4) / 32
        self.store.put(src_key, source.geometry, (row,))
        head_key = CacheKey.from_tokens("tenant-a", "handoff", tokens[:2], target.sha256, target.tokenizer_sha256)
        native = torch.full((2, 2, 1, 4), 7, dtype=torch.float16)
        self.store.put(head_key, target.geometry, (native, native.clone()))
        for bundle in (None, residual_dir):
            with self.subTest(residual=bundle is not None):
                mapper = VllmCacheMapper(source, target, affine_dir, residual_dir=bundle, native_head_tokens=2)
                key = mapper.prepare(self.store, "tenant-a", "handoff", tokens)
                self.assertEqual(key.model_sha256, target.sha256)
                self.assertEqual(key.mapper_sha256, mapper.sha256)
                for mapped in self.store.read(key, target.geometry):
                    torch.testing.assert_close(mapped[:, :2], native, atol=0, rtol=0)
                    torch.testing.assert_close(mapped[:, 2:], row[:, 2:])
