"""Optional real-model adapter. Imports Torch only when explicitly instantiated.

Scope: dense Qwen3, batch=1, no padding, standard RoPE, local safetensors.
The initial mapper is a depth-aligned per-head affine ridge baseline, not an
implementation of OneTriangle's proprietary/hybrid mapper or top-k layer search.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

from .data import checkpoint_identity, json_digest, validate_pair


def dependencies():
    import torch
    from transformers import DynamicCache
    return torch, DynamicCache


def unrotate(keys, model, inverse=False):
    torch, _ = dependencies()
    positions = torch.arange(keys.shape[-2], device=keys.device).unsqueeze(0)
    # Qwen3 uses full, standard RoPE here; scaled/dynamic variants are rejected.
    cos, sin = model.model.rotary_emb(keys[:, 0].float(), positions)
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    x = keys.float()
    first, second = x.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return x*cos + rotated_half*sin*(-1 if inverse else 1)


def legacy(cache):
    result = cache.to_legacy_cache()
    if not result or any(k is None or v is None or k.shape != v.shape for k, v in result):
        raise ValueError("Uninitialized or mismatched KV cache")
    return result


class RidgeMapper:
    def __init__(self, manifest, tensors):
        self.manifest, self.tensors = manifest, tensors

    def to(self, device):
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}
        return self

    def apply(self, source_cache, source_model, target_model):
        torch, DynamicCache = dependencies()
        rows = legacy(source_cache)
        target_device = next(target_model.parameters()).device
        dtype = next(target_model.parameters()).dtype
        target_geometry = self.manifest["target_geometry"]
        if len(rows) != self.manifest["source_geometry"][2]:
            raise ValueError("Source layer count changed")
        seq = rows[0][0].shape[-2]
        if seq <= 0 or any(k.shape[0] != 1 or k.shape[1:2] != (self.manifest["source_geometry"][0],) or k.shape[-1] != self.manifest["source_geometry"][1] or k.shape[-2] != seq for k, _ in rows):
            raise ValueError("Unsupported source cache geometry")
        output = []
        for layer, source_layer in enumerate(self.manifest["source_layers"]):
            k, v = rows[source_layer]
            k = unrotate(k, source_model, inverse=True).to(target_device)
            v = v.float().to(target_device)
            mapped = []
            for kind, tensor in (("k", k), ("v", v)):
                w = self.tensors[kind+"_weight"][layer].to(target_device)
                b = self.tensors[kind+"_bias"][layer].to(target_device)
                value = torch.einsum("bhsd,hdf->bhsf", tensor, w) + b[None, :, None, :]
                if kind == "k":
                    value = unrotate(value, target_model)
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite mapped cache")
                mapped.append(value.to(dtype).contiguous())
            output.append(tuple(mapped))
        if len(output) != target_geometry[2]:
            raise ValueError("Target layer count mismatch")
        return DynamicCache.from_legacy_cache(tuple(output))

    def save(self, path):
        from safetensors.torch import save_file
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        weights = root / "weights.safetensors"
        save_file({k: v.contiguous().cpu() for k, v in self.tensors.items()}, str(weights))
        manifest = {**self.manifest, "format_version": 1, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        return manifest

    @classmethod
    def load(cls, path, source_sha, target_sha, tokenizer_sha):
        from safetensors.torch import load_file
        root = Path(path)
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest.get("format_version") != 1:
            raise ValueError("Unknown mapper format")
        for key, expected in (("source_sha256", source_sha), ("target_sha256", target_sha), ("tokenizer_sha256", tokenizer_sha)):
            if manifest.get(key) != expected:
                raise ValueError("Mapper checkpoint/tokenizer mismatch: " + key)
        weights = root/"weights.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("Mapper weights checksum mismatch")
        tensors = load_file(str(weights))
        heads, dim, layers = manifest["target_geometry"]
        if len(manifest["source_layers"]) != layers or any(type(i) is not int or not 0 <= i < manifest["source_geometry"][2] for i in manifest["source_layers"]):
            raise ValueError("Invalid mapper source-layer selection")
        torch, _ = dependencies()
        for kind in ("k", "v"):
            for suffix, shape in (("weight", (layers, heads, dim, dim)), ("bias", (layers, heads, dim))):
                tensor = tensors[kind+"_"+suffix]
                if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid mapper tensor")
        return cls(manifest, tensors)


class DenseRuntime:
    def __init__(self, source, target, tokenizer=None, identities=None):
        self.torch, self.DynamicCache = dependencies()
        self.source, self.target, self.tokenizer = source.eval(), target.eval(), tokenizer
        self.source_geometry, self.target_geometry = validate_pair(source.config.to_dict(), target.config.to_dict())
        self.identities = identities or {"source": {"sha256": "random-smoke-source"}, "target": {"sha256": "random-smoke-target"}}
        self.tokenizer_sha256 = self.tokenizer_fingerprint(tokenizer) if tokenizer is not None else "random-token-smoke"
        self.mapper = None

    @staticmethod
    def tokenizer_fingerprint(tokenizer):
        return json_digest({"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map})

    @classmethod
    def from_local(cls, source_path, target_path, device="cuda:0", dtype="bfloat16"):
        if not device.startswith("cuda"):
            raise ValueError("Real checkpoint commands require an approved CUDA worker; CPU/MPS is reserved for tiny explicit tests")
        torch, _ = dependencies()
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; no CPU fallback")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if importlib.metadata.version("transformers") != "4.52.4":
            raise ValueError("This adapter targets transformers==4.52.4; use an isolated approved environment")
        identities = {"source": checkpoint_identity(source_path), "target": checkpoint_identity(target_path)}
        for path in (source_path, target_path):
            config = json.loads((Path(path)/"config.json").read_text())
            validate_pair(config, config)
        st = AutoTokenizer.from_pretrained(source_path, local_files_only=True, trust_remote_code=False)
        tt = AutoTokenizer.from_pretrained(target_path, local_files_only=True, trust_remote_code=False)
        if cls.tokenizer_fingerprint(st) != cls.tokenizer_fingerprint(tt):
            raise ValueError("Tokenizer vocabulary/special-token mismatch")
        dt = getattr(torch, dtype)
        kwargs = {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                  "torch_dtype": dt, "attn_implementation": "sdpa"}
        source = AutoModelForCausalLM.from_pretrained(source_path, **kwargs).to(device)
        target = source if identities["source"]["sha256"] == identities["target"]["sha256"] else AutoModelForCausalLM.from_pretrained(target_path, **kwargs).to(device)
        runtime = cls(source, target, tt, identities)
        runtime.source_tokenizer = st
        return runtime

    def sync(self):
        devices = {next(m.parameters()).device for m in (self.source, self.target)}
        for device in devices:
            if device.type == "cuda":
                self.torch.cuda.synchronize(device)

    def timed(self, fn):
        self.sync()
        start = time.perf_counter()
        result = fn()
        self.sync()
        return result, (time.perf_counter()-start)*1000

    def forward(self, model, ids, cache=None):
        torch = self.torch
        device = next(model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)
        offset = cache.get_seq_length() if cache is not None else 0
        with torch.inference_mode():
            return model(input_ids=x, past_key_values=cache, use_cache=True,
                         attention_mask=torch.ones((1, offset+len(ids)), dtype=torch.long, device=device),
                         position_ids=torch.arange(offset, offset+len(ids), device=device).unsqueeze(0),
                         cache_position=torch.arange(offset, offset+len(ids), device=device),
                         logits_to_keep=1, return_dict=True)

    def clone_cache(self, cache):
        return self.DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in legacy(cache)))

    def identity_mapper(self):
        if self.source is not self.target:
            raise ValueError("Identity mapping requires the exact same model object")
        torch = self.torch
        h, d, layers = self.target_geometry
        weights = torch.eye(d).repeat(layers, h, 1, 1)
        bias = torch.zeros(layers, h, d)
        return RidgeMapper({"source_geometry": self.source_geometry, "target_geometry": self.target_geometry,
                            "source_layers": list(range(layers)), "method": "identity"},
                           {"k_weight": weights, "v_weight": weights.clone(), "k_bias": bias, "v_bias": bias.clone()})

    def fit(self, examples, alpha=.01, prefix_tokens=1024):
        if not math.isfinite(alpha) or not 0 < alpha or prefix_tokens < 2:
            raise ValueError("Positive ridge regularization and >=2 prefix tokens required")
        if prefix_tokens > min(self.source.config.max_position_embeddings, self.target.config.max_position_embeddings):
            raise ValueError("Calibration prefix exceeds the native context limit")
        torch = self.torch
        h, d, target_layers = self.target_geometry
        source_layers = [round(i*(self.source_geometry[2]-1)/max(1, target_layers-1)) for i in range(target_layers)]
        shape = (2, target_layers, h)
        count = 0
        sx = torch.zeros(*shape, d, dtype=torch.float64)
        sy = torch.zeros_like(sx)
        xx = torch.zeros(*shape, d, d, dtype=torch.float64)
        xy = torch.zeros_like(xx)
        used = []
        for example in examples:
            ids = self.tokenizer(example.context, add_special_tokens=False)["input_ids"][:prefix_tokens]
            if len(ids) < 2:
                raise ValueError("Calibration document too short")
            with torch.inference_mode():
                source = legacy(self.forward(self.source, ids).past_key_values)
                target = legacy(self.forward(self.target, ids).past_key_values)
                for layer, sl in enumerate(source_layers):
                    for kind in range(2):
                        x = unrotate(source[sl][kind], self.source, True) if kind == 0 else source[sl][kind]
                        y = unrotate(target[layer][kind], self.target, True) if kind == 0 else target[layer][kind]
                        x, y = x[0].double().cpu(), y[0].double().cpu()
                        sx[kind, layer] += x.sum(1)
                        sy[kind, layer] += y.sum(1)
                        xx[kind, layer] += x.transpose(1, 2) @ x
                        xy[kind, layer] += x.transpose(1, 2) @ y
            count += len(ids)
            used.append({"id": example.id, "prefix_tokens": len(ids), "document_id": example.document_id})
        if count < 2:
            raise ValueError("Insufficient calibration data")
        mx, my = sx/count, sy/count
        covariance = xx/count - mx.unsqueeze(-1)*mx.unsqueeze(-2)
        cross = xy/count - mx.unsqueeze(-1)*my.unsqueeze(-2)
        weights = torch.linalg.solve(covariance + alpha*torch.eye(d, dtype=torch.float64), cross)
        bias = my - torch.einsum("klhi,klhij->klhj", mx, weights)
        if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
            raise ValueError("Nonfinite fit")
        manifest = {"method": "depth_aligned_per_head_affine_ridge_baseline", "source_geometry": self.source_geometry,
                    "target_geometry": self.target_geometry, "source_layers": source_layers, "alpha": alpha,
                    "calibration_tokens": count, "calibration_prefix_limit": prefix_tokens, "calibration_used": used,
                    "source_sha256": self.identities["source"]["sha256"], "target_sha256": self.identities["target"]["sha256"],
                    "tokenizer_sha256": self.tokenizer_sha256,
                    "calibration_ids": [e.id for e in examples], "calibration_documents": sorted({e.document_id for e in examples}),
                    "calibration_context_hashes": sorted({e.context_hash for e in examples}),
                    "limitations": "Fixed depth alignment, k=1; no task-trained residual adapters, top-k search or hybrid recurrent-state transfer."}
        self.mapper = RidgeMapper(manifest, {"k_weight": weights[0].float(), "v_weight": weights[1].float(),
                                             "k_bias": bias[0].float(), "v_bias": bias[1].float()})
        return self.mapper

    def tokenize(self, example):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": example.prompt()}],
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(self, "source_tokenizer") and self.source_tokenizer(text, add_special_tokens=False)["input_ids"] != ids:
            raise ValueError("Prompt token IDs differ between models")
        return ids

    def answer(self, example, arm, repeat, config_hash, max_new_tokens=4):
        from .evidence import ARMS
        if arm not in ARMS or max_new_tokens < 1:
            raise ValueError("Invalid generation request")
        ids = self.tokenize(example)
        base = {"id": example.id, "document_id": example.document_id, "context_hash": example.context_hash,
                "prompt_hash": json_digest(ids), "run_config_hash": config_hash, "repeat": repeat,
                "arm": arm, "evidence_kind": "hf_model", "prompt_tokens": len(ids), "fallback": False}
        if len(ids) < 2 or len(ids) > getattr(self, 'prompt_token_limit', float('inf')) or len(ids)+max_new_tokens > min(self.source.config.max_position_embeddings, self.target.config.max_position_embeddings):
            return {**base, "status": "over_limit", "correct": False}
        model = self.source if arm == "source" else self.target
        torch = self.torch
        setup_ms = 0.
        prepared = None
        if arm in ("native_warm", "transfer_warm"):
            reader = self.target if arm == "native_warm" else self.source
            prepared, setup_ms = self.timed(lambda: self.forward(reader, ids[:-1]).past_key_values)
        for device in {next(m.parameters()).device for m in (self.source, self.target)}:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        times = {"source_prefill_ms": 0., "map_ms": 0., "import_ms": 0.}
        self.sync()
        started = time.perf_counter()
        if arm.startswith("transfer"):
            if self.mapper is None:
                raise ValueError("A verified mapper is required; no fallback")
            if arm == "transfer_cold":
                prepared, times["source_prefill_ms"] = self.timed(lambda: self.forward(self.source, ids[:-1]).past_key_values)
            else:
                times["source_prefill_ms"] = setup_ms
            with torch.inference_mode():
                mapped, times["map_ms"] = self.timed(lambda: self.mapper.apply(prepared, self.source, self.target))
            imported = mapped.get_seq_length()
            if imported != len(ids)-1 or len(legacy(mapped)) != self.target_geometry[2]:
                raise ValueError("Cache import length/layer verification failed")
            # Mapper already includes movement and DynamicCache construction.
            cache, tail = mapped, ids[-1:]
            base.update(imported_tokens=imported, target_prefill_tokens=1, cache_import_verified=True,
                        map_includes_move_and_import=True, mapper_method=self.mapper.manifest["method"])
        elif arm == "native_warm":
            cache, times["import_ms"] = self.timed(lambda: self.clone_cache(prepared))
            tail = ids[-1:]
        else:
            cache, tail = None, ids
        def first():
            out = self.forward(model, tail, cache)
            return int(out.logits[0, -1].argmax().item()), out.past_key_values, out.logits[0, -1]
        (first_id, cache, logits), times["target_first_ms"] = self.timed(first)
        self.sync()
        ttft = (time.perf_counter()-started)*1000
        generated = [first_id]
        eos = model.config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for _ in range(max_new_tokens-1):
            if generated[-1] in eos:
                break
            out = self.forward(model, [generated[-1]], cache)
            generated.append(int(out.logits[0, -1].argmax().item()))
            cache = out.past_key_values
        self.sync()
        total = (time.perf_counter()-started)*1000
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        memory = {str(d): {"allocated_mib": torch.cuda.max_memory_allocated(d)/2**20,
                            "reserved_mib": torch.cuda.max_memory_reserved(d)/2**20}
                  for d in {next(m.parameters()).device for m in (self.source, self.target)} if d.type == "cuda"}
        logit_sha = hashlib.sha256(logits.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        # Optional prospective scoring runs AFTER all model execution timers.
        # It cannot change the decoded answer, trace correctness, or TTFT.
        observer = getattr(self, 'quality_observer', None)
        diagnostics = observer(logits, example, answer, generated, eos, max_new_tokens) if observer else {}
        reserved = set(base) | set(times) | {'status', 'ttft_ms', 'total_ms', 'correct', 'answer'}
        if reserved.intersection(diagnostics):
            raise ValueError('Quality observer cannot overwrite evidence fields')
        return {**base, **times, "status": "ok", "ttft_ms": ttft, "total_ms": total,
                "warm_cache_setup_ms_excluded": setup_ms, "answer": answer, "correct": answer == example.answer,
                "first_token_id": first_id, "first_logits_sha256": logit_sha, "generated_token_ids": generated,
                "gpu_peak_memory": memory, "measurement": "in-process synchronized HF first-token selection; not HTTP TTFT or visible-answer latency", **diagnostics}
