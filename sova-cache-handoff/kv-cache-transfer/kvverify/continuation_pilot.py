"""Train a multi-donor KV correction on native generated-answer trajectories."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from .blocked_attention import blocked_attention
from .hf_backend import legacy
from .native_prefix import replace_prefix
from .train_mapper import NeuralMapper, model_forward


class MultiDonorMapper(nn.Module):
    """Retain the affine initializer and learn a low-rank residual over three donors."""

    def __init__(self, base, rank=16, donors=3):
        super().__init__()
        if not base.manifest["head_mixing"] or rank < 1 or donors < 1:
            raise ValueError("A full-head affine base and positive residual geometry are required")
        self.base = base.requires_grad_(False)
        heads, dim, count = base.manifest["target_geometry"]
        available = base.manifest["source_geometry"][2]
        self.groups = tuple(tuple(sorted(range(available), key=lambda index: (abs(index-center), index))[:donors])
                            for center in base.manifest["source_layers"])
        width, actual_donors = heads*dim, len(self.groups[0])
        device = base.weight.device
        self.down = nn.Parameter(torch.empty(count, actual_donors, width, rank, device=device))
        self.up = nn.Parameter(torch.zeros(2, count, rank, width, device=device))
        nn.init.normal_(self.down, std=(width*actual_donors)**-.5)

    def residual_parameters(self):
        return self.down, self.up

    def apply(self, source_cache, source_model, target_model, block_tokens=1024):
        rows = legacy(source_cache)
        heads, dim, _ = self.base.manifest["target_geometry"]
        if len(rows) != self.base.manifest["source_geometry"][2] or block_tokens < 1:
            raise ValueError("Invalid source cache or block size")
        count = rows[0][0].shape[-2]
        if count < 1 or any(k.shape != (1, heads, count, dim) or v.shape != k.shape for k, v in rows):
            raise ValueError("Invalid source cache geometry")
        dtype = next(target_model.parameters()).dtype
        output = []
        requires_grad = torch.is_grad_enabled()
        for layer, donors in enumerate(self.groups):
            parts = []
            for offset in range(0, count, block_tokens):
                tensors = tuple(t[..., offset:offset+block_tokens, :].detach() for donor in donors for t in rows[donor])
                if requires_grad:
                    tensors = tuple(t.clone() if torch.is_inference(t) else t for t in tensors)

                def convert(*values, layer=layer, offset=offset):
                    down, up, *cache_rows = values
                    n = cache_rows[0].shape[-2]
                    result = []
                    for side in range(2):
                        features = []
                        for tensor in cache_rows[side::2]:
                            tensor = (NeuralMapper._rotate_token_block(tensor, source_model, offset, True)
                                      if side == 0 else tensor.float())
                            features.append(tensor.transpose(1, 2).reshape(n, heads * dim))
                        # The aligned donor is first in every group.
                        aligned = features[0] @ self.base.weight[side, layer] + self.base.bias[side, layer]
                        fused = sum(features[i] @ down[i] for i in range(len(features)))
                        values = (aligned + fused @ up[side]).reshape(1, n, heads, dim).transpose(1, 2)
                        if side == 0:
                            values = NeuralMapper._rotate_token_block(values, target_model, offset)
                        result.append(values.to(dtype).contiguous())
                    return tuple(result)

                args = (self.down[layer], self.up[:, layer], *tensors)
                parts.append(checkpoint(convert, *args, use_reentrant=False, preserve_rng_state=False)
                             if requires_grad else convert(*args))
            output.append(tuple(torch.cat([part[side] for part in parts], dim=-2) for side in range(2)))
        return DynamicCache.from_legacy_cache(tuple(output))

    def save(self, directory, metadata):
        from safetensors.torch import save_file

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        path = root/"weights.safetensors"
        weights = {name: getattr(self, name).detach().cpu().contiguous() for name in ("down", "up")}
        if not all(torch.isfinite(tensor).all() for tensor in weights.values()):
            raise ValueError("Nonfinite residual checkpoint")
        save_file(weights, str(path))
        manifest = {**metadata, "format": "multi-donor-answer-residual-v1",
                    "base_weights_sha256": self.base.manifest["weights_sha256"],
                    "rank": self.down.shape[-1], "source_layer_groups": self.groups,
                    "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return manifest

    @classmethod
    def load(cls, directory, base):
        from safetensors.torch import load_file

        root = Path(directory)
        manifest = json.loads((root/"manifest.json").read_text())
        path = root/"weights.safetensors"
        if (manifest["format"] != "multi-donor-answer-residual-v1"
                or manifest["base_weights_sha256"] != base.manifest["weights_sha256"]
                or manifest["weights_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("Residual checkpoint identity differs")
        instance = cls(base, rank=manifest["rank"], donors=len(manifest["source_layer_groups"][0]))
        if [list(group) for group in instance.groups] != manifest["source_layer_groups"]:
            raise ValueError("Residual checkpoint donor layout differs")
        values = load_file(str(path))
        if set(values) != {"down", "up"}:
            raise ValueError("Residual checkpoint tensors differ")
        with torch.no_grad():
            for name, tensor in values.items():
                parameter = getattr(instance, name)
                if tensor.shape != parameter.shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("Residual checkpoint tensor differs: " + name)
                parameter.copy_(tensor)
        return instance.eval()


@dataclass(frozen=True)
class Trace:
    tokens: tuple
    text: str
    ended: bool
    log_probs: torch.Tensor | None = None


def decode(runtime, model, ids, *, cache=None, maximum=24, store_distributions=False, first=None):
    """Greedy native/student decoding. Optionally retain the teacher's log probabilities."""
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids if isinstance(stop_ids, (tuple, list)) else (stop_ids,))
    generated, distributions = [], []
    incoming = ids
    ended = False
    for position in range(maximum):
        out = first if position == 0 and first is not None else runtime.forward(model, incoming, cache)
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        token = int(logits.argmax())
        generated.append(token)
        if store_distributions:
            distributions.append(F.log_softmax(logits, dim=-1).to(dtype=torch.float16, device="cpu"))
        if token in stop_ids:
            ended = True
            break
        incoming = [token]
    text = runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return Trace(tuple(generated), text, ended, torch.stack(distributions) if distributions else None)


def answer_token_loss(runtime, mapper, prefix, suffix, teacher, *, native_head=256, decision_weight=0.):
    """Align the question's final token and each teacher-forced answer position."""
    if (not teacher.ended or not teacher.tokens or not suffix or teacher.log_probs is None
            or teacher.log_probs.shape[0] != len(teacher.tokens) or not 0 <= native_head < len(prefix)
            or not 0 <= decision_weight < 1):
        raise ValueError("Invalid answer trajectory or cache boundary")
    with torch.no_grad():
        source = model_forward(runtime.source, prefix).past_key_values
        head = model_forward(runtime.target, prefix[:native_head]).past_key_values if native_head else None
    mapped = mapper.apply(source, runtime.source, runtime.target)
    if head is not None:
        mapped = replace_prefix(mapped, head, native_head)
    questions_and_answers = suffix + list(teacher.tokens[:-1])
    query_positions = torch.arange(len(suffix)-1, len(questions_and_answers), device=mapper.up.device)
    with sdpa_kernel(SDPBackend.MATH), blocked_attention(64, rematerialize=True):
        logits = model_forward(runtime.target, questions_and_answers, mapped, logits_to_keep=query_positions).logits[0].float()
    if logits.shape != teacher.log_probs.shape:
        raise ValueError("Generated-answer distributions have different shapes")
    target_log_probs = F.log_softmax(teacher.log_probs.to(device=logits.device, dtype=torch.float32), dim=-1)
    student_log_probs = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(student_log_probs, target_log_probs, reduction="batchmean", log_target=True)
    if decision_weight:
        first_token = F.kl_div(student_log_probs[0], target_log_probs[0], reduction="sum", log_target=True)
        kl = (1 - decision_weight) * kl + decision_weight * first_token
    if not torch.isfinite(kl):
        raise ValueError("Nonfinite generated-answer distillation loss")
    return kl
