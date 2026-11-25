"""Rank-limited answer distillation with affine export for the fast runtime."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .affine_cache import AffineCacheMapper
from .blocked_attention import blocked_attention
from .hf_backend import legacy
from .native_prefix import replace_prefix
from .train_mapper import NeuralMapper, model_forward


class AnswerMapper(nn.Module):
    """Train a shared input basis and one K and V projection per target layer."""

    def __init__(self, base, rank):
        super().__init__()
        if not base.manifest["head_mixing"] or rank < 1:
            raise ValueError("Full-head affine base and positive rank required")
        self.base = base.requires_grad_(False)
        _, layers, width, _ = base.weight.shape
        self.down = nn.Parameter(torch.empty(layers, width, rank, device=base.weight.device))
        self.up = nn.Parameter(torch.zeros(2, layers, rank, width, device=base.weight.device))
        nn.init.normal_(self.down, std=width**-.5)

    def map_block(self, layer, start, key, value, down, up, source_model, target_model):
        heads, dim, _ = self.base.manifest["target_geometry"]
        dtype = next(target_model.parameters()).dtype
        key = NeuralMapper._rotate_token_block(key, source_model, start, True)
        result = []
        for side, tensor in enumerate((key, value.float())):
            count = tensor.shape[-2]
            features = tensor.transpose(1, 2).reshape(1, count, heads * dim)
            base = features @ self.base.weight[side, layer] + self.base.bias[side, layer]
            corrected = base + (features @ down) @ up[side]
            corrected = corrected.reshape(1, count, heads, dim).transpose(1, 2)
            if side == 0:
                corrected = NeuralMapper._rotate_token_block(corrected, target_model, start)
            result.append(corrected.to(dtype).contiguous())
        return tuple(result)

    def apply_checkpointed(self, source_cache, source_model, target_model, token_block_size=256):
        from torch.utils.checkpoint import checkpoint
        from transformers import DynamicCache

        rows = legacy(source_cache)
        count = rows[0][0].shape[-2]
        out = []
        for layer, source_layer in enumerate(self.base.manifest["source_layers"]):
            parts = []
            source_key, source_value = rows[source_layer]
            for start in range(0, count, token_block_size):
                args = (source_key[:, :, start:start+token_block_size],
                        source_value[:, :, start:start+token_block_size],
                        self.down[layer], self.up[:, layer])

                def block(key, value, down, up, layer=layer, start=start):
                    return self.map_block(layer, start, key, value, down, up, source_model, target_model)

                parts.append(checkpoint(block, *args, use_reentrant=False) if torch.is_grad_enabled()
                             else block(*args))
            out.append(tuple(torch.cat([part[side] for part in parts], dim=-2) for side in range(2)))
        return DynamicCache.from_legacy_cache(tuple(out))

    @torch.inference_mode(False)
    @torch.no_grad()
    def folded(self, metadata=None):
        """Export the actual serving matrix, evaluated separately after folding."""
        result = AffineCacheMapper({**self.base.manifest, **(metadata or {})}, device=self.base.weight.device)
        result.weight.copy_(self.base.weight + self.down.unsqueeze(0) @ self.up)
        result.bias.copy_(self.base.bias)
        return result.eval().requires_grad_(False)


def soft_targets(runtime, full, prefix, native_prefix_tokens):
    with torch.no_grad():
        native_logits = model_forward(runtime.target, full, logits_to_keep=1).logits[0, -1].float()
        prefix_cache = model_forward(runtime.source, prefix, logits_to_keep=1).past_key_values
        native_head = (model_forward(runtime.target, prefix[:native_prefix_tokens], logits_to_keep=1).past_key_values
                       if native_prefix_tokens else None)
    return native_logits, prefix_cache, native_head


def answer_loss(runtime, mapper, full, prefix, suffix, option_ids, *, native_prefix_tokens):
    if full != prefix + suffix or not 0 <= native_prefix_tokens < len(prefix):
        raise ValueError("Training requires an exact cache handoff and a valid native prefix")
    teacher, source, head = soft_targets(runtime, full, prefix, native_prefix_tokens)
    mapped = mapper.apply_checkpointed(source, runtime.source, runtime.target)
    if head is not None:
        mapped = replace_prefix(mapped, head, native_prefix_tokens)
    # Torch 2.5.1's efficient backward rejects the saved causal mask's
    # head stride after activation offload. Only the differentiable query
    # uses blocked math SDPA; the teacher and serving keep their fast backend.
    with sdpa_kernel(SDPBackend.MATH), blocked_attention(64):
        student = model_forward(runtime.target, suffix, mapped, logits_to_keep=1).logits[0, -1].float()
    options = torch.tensor(sorted(option_ids.values()), dtype=torch.long, device=student.device)

    def kl(candidate, target):
        return F.kl_div(F.log_softmax(candidate, dim=-1), F.softmax(target, dim=-1), reduction="sum")

    full_kl = kl(student, teacher)
    choice_kl = kl(student.index_select(0, options), teacher.index_select(0, options))
    loss = .5 * (full_kl + choice_kl)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite answer distillation loss")
    return loss, {"loss": float(loss.detach()), "full_kl": float(full_kl.detach()),
                  "option_kl": float(choice_kl.detach()), "prefix_tokens": len(prefix)}


def sequence_kl_loss(runtime, mapper, prefix, suffix, option_ids, *, native_prefix_tokens, positions=8):
    """Backpropagate native distributions at spaced question positions and the decision."""
    if not 0 <= native_prefix_tokens < len(prefix) or not suffix or positions < 2:
        raise ValueError("Invalid sequence distillation boundary")
    count = min(positions, len(suffix))
    indices = [i * (len(suffix) - 1) // (count - 1) for i in range(count)] if count > 1 else [0]
    device = mapper.base.weight.device
    query_rows = torch.tensor(indices, dtype=torch.int64, device=device)
    with torch.no_grad():
        native = model_forward(runtime.target, prefix + suffix, logits_to_keep=query_rows + len(prefix))
        teacher = native.logits[0].float()
        del native
        source = model_forward(runtime.source, prefix, logits_to_keep=1).past_key_values
        head = (model_forward(runtime.target, prefix[:native_prefix_tokens], logits_to_keep=1).past_key_values
                if native_prefix_tokens else None)
    cache = mapper.apply_checkpointed(source, runtime.source, runtime.target)
    if head is not None:
        cache = replace_prefix(cache, head, native_prefix_tokens)
    with sdpa_kernel(SDPBackend.MATH), blocked_attention(64, rematerialize=True):
        student = model_forward(runtime.target, suffix, cache, logits_to_keep=query_rows).logits[0].float()
    if student.shape != teacher.shape or student.shape[0] != count:
        raise ValueError("Distillation positions differ across the cache handoff")
    options = torch.tensor(sorted(option_ids.values()), dtype=torch.int64, device=device)
    full_kl = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="batchmean")
    choice_kl = F.kl_div(F.log_softmax(student[-1, options], dim=-1), F.softmax(teacher[-1, options], dim=-1),
                        reduction="sum")
    loss = .75 * full_kl + .25 * choice_kl
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite sequence distillation loss")
    return loss, {"loss": float(loss.detach()), "full_kl": float(full_kl.detach()),
                  "option_kl": float(choice_kl.detach()), "prefix_tokens": len(prefix), "query_positions": indices}
