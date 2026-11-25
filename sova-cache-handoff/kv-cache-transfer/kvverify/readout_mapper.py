"""Fit the affine residual against actual frozen target attention reads."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .fast_affine import map_affine_fast
from .hf_backend import legacy
from .native_prefix import replace_prefix
from .state_diagnostic import attention_readout


@contextmanager
def capture_queries(model, layers, positions):
    """Capture only selected post-RoPE queries while preserving native SDPA."""
    active = None
    captured = {}
    original = F.scaled_dot_product_attention

    def before(layer):
        def hook(_module, _args):
            nonlocal active
            active = layer
        return hook

    def after(_module, _args, _output):
        nonlocal active
        active = None

    def capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if active is not None:
            if active in captured or dropout_p or q.shape[0] != 1 or max(positions) >= q.shape[-2]:
                raise ValueError("Unexpected target attention invocation")
            captured[active] = (
                q[:, :, positions].detach(),
                float(scale if scale is not None else q.shape[-1] ** -0.5),
            )
        return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                        scale=scale, enable_gqa=enable_gqa)

    handles = []
    try:
        for layer in layers:
            attention = model.model.layers[layer].self_attn
            handles.append(attention.register_forward_pre_hook(before(layer)))
            handles.append(attention.register_forward_hook(after))
        F.scaled_dot_product_attention = capture
        yield captured
        if captured.keys() != set(layers):
            raise ValueError("Target forward did not capture all selected layers")
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()


def readout_loss(query, native, hybrid, *, prefix_tokens, positions, scale,
                 mapped_query=None, output_projection=None):
    """Match the value read and weakly constrain the QK attention distribution."""
    arguments = dict(prefix_tokens=prefix_tokens, query_indices=positions, scale=scale)
    with torch.no_grad():
        target, weights, log_weights, _ = attention_readout(query, *native, **arguments)
    actual, _, log_actual, _ = attention_readout(
        query if mapped_query is None else mapped_query, *hybrid, **arguments
    )
    # Mask each log before subtracting so fully masked positions never enter loss math.
    mask = weights > 0
    log_ratio = log_weights.masked_fill(~mask, 0) - log_actual.masked_fill(~mask, 0)
    attention_kl = (weights * log_ratio).sum(-1).mean()
    energy = target.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    relative_mse = ((actual - target).square() / energy).mean()
    if output_projection is not None:
        def project(readout):
            hidden = readout.transpose(1, 2).reshape(1, len(positions), -1)
            return F.linear(hidden, output_projection)

        reference_output = project(target)
        projected_mse = (project(actual) - reference_output).square().mean() / reference_output.square().mean().clamp_min(1e-5)
        relative_mse = 0.5 * (relative_mse + projected_mse)
    return relative_mse + 0.02 * attention_kl


def fit_document(runtime, mapper, optimizer, prefix, suffix, layers, *,
                 native_head_tokens, query_count, student_trajectory=False):
    """One update on one complete document with only mapper gradients."""
    if not 0 < native_head_tokens < len(prefix) or not suffix or not layers or query_count < 2:
        raise ValueError("Invalid readout training boundary")
    count = min(query_count, len(suffix))
    positions = tuple(i * (len(suffix) - 1) // (count - 1) for i in range(count)) if count > 1 else (0,)
    source_cache = runtime.forward(runtime.source, prefix).past_key_values
    source = legacy(source_cache)
    native_prefix = runtime.forward(runtime.target, prefix).past_key_values
    with capture_queries(runtime.target, layers, positions) as captured:
        native = legacy(runtime.forward(runtime.target, suffix, native_prefix).past_key_values)
    native_head_cache = runtime.forward(runtime.target, prefix[:native_head_tokens]).past_key_values
    native_head = legacy(native_head_cache)
    mapped_queries = {}
    mapped_suffix = {}
    if student_trajectory:
        folded = mapper.folded()
        with torch.inference_mode():
            current = map_affine_fast(folded, source_cache, runtime.source, runtime.target)
            current = replace_prefix(current, native_head_cache, native_head_tokens)
            with capture_queries(runtime.target, layers, positions) as mapped_queries:
                student = runtime.forward(runtime.target, suffix, current)
            mapped_suffix = {layer: tuple(tensor[..., len(prefix):, :].clone()
                                          for tensor in legacy(student.past_key_values)[layer])
                             for layer in layers}
            del current, student

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for layer in layers:
        query, scale = captured.pop(layer)
        reference = tuple(tensor.clone() for tensor in native[layer])
        input_pair = tuple(tensor[..., native_head_tokens:, :].clone()
                           for tensor in source[mapper.base.manifest["source_layers"][layer]])
        mapped = mapper.map_block(layer, native_head_tokens, *input_pair, mapper.down[layer], mapper.up[:, layer],
                                  runtime.source, runtime.target)
        history_tail = mapped_suffix[layer] if student_trajectory else tuple(
            tensor[..., len(prefix):, :] for tensor in reference
        )
        hybrid = tuple(torch.cat((head, middle, tail), dim=-2)
                       for head, middle, tail in zip(native_head[layer], mapped, history_tail))
        mapped_query, projection = None, None
        if student_trajectory:
            mapped_query, mapped_scale = mapped_queries.pop(layer)
            if mapped_scale != scale:
                raise ValueError("Native and mapped readout scales differ")
            mapped_query = mapped_query.clone()
            projection = runtime.target.model.layers[layer].self_attn.o_proj.weight.detach().float()
        loss = readout_loss(query.clone(), reference, hybrid, prefix_tokens=len(prefix),
                            positions=positions, scale=scale, mapped_query=mapped_query,
                            output_projection=projection)
        losses.append(loss.detach())
        (loss / len(layers)).backward()
    loss = float(torch.stack(losses).mean())
    norm = float(torch.nn.utils.clip_grad_norm_((mapper.down, mapper.up), max_norm=1.0, error_if_nonfinite=True))
    if not math.isfinite(loss):
        raise ValueError("Nonfinite joint cache readout loss")
    optimizer.step()
    return {"loss": loss, "gradient_norm": norm, "prefix_tokens": len(prefix), "query_positions": list(positions)}
