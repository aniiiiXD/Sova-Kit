"""Inference-only cache maps with reusable rotary tables and donor features."""
import torch

from .hf_backend import legacy


def _rotary_tables(model, positions, sample):
    cos, sin = model.model.rotary_emb(sample[:, 0].float(), positions)
    return cos.unsqueeze(1).float(), sin.unsqueeze(1).float()


def _rotate(value, tables, inverse=False):
    cos, sin = tables
    value = value.float()
    left, right = value.chunk(2, dim=-1)
    cross = torch.cat((-right, left), dim=-1)
    return value * cos + cross * sin * (-1 if inverse else 1)


@torch.inference_mode()
def map_affine_fast(mapper, source_cache, source_model, target_model, *, block_tokens=256):
    """Reuse RoPE tables while keeping the reference projection block size by default."""
    from transformers import DynamicCache

    manifest = mapper.manifest
    source_heads, source_dim, source_layers = manifest["source_geometry"]
    heads, dim, layers = manifest["target_geometry"]
    indices = manifest["source_layers"]
    if (not manifest["head_mixing"] or (source_heads, source_dim) != (heads, dim)
            or len(indices) != layers or any(not 0 <= i < source_layers for i in indices)
            or type(block_tokens) is not int or block_tokens <= 0):
        raise ValueError("Fast mapper requires aligned full-head affine geometry")
    rows = legacy(source_cache)
    count = rows[0][0].shape[-2]
    target = next(target_model.parameters())
    device = target.device
    if (len(rows) != source_layers or count < 1 or mapper.weight.device != device
            or mapper.weight.dtype != torch.float32 or mapper.bias.dtype != torch.float32
            or any(k.shape != (1, heads, count, dim) or k.device != device or v.device != device
                   for k, v in rows)):
        raise ValueError("Fast mapper requires same-device source cache and FP32 weights")
    positions = torch.arange(count, device=device).unsqueeze(0)
    source_tables = _rotary_tables(source_model, positions, rows[0][0][:, :, :1])
    target_tables = _rotary_tables(target_model, positions, rows[0][0][:, :, :1])

    result, finite = [], []
    width = heads * dim
    # The source mapping is often repeated in adjacent target layers. Keep
    # one source layer in FP32 rather than retaining an entire duplicate cache.
    previous = -1
    for layer, source_layer in enumerate(indices):
        if source_layer != previous:
            key, value = rows[source_layer]
            prepared_key = _rotate(key, source_tables, inverse=True).transpose(1, 2).reshape(count, width)
            prepared_value = value.float().transpose(1, 2).reshape(count, width)
            previous = source_layer
        pair = []
        for side, prepared in enumerate((prepared_key, prepared_value)):
            weight, bias = mapper.weight[side, layer], mapper.bias[side, layer]
            if count <= block_tokens:
                values = prepared @ weight + bias
            else:
                values = torch.empty_like(prepared)
                for start in range(0, count, block_tokens):
                    end = min(count, start + block_tokens)
                    values[start:end] = prepared[start:end] @ weight + bias
            values = values.reshape(1, count, heads, dim).transpose(1, 2)
            if side == 0:
                values = _rotate(values, target_tables)
            values = values.to(target.dtype).contiguous()
            finite.append(torch.isfinite(values).all())
            pair.append(values)
        result.append(tuple(pair))
    if not bool(torch.stack(finite).all()):
        raise ValueError("Nonfinite affine cache")
    return DynamicCache.from_legacy_cache(tuple(result))


@torch.inference_mode()
def map_multi_donor_fast(mapper, source_cache, source_model, target_model, *, block_tokens=4096):
    """Evaluate the trained residual while reusing adjacent layers' prepared donors."""
    from transformers import DynamicCache

    rows = legacy(source_cache)
    heads, dim, target_layers = mapper.base.manifest["target_geometry"]
    _, _, source_layers = mapper.base.manifest["source_geometry"]
    target = next(target_model.parameters())
    if len(rows) != source_layers or not rows or type(block_tokens) is not int or block_tokens < 1:
        raise ValueError("Invalid source cache or block size")
    count = rows[0][0].shape[-2]
    shape = (1, heads, count, dim)
    weights = (mapper.base.weight, mapper.base.bias, mapper.down, mapper.up)
    if (count < 1 or len(mapper.groups) != target_layers
            or any(k.shape != shape or v.shape != shape or k.device != target.device or v.device != target.device
                   for k, v in rows)
            or any(w.device != target.device or w.dtype != torch.float32 for w in weights)):
        raise ValueError("Fast mapper requires compatible source caches and same-device FP32 weights")

    positions = torch.arange(count, device=target.device).unsqueeze(0)
    source_tables = _rotary_tables(source_model, positions, rows[0][0][:, :, :1])
    cos, sin = _rotary_tables(target_model, positions, rows[0][0][:, :, :1])
    base_weight, bias, down, up = weights
    prepared = {}
    result = []
    for layer, donors in enumerate(mapper.groups):
        # Keep at most one donor group in FP32, including when alignment jumps.
        prepared = {donor: prepared[donor] for donor in donors if donor in prepared}
        for donor in donors:
            if donor not in prepared:
                key, value = rows[donor]
                prepared[donor] = (
                    _rotate(key, source_tables, inverse=True).transpose(1, 2).reshape(count, heads * dim),
                    value.float().transpose(1, 2).reshape(count, heads * dim),
                )

        pair = []
        for side in range(2):
            output = torch.empty(shape, device=target.device, dtype=target.dtype)
            features = [prepared[donor][side] for donor in donors]
            donor_weights = down[layer].unbind()
            for start in range(0, count, block_tokens):
                end = min(count, start + block_tokens)
                aligned = features[0][start:end]
                values = aligned @ base_weight[side, layer]
                values.add_(bias[side, layer])
                residual = aligned @ donor_weights[0]
                for donor_features, weight in zip(features[1:], donor_weights[1:]):
                    residual.add_(donor_features[start:end] @ weight)
                values.add_(residual @ up[side, layer])
                values = values.reshape(1, end - start, heads, dim).transpose(1, 2)
                if side == 0:
                    values = _rotate(values, (cos[:, :, start:end], sin[:, :, start:end]))
                output[:, :, start:end].copy_(values)
            pair.append(output)
        result.append(tuple(pair))
    return DynamicCache.from_legacy_cache(tuple(result))
