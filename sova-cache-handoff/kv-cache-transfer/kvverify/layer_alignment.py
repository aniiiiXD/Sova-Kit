"""Measure alternate donor layers without changing the fast affine format."""

import torch

from .affine_cache import AffineCacheMapper
from .hf_backend import legacy
from .readout_mapper import capture_queries, readout_loss


def nearby_layers(aligned, source_count):
    if source_count < 3 or not aligned:
        raise ValueError("Layer alignment search requires at least three source layers")
    return [tuple(min(max(index + offset, 0), source_count - 1) for index in aligned)
            for offset in (0, -2, 2)]


@torch.inference_mode()
def functional_layer_errors(runtime, mappers, prefix, suffix):
    """Compare history attention outputs using the same native suffix queries."""
    if not suffix or len(prefix) <= 256:
        raise ValueError("Selection requires full history and a nonempty query")
    positions = tuple(sorted({0, len(suffix) // 2, len(suffix) - 1}))
    layers = range(runtime.target_geometry[2])
    source = legacy(runtime.forward(runtime.source, prefix).past_key_values)
    native_cache = runtime.forward(runtime.target, prefix).past_key_values
    with capture_queries(runtime.target, layers, positions) as captured:
        target = legacy(runtime.forward(runtime.target, suffix, native_cache).past_key_values)
    native_head = legacy(runtime.forward(runtime.target, prefix[:256]).past_key_values)
    scores = []
    for mapper in mappers:
        layer_errors = []
        for layer, source_index in enumerate(mapper.manifest["source_layers"]):
            pair = tuple(tensor[..., 256:, :] for tensor in source[source_index])
            mapped = mapper._map_layer_token_block(layer, 256, (source_index,), pair, runtime.source, runtime.target)
            hybrid = tuple(torch.cat((head, middle, end[..., len(prefix):, :]), dim=-2)
                           for head, middle, end in zip(native_head[layer], mapped, target[layer]))
            query, scale = captured[layer]
            layer_errors.append(readout_loss(query, target[layer], hybrid, prefix_tokens=len(prefix),
                                             positions=positions, scale=scale))
        scores.append(torch.stack(layer_errors))
    return torch.stack(scores)


@torch.no_grad()
def assemble_selected_layers(mappers, errors, metadata):
    if errors.ndim != 2 or errors.shape != (len(mappers), mappers[0].weight.shape[1]):
        raise ValueError("Expected one functional error per mapper and target layer")
    if not torch.isfinite(errors).all():
        raise ValueError("Nonfinite layer selection loss")
    choices = errors.argmin(0).tolist()
    source_layers = [mappers[choice].manifest["source_layers"][layer] for layer, choice in enumerate(choices)]
    common = mappers[0]
    combined = AffineCacheMapper({
        **common.manifest, **metadata, "source_layers": source_layers,
        "source_layer_groups": [[source] for source in source_layers],
        "method": "query_readout_selected_single_source_layer",
    }).to(common.weight.device)
    for layer, choice in enumerate(choices):
        combined.weight[:, layer].copy_(mappers[choice].weight[:, layer])
        combined.bias[:, layer].copy_(mappers[choice].bias[:, layer])
    return combined.eval().requires_grad_(False), choices
