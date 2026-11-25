"""Offload saved activations while retaining already-resident frozen weights.

This is an autograd memory policy, not model-weight offloading. Checkpoint's
inner saved-tensor hooks retain precedence; bounded mapper replay is separate.
"""
from contextlib import contextmanager


def _storage_key(tensor):
    """Views/transposes of a parameter share its complete underlying storage."""
    return (tensor.device.type, tensor.device.index,
            tensor.untyped_storage().data_ptr())


class _SavedTensorPolicy:
    def __init__(self, torch, models, pin_memory=False, *, offload_device_types=('cuda',)):
        self.torch = torch
        self.pin_memory = bool(pin_memory)
        # The override supports tiny CPU tests of the copy/restore branches;
        # the public context always offloads CUDA activations only.
        self.offload_device_types = frozenset(offload_device_types)
        self.weight_storages = {_storage_key(parameter)
            for model in models for parameter in model.parameters()
            if not parameter.requires_grad and parameter.numel()}
        self.stats = {'policy': 'frozen-storage-selective-activation-cpu-v1',
            'pin_memory': self.pin_memory, 'copied_bytes': 0,
            'restored_bytes': 0, 'kept_weight_bytes': 0,
            'copied_tensors': 0, 'kept_weight_tensors': 0,
            'non_offloaded_tensors': 0,
            'counter_scope': 'Cumulative saved-tensor traffic, not live or peak host memory'}

    def is_frozen_weight_storage(self, tensor):
        return bool(tensor.numel()) and _storage_key(tensor) in self.weight_storages

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        # Detach the wrapper, never alter tensor storage, dtype, values or graph.
        detached = tensor.detach()
        if self.is_frozen_weight_storage(tensor):
            self.stats['kept_weight_bytes'] += size
            self.stats['kept_weight_tensors'] += 1
            return ('resident', tensor.device, tensor.dtype, detached)
        if tensor.device.type not in self.offload_device_types:
            self.stats['non_offloaded_tensors'] += 1
            return ('resident', tensor.device, tensor.dtype, detached)
        if self.pin_memory and tensor.device.type == 'cuda':
            packed = self.torch.empty(tensor.size(), dtype=tensor.dtype,
                                      device='cpu', pin_memory=True)
            packed.copy_(detached)
        else:
            packed = detached.to(device='cpu', copy=True)
        self.stats['copied_bytes'] += size
        self.stats['copied_tensors'] += 1
        return ('cpu', tensor.device, tensor.dtype, packed)

    def unpack(self, record):
        kind, device, dtype, tensor = record
        if kind == 'resident':
            return tensor
        if kind != 'cpu':
            raise ValueError('Unknown saved-tensor storage policy')
        self.stats['restored_bytes'] += tensor.numel() * tensor.element_size()
        return tensor.to(device=device, dtype=dtype, non_blocking=self.pin_memory)


@contextmanager
def selective_saved_tensors_cpu(models, pin_memory=False):
    """Save CUDA activations on CPU, excluding frozen model-parameter storage.

    Models and their parameters must remain unchanged during the context and
    corresponding backward. CPU tensors keep their existing storage. Pageable
    CPU copies are the default to avoid excessive pinned-host allocations.
    The yielded counters measure cumulative traffic, not current RAM usage.
    No global torch state remains changed after leaving this context.
    """
    import torch
    policy = _SavedTensorPolicy(torch, tuple(models), pin_memory)
    with torch.autograd.graph.saved_tensors_hooks(policy.pack, policy.unpack):
        yield policy.stats
