"""Train a pair-specific nonlinear KV mapper while keeping both LMs frozen.

This module is imported only for explicitly scoped Torch work. Training matches
native target distributions on unlabeled context windows; it never treats a
cropped document's original answer as a valid training label. Depth neighbors
are concatenated (fixed top-k, not a learned layer search). The cache adapter
is experimental and does not reproduce any vendor's private implementation.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .data import assert_disjoint, json_digest
from .hf_backend import RidgeMapper, dependencies, legacy, unrotate


class NeuralMapper(nn.Module):
    """Trainable affine + nonlinear cross-head/depth residual, with key RMS.

    The base affine maps are initialized from a ridge fit. The residual reads
    three nearby source layers and all KV heads, so information need not stay
    in the same head. Key normalization prevents a regressor's shrunken keys
    from silently flattening target attention. Target RoPE is applied last.
    """
    def __init__(self, manifest, ridge_tensors=None):
        super().__init__()
        self.manifest = dict(manifest)
        heads, dim, layers = self.manifest['target_geometry']
        self.rank = int(self.manifest['rank'])
        width = heads * dim
        count = len(self.manifest['source_layer_groups'][0])
        self.affine_weight = nn.Parameter(torch.empty(2, layers, heads, dim, dim))
        self.affine_bias = nn.Parameter(torch.zeros(2, layers, heads, dim))
        self.down = nn.Parameter(torch.randn(2, layers, count*width, self.rank) / math.sqrt(count*width))
        self.down_bias = nn.Parameter(torch.zeros(2, layers, self.rank))
        self.up = nn.Parameter(torch.zeros(2, layers, self.rank, width))
        self.up_bias = nn.Parameter(torch.zeros(2, layers, width))
        self.key_log_rms = nn.Parameter(torch.zeros(layers, heads))
        with torch.no_grad():
            if ridge_tensors is None:
                self.affine_weight.copy_(torch.eye(dim).repeat(2, layers, heads, 1, 1))
            else:
                for kind, name in enumerate(('k', 'v')):
                    self.affine_weight[kind].copy_(ridge_tensors[name+'_weight'].detach().float().cpu())
                    self.affine_bias[kind].copy_(ridge_tensors[name+'_bias'].detach().float().cpu())

    @classmethod
    def from_ridge(cls, ridge, target_model, rank=16, topk=3):
        if rank < 1 or topk < 1:
            raise ValueError('Positive residual rank and top-k required')
        manifest = dict(ridge.manifest)
        source_layers = manifest['source_geometry'][2]
        groups = []
        for aligned in manifest['source_layers']:
            nearby = sorted(range(source_layers), key=lambda i: (abs(i-aligned), i))[:topk]
            # Fixed-size branch even for tiny one-layer smoke models.
            groups.append(nearby + [nearby[-1]] * (topk-len(nearby)))
        manifest.update(method='ridge_initialized_topk_cross_head_mlp_distillation',
                        mapper_format='neural-v1', rank=rank, topk=topk,
                        source_layer_groups=groups, key_rms_normalization=True,
                        limitations='Fixed nearest-depth top-k, not learned layer search; standard full RoPE only; no hybrid state transfer.')
        mapper = cls(manifest, ridge.tensors)
        with torch.no_grad():
            for layer in range(manifest['target_geometry'][2]):
                norm = target_model.model.layers[layer].self_attn.k_norm.weight.detach().float()
                mapper.key_log_rms[layer].fill_(float(norm.square().mean().sqrt().clamp_min(1e-6).log()))
        return mapper

    def apply(self, source_cache, source_model, target_model, *, normalize_keys=True):
        _, DynamicCache = dependencies()
        rows = legacy(source_cache)
        heads, dim, layers = self.manifest['target_geometry']
        source_heads, source_dim, source_layers = self.manifest['source_geometry']
        if len(rows) != source_layers or (source_heads, source_dim) != (heads, dim):
            raise ValueError('Source KV geometry changed')
        seq = rows[0][0].shape[-2]
        if seq < 1 or any(k.shape != (1, heads, seq, dim) or v.shape != k.shape for k, v in rows):
            raise ValueError('Unsupported source KV batch/shape')
        device = next(target_model.parameters()).device
        dtype = next(target_model.parameters()).dtype
        if next(self.parameters()).device != device:
            raise ValueError('Place mapper on target device before applying')
        # A caller may supply inference-mode tensors. Clone outside that mode
        # before autograd saves them for a mapper parameter's backward pass.
        with torch.inference_mode(False):
            prepared = [(unrotate(k.detach().clone(), source_model, True).to(device),
                         v.detach().clone().float().to(device)) for k, v in rows]
        result, finite = [], []
        for layer in range(layers):
            aligned = self.manifest['source_layers'][layer]
            output = []
            for kind in range(2):
                x = prepared[aligned][kind]
                value = torch.einsum('bhsd,hdf->bhsf', x, self.affine_weight[kind, layer])
                value = value + self.affine_bias[kind, layer][None, :, None, :]
                features = torch.cat([prepared[i][kind].transpose(1, 2).reshape(1, seq, heads*dim)
                                      for i in self.manifest['source_layer_groups'][layer]], dim=-1)
                features = F.layer_norm(features, (features.shape[-1],))
                residual = F.silu(features @ self.down[kind, layer] + self.down_bias[kind, layer])
                residual = residual @ self.up[kind, layer] + self.up_bias[kind, layer]
                value = value + residual.reshape(1, seq, heads, dim).transpose(1, 2)
                if kind == 0:
                    if normalize_keys:
                        rms = value.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                        value = value / rms * self.key_log_rms[layer].clamp(-6, 6).exp()[None, :, None, None]
                    value = unrotate(value, target_model)
                value = value.to(dtype).contiguous()
                finite.append(torch.isfinite(value).all())
                output.append(value)
            result.append(tuple(output))
        if not torch.stack(finite).all():
            raise ValueError('Nonfinite mapped cache after target-dtype conversion')
        return DynamicCache.from_legacy_cache(tuple(result))

    @staticmethod
    def _rotate_token_block(keys, model, position_offset, inverse=False):
        """The same full-RoPE math as unrotate, at absolute prefix positions."""
        positions = torch.arange(position_offset,position_offset+keys.shape[-2],
                                 device=keys.device).unsqueeze(0)
        cos,sin = model.model.rotary_emb(keys[:,0].float(),positions)
        cos,sin = cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
        value=keys.float()
        first,second=value.chunk(2,dim=-1)
        rotated_half=torch.cat((-second,first),dim=-1)
        return value*cos+rotated_half*sin*(-1 if inverse else 1)

    def _map_layer_token_block(self,layer,position_offset,source_layers,tensors,
                               source_model,target_model,*,normalize_keys=True):
        """Map only one target layer/block; no full-cache preparation list."""
        heads,dim,_=self.manifest['target_geometry']
        device=next(target_model.parameters()).device
        dtype=next(target_model.parameters()).dtype
        prepared={}
        for index,source_layer in enumerate(source_layers):
            key,value=tensors[2*index:2*index+2]
            prepared[source_layer]=(self._rotate_token_block(key,source_model,position_offset,True).to(device),
                                    value.float().to(device))
        aligned=self.manifest['source_layers'][layer]
        count=tensors[0].shape[-2]
        output=[]
        for kind in range(2):
            x=prepared[aligned][kind]
            value=torch.einsum('bhsd,hdf->bhsf',x,self.affine_weight[kind,layer])
            value=value+self.affine_bias[kind,layer][None,:,None,:]
            features=torch.cat([prepared[i][kind].transpose(1,2).reshape(1,count,heads*dim)
                                for i in self.manifest['source_layer_groups'][layer]],dim=-1)
            features=F.layer_norm(features,(features.shape[-1],))
            residual=F.silu(features@self.down[kind,layer]+self.down_bias[kind,layer])
            residual=residual@self.up[kind,layer]+self.up_bias[kind,layer]
            value=value+residual.reshape(1,count,heads,dim).transpose(1,2)
            if kind==0:
                if normalize_keys:
                    rms=value.square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
                    value=value/rms*self.key_log_rms[layer].clamp(-6,6).exp()[None,:,None,None]
                value=self._rotate_token_block(value,target_model,position_offset)
            output.append(value.to(dtype).contiguous())
        return tuple(output)

    def apply_checkpointed(self,source_cache,source_model,target_model,token_block_size=256):
        """Exact mapping with per-layer/token-block activation checkpoints.

        Raw frozen source caches are shared by reference. Only the selected
        source layers' current token block is de-RoPE'd or cloned; backward
        recomputes that block, never all target layers or all prefix positions.
        Absolute RoPE offsets preserve the original full-prefix function.
        """
        from torch.utils.checkpoint import checkpoint
        _,DynamicCache=dependencies()
        if not isinstance(token_block_size,int) or token_block_size<1:
            raise ValueError('Positive mapper token block size required')
        rows=legacy(source_cache)
        heads,dim,layers=self.manifest['target_geometry']
        sh,sd,sl=self.manifest['source_geometry']
        seq=rows[0][0].shape[-2]
        if (sh,sd)!=(heads,dim) or len(rows)!=sl or seq<1 or any(
                k.shape!=(1,heads,seq,dim) or v.shape!=k.shape for k,v in rows):
            raise ValueError('Unsupported source KV geometry for checkpointed mapping')
        device=next(target_model.parameters()).device
        if next(self.parameters()).device!=device:
            raise ValueError('Place mapper on target device before applying')
        need_grad=torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        result,finite=[],[]
        for layer in range(layers):
            needed=tuple(sorted({self.manifest['source_layers'][layer],
                                 *self.manifest['source_layer_groups'][layer]}))
            blocks=[]
            for offset in range(0,seq,token_block_size):
                # Only views of the selected source slices are saved as
                # checkpoint inputs. Inference-only inputs need small normal
                # clones; ordinary no-grad source caches require no clone.
                with torch.inference_mode(False):
                    tensors=tuple(t[:,:,offset:offset+token_block_size].detach().clone()
                                  if torch.is_inference(t) else t[:,:,offset:offset+token_block_size].detach()
                                  for index in needed for t in rows[index])
                def convert(*values,_layer=layer,_offset=offset,_needed=needed):
                    return self._map_layer_token_block(_layer,_offset,_needed,values,source_model,target_model)
                mapped=checkpoint(convert,*tensors,use_reentrant=False) if need_grad else convert(*tensors)
                finite.extend(torch.isfinite(value).all() for value in mapped)
                blocks.append(mapped)
            result.append(tuple(torch.cat([block[kind] for block in blocks],dim=-2) for kind in range(2)))
        if not torch.stack(finite).all():
            raise ValueError('Nonfinite mapped cache after target-dtype conversion')
        return DynamicCache.from_legacy_cache(tuple(result))

    def save(self, directory):
        from safetensors.torch import save_file
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        weights = root/'weights.safetensors'
        save_file({name: value.detach().float().cpu().contiguous()
                   for name, value in self.state_dict().items()}, str(weights))
        manifest = {**self.manifest, 'format_version': 2,
                    'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                    'parameter_count': sum(p.numel() for p in self.parameters())}
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        return manifest

    @classmethod
    def load(cls, directory, source_sha, target_sha, tokenizer_sha):
        from safetensors.torch import load_file
        root = Path(directory)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('format_version') != 2 or manifest.get('mapper_format') != 'neural-v1':
            raise ValueError('Unsupported neural mapper format')
        for field, expected in [('source_sha256', source_sha), ('target_sha256', target_sha),
                                ('tokenizer_sha256', tokenizer_sha)]:
            if manifest.get(field) != expected:
                raise ValueError('Mapper checkpoint/tokenizer mismatch: '+field)
        if hashlib.sha256((root/'weights.safetensors').read_bytes()).hexdigest() != manifest['weights_sha256']:
            raise ValueError('Mapper weights checksum mismatch')
        h, d, layers = manifest['target_geometry']
        sh, sd, source_layers = manifest['source_geometry']
        if (h, d) != (sh, sd) or not 1 <= manifest['rank'] <= 1024 or not 1 <= manifest['topk'] <= 8:
            raise ValueError('Invalid mapper architecture')
        if len(manifest['source_layers']) != layers or len(manifest['source_layer_groups']) != layers:
            raise ValueError('Invalid mapper layer count')
        for group in manifest['source_layer_groups']:
            if len(group) != manifest['topk'] or any(type(i) is not int or not 0 <= i < source_layers for i in group):
                raise ValueError('Invalid mapper source layer group')
        if any(type(i) is not int or not 0 <= i < source_layers for i in manifest['source_layers']):
            raise ValueError('Invalid mapper depth alignment')
        mapper = cls(manifest)
        state = load_file(str(root/'weights.safetensors'))
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Invalid mapper parameters')
        mapper.load_state_dict(state, strict=True)
        return mapper.eval()


def model_forward(model, ids, cache=None, logits_to_keep=1):
    """Differentiable target forward; DenseRuntime.forward is inference-only."""
    device = next(model.parameters()).device
    offset = cache.get_seq_length() if cache is not None else 0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model(input_ids=x, past_key_values=cache, use_cache=True,
                 attention_mask=torch.ones((1, offset+len(ids)), dtype=torch.long, device=device),
                 position_ids=torch.arange(offset, offset+len(ids), device=device).unsqueeze(0),
                 cache_position=torch.arange(offset, offset+len(ids), device=device),
                 logits_to_keep=logits_to_keep, return_dict=True)


def training_loss(runtime, mapper, ids, tail_tokens=8, kv_weight=1., kl_weight=1., temperature=1.):
    """Recompute frozen teacher/source; retain gradients only through mapped tail.

    ids is one context window with at least one prefix and one tail token. The
    last tail_tokens causal positions supervise next-token distributions, not
    answer labels. KV reconstruction is averaged equally across target layers.
    """
    if tail_tokens < 1 or len(ids) <= tail_tokens or not temperature > 0:
        raise ValueError('A nonempty prefix/tail and positive temperature are required')
    prefix_length = len(ids)-tail_tokens
    with torch.no_grad():
        source = model_forward(runtime.source, ids[:prefix_length]).past_key_values
        teacher = model_forward(runtime.target, ids, logits_to_keep=tail_tokens)
        teacher_logits = teacher.logits.detach().float()
        target_rows = tuple((k[:, :, :prefix_length].detach(), v[:, :, :prefix_length].detach())
                            for k, v in legacy(teacher.past_key_values))
    mapped = mapper.apply(source, runtime.source, runtime.target)
    reconstruction = torch.zeros((), device=teacher_logits.device)
    for predicted, target in zip(legacy(mapped), target_rows):
        for actual, expected in zip(predicted, target):
            expected = expected.float()
            scale = expected.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            reconstruction = reconstruction + ((actual.float()-expected).square()/scale).mean()
    reconstruction = reconstruction/(2*len(target_rows))
    del teacher, target_rows, source
    student = model_forward(runtime.target, ids[prefix_length:], mapped, logits_to_keep=tail_tokens)
    predicted_logp = F.log_softmax(student.logits[0].float()/temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits[0]/temperature, dim=-1)
    kl = F.kl_div(predicted_logp, teacher_p, reduction='batchmean') * temperature**2
    loss = kv_weight*reconstruction + kl_weight*kl
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training objective')
    metrics = {'loss': float(loss.detach()), 'kv_loss': float(reconstruction.detach()),
               'kl_loss': float(kl.detach()), 'prefix_tokens': prefix_length, 'tail_tokens': tail_tokens}
    return loss, metrics


def _unique_documents(examples):
    result, contexts = {}, {}
    for example in examples:
        if example.document_id in result and result[example.document_id].context_hash != example.context_hash:
            raise ValueError('A training document ID refers to different contexts')
        if example.context_hash in contexts and contexts[example.context_hash] != example.document_id:
            raise ValueError('Duplicate training contexts must share a document ID')
        result.setdefault(example.document_id, example)
        contexts[example.context_hash] = example.document_id
    return list(result.values())


class MapperGradientOverflow(RuntimeError):
    """A bounded update failed without being counted as an optimizer update."""
    def __init__(self, details):
        super().__init__('Mapper gradient update failed: '+details['reason'])
        self.details = details


def bounded_optimizer_update(loss_fn, mapper, optimizer, scaler, *, max_attempts=8,
                             min_loss_scale=1., deadline=None, on_overflow=None):
    """Replay one fixed loss closure with adaptive CUDA scale on inf/nan.

    Only a finite optimizer update returns success. Overflow attempts call the
    public GradScaler step/update protocol, which skips optimizer mutation and
    reduces the scale, then recompute the *same* token window. A finite-gradient
    but nonfinite norm is a different failure: never call scaler.step there,
    since its found-inf flag would be false and it could update the parameters.
    """
    if max_attempts < 1 or min_loss_scale <= 0:
        raise ValueError('Positive bounded retry count and minimum scale required')
    parameters = list(mapper.parameters())
    for attempt in range(1, max_attempts+1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Mapper training time budget reached before optimizer update')
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_fn()
        loss_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError('Training objective has no mapper gradients')
        finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
        if not finite:
            details = {'attempt': attempt, 'max_attempts': max_attempts,
                       'loss_scale_before': loss_scale, 'gradients_finite': False, **metrics}
            if not scaler.is_enabled():
                optimizer.zero_grad(set_to_none=True)
                raise MapperGradientOverflow({**details, 'reason': 'nonfinite_gradients_without_cuda_scaling'})
            # unscale_ has recorded actual per-gradient inf/nan. Therefore
            # GradScaler.step skips the optimizer entirely (including Adam
            # state and decoupled weight decay); update applies backoff.
            scaler.step(optimizer)
            scaler.update()
            reduced = float(scaler.get_scale())
            if reduced < min_loss_scale:
                scaler.update(new_scale=min_loss_scale)
                reduced = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            details.update(loss_scale_after=reduced, optimizer_updated=False)
            if on_overflow:
                on_overflow(details)
            if loss_scale <= min_loss_scale or not math.isfinite(reduced) or reduced >= loss_scale:
                raise MapperGradientOverflow({**details, 'reason': 'nonfinite_gradients_at_minimum_or_nonreducing_scale'})
            if attempt == max_attempts:
                raise MapperGradientOverflow({**details, 'reason': 'nonfinite_gradients_retry_limit'})
            continue
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=False)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise MapperGradientOverflow({'attempt': attempt, 'max_attempts': max_attempts,
                'loss_scale_before': loss_scale, 'gradients_finite': True,
                'reason': 'nonfinite_norm_with_finite_gradients', **metrics})
        scaler.step(optimizer)
        scaler.update()
        return {**metrics, 'grad_norm': float(norm), 'loss_scale': loss_scale,
                'loss_scale_after': scaler.get_scale(), 'gradients_finite': True,
                'optimizer_attempts': attempt, 'overflow_attempts': attempt-1}
    raise AssertionError('Unreachable bounded optimizer loop')


def train_mapper(runtime, training_examples, validation_examples, output, *, steps=400,
                 prefix_tokens=1024, tail_tokens=8, rank=16, lr=3e-4,
                 kv_weight=1., kl_weight=1., checkpoint_every=50, callback=None,
                 max_seconds=5400, seed=19):
    """Train under caller's approved runtime budget and save validation-best.

    Output: best/{manifest.json,weights.safetensors}, checkpoints/step-N/,
    training-log.jsonl, training-summary.json. Validation selects checkpoints
    only from a document-disjoint training-domain split; official dev/test must
    never be passed here. Both LMs stay in eval mode with requires_grad=False.
    Returns the selected NeuralMapper and assigns it to runtime.mapper.
    """
    if steps < 1 or prefix_tokens < 2 or tail_tokens < 1 or checkpoint_every < 1 or max_seconds <= 0 or lr <= 0:
        raise ValueError('Invalid bounded training configuration')
    if kv_weight < 0 or kl_weight < 0 or kv_weight+kl_weight <= 0:
        raise ValueError('Positive combined objective weight required')
    if any(e.split != 'calibration' for e in [*training_examples, *validation_examples]):
        raise ValueError('Mapper training/selection accept calibration splits only, never eval')
    train = _unique_documents(training_examples)
    validation = _unique_documents(validation_examples)
    if not train or not validation:
        raise ValueError('Nonempty training and independent mapper validation required')
    assert_disjoint(train, validation)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    started = time.perf_counter()
    device = next(runtime.target.parameters()).device
    for model in (runtime.source, runtime.target):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    # Fit only on deduplicated training contexts. An existing ridge is accepted
    # only if its checkpoint identities and data registries match this split.
    ridge = runtime.mapper
    if not isinstance(ridge, RidgeMapper):
        ridge = runtime.fit(train, prefix_tokens=prefix_tokens)
    for field, expected in [('source_sha256', runtime.identities['source']['sha256']),
                            ('target_sha256', runtime.identities['target']['sha256']),
                            ('tokenizer_sha256', runtime.tokenizer_sha256)]:
        if ridge.manifest.get(field) != expected:
            raise ValueError('Warm-start mapper identity mismatch: '+field)
    from .data import check_heldout
    check_heldout(ridge.manifest, validation)
    if set(ridge.manifest['calibration_documents']) - {e.document_id for e in train}:
        raise ValueError('Warm-start mapper includes documents outside the training split')
    mapper = NeuralMapper.from_ridge(ridge, runtime.target, rank=rank).to(device)
    tokenized = [(example, runtime.tokenize(example)) for example in train]
    val_tokenized = [(example, runtime.tokenize(example)) for example in validation]
    if any(len(ids) < tail_tokens+2 for _, ids in tokenized+val_tokenized):
        raise ValueError('A document is too short for the declared training tail')
    registry = [*train, *validation]
    mapper.manifest.update(
        calibration_ids=sorted({e.id for e in registry}),
        calibration_documents=sorted({e.document_id for e in registry}),
        calibration_context_hashes=sorted({e.context_hash for e in registry}),
        training_documents=sorted(e.document_id for e in train),
        mapper_validation_documents=sorted(e.document_id for e in validation),
        training_input_sha256=json_digest([asdict(e) for e in train]),
        validation_input_sha256=json_digest([asdict(e) for e in validation]),
        training_objective='Full-vocabulary native-target KL at causal tail positions + normalized KV MSE; no gold answer loss',
        window_policy='Deduplicate documents first. Random contiguous token windows of formatted document+question; positions restart at zero; every fourth step uses prompt-ending window. No label validity assumed after cropping.',
        prefix_tokens=prefix_tokens, tail_tokens=tail_tokens, seed=seed,
        lr=lr, kv_weight=kv_weight, kl_weight=kl_weight,
        thesis_verified=False)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=.001)
    # Gradients through a long FP16 cache may be below the smallest half
    # subnormal even when the aggregate loss is healthy. Scale before that
    # cast's backward and unscale FP32 mapper gradients before clipping.
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=1024.)
    mapper.manifest['gradient_scaling'] = {'cuda_enabled': device.type == 'cuda', 'initial_scale': 1024.,
        'max_attempts_per_update': 8, 'minimum_scale': 1.,
        'nonfinite_policy': 'Replay same token window with scaler backoff; log skipped optimizer attempts; fail closed at retry/scale limit. Only actual finite optimizer updates count.'}
    history = []
    best = {'loss': math.inf, 'step': 0, 'state': None}

    def window(ids, randomize):
        length = min(len(ids), prefix_tokens+tail_tokens)
        start = rng.randrange(len(ids)-length+1) if randomize else len(ids)-length
        return ids[start:start+length], start

    def validate(step):
        mapper.eval()
        measurements = []
        with torch.no_grad():
            for _, ids in val_tokenized:
                selected, _ = window(ids, False)
                _, metrics = training_loss(runtime, mapper, selected, tail_tokens, kv_weight, kl_weight)
                measurements.append(metrics)
        score = sum(row['loss'] for row in measurements)/len(measurements)
        if score < best['loss']:
            best.update(loss=score, step=step,
                        state={key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()})
        mapper.manifest.update(trained_steps=step, selected_at_step=step)
        path = root/'checkpoints'/f'step-{step:06d}'
        mapper.save(path)
        row = {'step': step, 'validation_loss': score, 'best_step': best['step'],
               'best_validation_loss': best['loss'], 'elapsed_seconds': time.perf_counter()-started,
               'checkpoint_path': str(path), 'validation_documents': len(validation),
               'loss_scale': scaler.get_scale(), 'gradients_finite': True if step else None}
        with (root/'training-log.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if callback:
            callback(row)
        mapper.train()
        return row

    validate(0)
    completed = 0
    overflow_attempts = 0
    for step in range(1, steps+1):
        if time.perf_counter()-started > max_seconds:
            break
        example, ids = tokenized[rng.randrange(len(tokenized))]
        selected, start = window(ids, step % 4 != 0)

        def record_overflow(details):
            nonlocal overflow_attempts
            overflow_attempts += 1
            event = {'status': 'gradient_overflow_retry', 'step': completed,
                'attempted_step': step, 'completed_updates': completed,
                'document_id': example.document_id, 'window_start': start,
                'window_length': len(selected), 'elapsed_seconds': time.perf_counter()-started,
                **details}
            with (root/'training-log.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print('mapper_overflow '+json.dumps(event), flush=True)
            if callback:
                callback(event)

        try:
            metrics = bounded_optimizer_update(
                lambda: training_loss(runtime, mapper, selected, tail_tokens, kv_weight, kl_weight),
                mapper, optimizer, scaler, deadline=started+max_seconds, on_overflow=record_overflow)
        except TimeoutError:
            break
        except MapperGradientOverflow as exc:
            failure = {'status': 'nonfinite_gradients', 'attempted_step': step,
                       'completed_updates': completed, 'overflow_attempts_total': overflow_attempts,
                       **exc.details}
            (root/'training-failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            if callback:
                callback(failure)
            raise
        completed = step
        row = {'step': step, **metrics, 'document_id': example.document_id,
               'window_start': start, 'window_length': len(selected),
               'elapsed_seconds': time.perf_counter()-started}
        history.append(row)
        with (root/'training-log.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print('mapper_train '+json.dumps(row), flush=True)
        if step % checkpoint_every == 0 or step == steps:
            validate(step)
    if completed % checkpoint_every and completed != steps:
        validate(completed)
    mapper.load_state_dict(best['state'])
    mapper.eval()
    mapper.manifest.update(trained_steps=completed, best_step=best['step'],
                           best_validation_loss=best['loss'], selected_at_step=best['step'])
    mapper.save(root/'best')
    summary = {'status': 'completed' if completed == steps else 'time_budget_reached',
               'requested_steps': steps, 'completed_steps': completed, 'best_step': best['step'],
               'best_validation_loss': best['loss'], 'elapsed_seconds': time.perf_counter()-started,
               'training_documents': len(train), 'mapper_validation_documents': len(validation),
               'base_models_frozen': all(not p.requires_grad and p.grad is None for m in (runtime.source, runtime.target) for p in m.parameters()),
               'parameter_count': sum(p.numel() for p in mapper.parameters()),
               'loss_scale': scaler.get_scale(), 'completed_updates': completed,
               'overflow_attempts': overflow_attempts,
               'forward_backward_attempts': completed+overflow_attempts,
               'best_mapper': str(root/'best'), 'thesis_verified': False}
    (root/'training-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    runtime.mapper = mapper
    if callback:
        callback(summary)
    return mapper
