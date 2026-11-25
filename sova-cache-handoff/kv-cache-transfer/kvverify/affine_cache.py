"""Centered ridge cache maps with controlled same-head/all-head support.

No training, model loading or submission at import. Calibration is supplied by
the caller; all candidates share exact samples and normalized ridge units.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from .hf_backend import legacy,unrotate


class AffineCacheMapper(nn.Module):
    def __init__(self,manifest,*,device=None):
        super().__init__();self.manifest=dict(manifest)
        h,d,l=self.manifest['target_geometry'];width=h*d
        shape=(2,l,width,width)if self.manifest['head_mixing']else(2,l,h,d,d)
        bias=(2,l,width)if self.manifest['head_mixing']else(2,l,h,d)
        self.weight=nn.Parameter(torch.empty(shape,device=device));self.bias=nn.Parameter(torch.empty(bias,device=device))

    def _map_layer_token_block(self,layer,position_offset,source_layers,tensors,source_model,target_model,*,normalize_keys=False):
        if normalize_keys:raise ValueError('This reference has no forced scalar key normalization')
        from .train_mapper import NeuralMapper
        h,d,_=self.manifest['target_geometry'];source_layer=self.manifest['source_layers'][layer]
        if tuple(source_layers)!=(source_layer,)or len(tensors)!=2:raise ValueError('One aligned source layer required')
        target_dtype=next(target_model.parameters()).dtype
        output=[]
        for side,value in enumerate(tensors):
            value=NeuralMapper._rotate_token_block(value,source_model,position_offset,True)if side==0 else value.float()
            if self.manifest['head_mixing']:
                count=value.shape[-2]
                value=value.transpose(1,2).reshape(1,count,h*d)@self.weight[side,layer]+self.bias[side,layer]
                value=value.reshape(1,count,h,d).transpose(1,2)
            else:
                value=torch.einsum('bhsd,hdf->bhsf',value,self.weight[side,layer])+self.bias[side,layer][None,:,None,:]
            if side==0:value=NeuralMapper._rotate_token_block(value,target_model,position_offset)
            output.append(value.to(target_dtype).contiguous())
        return tuple(output)

    def apply(self,source_cache,source_model,target_model):
        return self.apply_checkpointed(source_cache,source_model,target_model)

    def apply_checkpointed(self,source_cache,source_model,target_model,token_block_size=256):
        from transformers import DynamicCache
        from torch.utils.checkpoint import checkpoint
        rows=legacy(source_cache);h,d,l=self.manifest['source_geometry'];count=rows[0][0].shape[-2]
        if len(rows)!=l or count<1 or any(k.shape!=(1,h,count,d)or v.shape!=k.shape for k,v in rows):
            raise ValueError('Source cache geometry mismatch')
        result=[]
        for layer,sl in enumerate(self.manifest['source_layers']):
            parts=[]
            for start in range(0,count,token_block_size):
                pair=tuple(x[:,:,start:start+token_block_size].detach()for x in rows[sl])
                def convert(*values,_layer=layer,_start=start,_sl=sl):
                    return self._map_layer_token_block(_layer,_start,(_sl,),values,source_model,target_model)
                value=checkpoint(convert,*pair,use_reentrant=False,preserve_rng_state=False)if torch.is_grad_enabled() else convert(*pair)
                parts.append(value)
            result.append(tuple(torch.cat([p[side]for p in parts],dim=-2)for side in range(2)))
        if not all(bool(torch.isfinite(x).all())for pair in result for x in pair):raise ValueError('Nonfinite affine cache')
        return DynamicCache.from_legacy_cache(tuple(result))

    def save(self,directory):
        from safetensors.torch import save_file
        root=Path(directory);root.mkdir(parents=True,exist_ok=False)
        path=root/'weights.safetensors'
        save_file({k:v.detach().float().cpu().contiguous()for k,v in self.state_dict().items()},str(path))
        manifest={**self.manifest,'mapper_format':'affine-cache-v1','format_version':3,
            'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parameter_count':sum(p.numel()for p in self.parameters())}
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

    @classmethod
    def load(cls,directory,source_sha,target_sha,tokenizer_sha):
        from safetensors.torch import load_file
        root=Path(directory);manifest=json.loads((root/'manifest.json').read_text())
        if manifest.get('mapper_format')!='affine-cache-v1'or manifest.get('format_version')!=3:raise ValueError('Invalid affine format')
        for key,value in [('source_sha256',source_sha),('target_sha256',target_sha),('tokenizer_sha256',tokenizer_sha)]:
            if manifest.get(key)!=value:raise ValueError('Affine identity mismatch: '+key)
        path=root/'weights.safetensors'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['weights_sha256']:raise ValueError('Affine checksum mismatch')
        mapper=cls(manifest);state=load_file(str(path))
        if any(x.dtype!=torch.float32 or not torch.isfinite(x).all()for x in state.values()):raise ValueError('Invalid affine tensors')
        mapper.load_state_dict(state,strict=True);return mapper.eval()


class CrossHeadMoments:
    """One FP64 sufficient-statistic stream reused by all ridge candidates."""
    def __init__(self,geometry,source_layers,device,*,source_geometry=None):
        h,d,l=geometry;self.geometry=geometry;self.source_layers=source_layers;w=h*d
        self.source_geometry = geometry if source_geometry is None else source_geometry
        if (self.source_geometry[:2] != geometry[:2] or len(source_layers) != l
                or any(not 0 <= layer < self.source_geometry[2] for layer in source_layers)):
            raise ValueError("Invalid source layers or incompatible KV geometry")
        self.n=0
        self.sx=torch.zeros(2,l,w,dtype=torch.float64,device=device);self.sy=torch.zeros_like(self.sx)
        self.xx=torch.zeros(2,l,w,w,dtype=torch.float64,device=device);self.xy=torch.zeros_like(self.xx)

    def update(self,source,target,source_model,target_model,*,token_indices=None):
        source=legacy(source);target=legacy(target);h,d,l=self.geometry;full_count=source[0][0].shape[-2]
        if len(source)!=self.source_geometry[2] or len(target)!=l or target[0][0].shape[-2]!=full_count:raise ValueError('Calibration token alignment differs')
        indices=None;count=full_count
        if token_indices is not None:
            indices=torch.as_tensor(token_indices,device=source[0][0].device)
            if indices.ndim!=1 or indices.numel()<1 or indices.dtype not in(torch.int64,torch.int32):raise ValueError('Distinct integer calibration positions required')
            indices=indices.long();count=indices.numel()
            if int(indices.min())<0 or int(indices.max())>=full_count or indices.unique().numel()!=count:raise ValueError('Invalid or repeated calibration positions')
        for layer,sl in enumerate(self.source_layers):
            for side in range(2):
                # Undo RoPE at the ORIGINAL full-history positions before
                # subsampling; renumbering sparse cached positions corrupts K.
                x=unrotate(source[sl][side],source_model,True)if side==0 else source[sl][side]
                y=unrotate(target[layer][side],target_model,True)if side==0 else target[layer][side]
                if indices is not None:x=x.index_select(-2,indices);y=y.index_select(-2,indices)
                x=x[0].transpose(0,1).reshape(count,h*d).double();y=y[0].transpose(0,1).reshape(count,h*d).double()
                self.sx[side,layer]+=x.sum(0);self.sy[side,layer]+=y.sum(0)
                self.xx[side,layer]+=x.T@x;self.xy[side,layer]+=x.T@y
        self.n+=count

    @torch.inference_mode(False)
    @torch.no_grad()
    def fit(self,alpha,head_mixing,manifest,*,relative_regularization=False):
        if not math.isfinite(alpha)or alpha<=0 or self.n<2:raise ValueError('Positive normalized ridge and sufficient samples required')
        h,d,l=self.geometry
        metadata={**manifest,'source_geometry':list(self.source_geometry),'target_geometry':list(self.geometry),
            'source_layers':list(self.source_layers),'source_layer_groups':[[x]for x in self.source_layers],
            'head_mixing':bool(head_mixing),'key_rms_normalization':False,'alpha':alpha,
            'regularization': ('Centered XTX/n + alpha*diag(feature variance), with a variance floor'
                               if relative_regularization else
                               'Centered XTX/n + alpha*I; equivalent raw-sum lambda=n*alpha'),
            'raw_sum_lambda_equivalent':None if relative_regularization else self.n*alpha,
            'calibration_tokens':self.n,'trained_steps':0,
            'method':'centered_all_head_affine_ridge'if head_mixing else'centered_same_head_affine_ridge'}
        mapper=AffineCacheMapper(metadata)
        for side in range(2):
            for layer in range(l):
                mx=self.sx[side,layer]/self.n;my=self.sy[side,layer]/self.n
                cov=self.xx[side,layer]/self.n-mx[:,None]*mx[None,:]
                cross=self.xy[side,layer]/self.n-mx[:,None]*my[None,:]
                if not head_mixing:
                    index=torch.arange(h,device=cov.device)
                    cov=cov.reshape(h,d,h,d).permute(0,2,1,3)[index,index]
                    cross=cross.reshape(h,d,h,d).permute(0,2,1,3)[index,index]
                    mx=mx.reshape(h,d);my=my.reshape(h,d)
                if relative_regularization:
                    variances = cov.diagonal(dim1=-2, dim2=-1)
                    floor = (variances.mean(dim=-1, keepdim=True) * 1e-6).clamp_min(1e-12)
                    penalty = torch.diag_embed(variances.clamp_min(floor))
                else:
                    penalty = torch.eye(cov.shape[-1],dtype=torch.float64,device=cov.device)
                weight=torch.linalg.solve(cov+alpha*penalty,cross)
                bias=my-(mx@weight if head_mixing else torch.einsum('hi,hij->hj',mx,weight))
                if not torch.isfinite(weight).all()or not torch.isfinite(bias).all():raise ValueError('Nonfinite ridge solution')
                with torch.no_grad():
                    mapper.weight[side,layer].copy_(weight.float().cpu());mapper.bias[side,layer].copy_(bias.float().cpu())
        return mapper.eval().requires_grad_(False)
