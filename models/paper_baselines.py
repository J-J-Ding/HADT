"""Internally implemented, matched quantile controls; not official point-baseline reproductions."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.horizon_deformable import HorizonDeformableAttention


def ordered_from_raw(raw, min_gap=1e-4):
    median=raw[...,:1]
    gaps=F.softplus(raw[...,1:])+min_gap
    lower=(median-torch.cumsum(gaps[...,:3],dim=-1)).flip(-1)
    upper=median+torch.cumsum(gaps[...,3:],dim=-1)
    return torch.cat((lower,median,upper),dim=-1)


def initialize_raw_projection(layer, horizon):
    nn.init.normal_(layer.weight,mean=0.,std=1e-3)
    gap=math.log(math.expm1(.55))
    with torch.no_grad():
        bias=torch.tensor([0.,gap,gap,gap,gap,gap,gap],dtype=layer.bias.dtype)
        layer.bias.copy_(bias.repeat(horizon))


class DirectQuantileControl(nn.Module):
    def __init__(self, backbone, patch=False):
        super().__init__()
        self.seq_len=backbone.seq_len
        self.pred_len=backbone.pred_len
        self.quantiles=backbone.quantiles
        self.revin=backbone.revin
        self.patch=patch
        self.min_gap=backbone.quantile_head.min_gap
        if patch:
            self.patch_embedding=backbone.patch_embedding
            self.encoder=backbone.encoder
            self.n_patches=backbone.n_patches
            self.d_model=backbone.d_model
            input_dim=self.n_patches*self.d_model
        else:
            self.n_patches=0
            self.d_model=0
            input_dim=self.seq_len
        self.n_points=0
        self.output_projection=nn.Linear(input_dim,self.pred_len*7)
        initialize_raw_projection(self.output_projection,self.pred_len)

    def forward(self,x,return_aux=False,intervention=None):
        if return_aux or intervention is not None:
            raise ValueError('Direct controls do not expose deformable auxiliaries or interventions.')
        if x.ndim!=3 or x.shape[1]!=self.seq_len:
            raise ValueError('Unexpected input dimensions.')
        b,_,c=x.shape
        z,mean,std=self.revin.normalize(x)
        if self.patch:
            z,_,_=self.patch_embedding(z)
            features=self.encoder(z).reshape(b*c,-1)
        else:
            features=z.transpose(1,2).reshape(b*c,self.seq_len)
        raw=self.output_projection(features).reshape(b*c,self.pred_len,7)
        q=ordered_from_raw(raw,self.min_gap).reshape(b,c,self.pred_len,7).permute(0,2,1,3)
        return self.revin.denormalize(q,mean,std)


class DenseHorizonAttention(HorizonDeformableAttention):
    """Same context/query, projections and residual block; attend to every historical token."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        del self.offset_projection

    def _forward_chunk(self,query_base,key,value,reference,disable_offsets=False):
        bc,h,_=query_base.shape
        q=self.q_projection(query_base).view(bc,h,self.n_heads,self.head_dim)
        scores=torch.einsum('bhmd,bnmd->bhmn',q,key)/math.sqrt(self.head_dim)
        attention=scores.softmax(dim=-1)
        attended=torch.einsum('bhmn,bnmd->bhmd',attention,value).reshape(bc,h,self.d_model)
        attended=self.output_projection(attended)
        hidden=self.norm1(query_base+self.dropout(attended))
        out=hidden+self.dropout(self.ffn(self.norm2(hidden)))
        offsets=torch.zeros_like(attention)
        positions=torch.arange(key.shape[1],device=key.device,dtype=key.dtype).view(1,1,1,-1).expand_as(attention)
        return out,offsets,positions,attention


def make_paper_model(config,pred_len,method):
    from experiments.runner import _build_model
    model=_build_model(config,pred_len)
    if method=='hadt':
        return model
    if method in ('linear','patch'):
        return DirectQuantileControl(model,patch=method=='patch')
    if method=='fixed':
        model.ablation='no_offsets'
    elif method=='no_revin':
        model.ablation='no_revin'
    elif method=='shared':
        with torch.no_grad():model.decoder.horizon_embedding.weight.zero_()
        model.decoder.horizon_embedding.weight.requires_grad_(False)
    elif method=='independent':
        head=nn.Linear(model.d_model,7)
        nn.init.normal_(head.weight,mean=0.,std=1e-3)
        with torch.no_grad():head.bias.copy_(torch.tensor([-1.65,-1.10,-.55,0.,.55,1.10,1.65]))
        model.quantile_head=head
    elif method=='dense':
        c=config['model']
        decoder=DenseHorizonAttention(d_model=c['d_model'],n_heads=c['n_heads'],
                    n_points=model.n_patches,max_horizon=c['max_horizon'],
                    max_offset=c['max_offset'],horizon_chunk_size=c['horizon_chunk_size'],
                    dropout=c['dropout'],gradient_checkpointing=c['gradient_checkpointing'])
        decoder.load_state_dict({k:v for k,v in model.decoder.state_dict().items() if not k.startswith('offset_projection.')})
        model.decoder=decoder
        model.n_points=model.n_patches
    else:
        raise ValueError(f'Unknown paper method: {method}')
    return model
