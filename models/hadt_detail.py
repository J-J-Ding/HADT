"""Experimental HADT variants. Original HADTNQ and historical defaults stay intact."""
import math
import torch
from torch import nn
from models.hadt_nq import HADTNQ
from layers.hadt_nq.residual_encoder import ResidualTokenBlock
from layers.hadt_nq.local_detail import LocalResidualBlock, RecentGridAttention


class DetailHADTNQ(HADTNQ):
    def __init__(self, encoder_kind='mlp', local_kernel=3, grid_power=1.0,
                 median_skip='none', skip_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        width=self.d_model;hidden=kwargs.get('d_ff',256);drop=kwargs.get('dropout',0.1)
        depth=kwargs.get('e_layers',3)
        if encoder_kind in ('mlp','local','dilated'):
            blocks=[]
            for i in range(depth):
                if encoder_kind=='mlp':blocks.append(ResidualTokenBlock(width,hidden,drop))
                else:blocks.append(LocalResidualBlock(width,hidden,drop,local_kernel,2**i if encoder_kind=='dilated' else 1))
            self.encoder=nn.Sequential(*blocks,nn.LayerNorm(width))
        elif encoder_kind=='identity':self.encoder=nn.Identity()
        elif encoder_kind!='transformer':raise ValueError('Unknown encoder_kind')
        if grid_power!=1:
            old=self.decoder
            decoder=RecentGridAttention(grid_power=grid_power,d_model=width,n_heads=old.n_heads,
                n_points=old.n_points,max_horizon=old.horizon_embedding.num_embeddings,
                max_offset=old.max_offset,horizon_chunk_size=old.horizon_chunk_size,
                dropout=drop,gradient_checkpointing=old.gradient_checkpointing)
            decoder.load_state_dict(old.state_dict());self.decoder=decoder
        if median_skip not in ('none','linear','seasonal'):raise ValueError('Unknown median_skip')
        self.median_skip_kind=median_skip
        self.skip_projection=None
        if median_skip!='none':
            self.skip_projection=nn.Linear(self.seq_len,self.pred_len,bias=False)
            nn.init.zeros_(self.skip_projection.weight)
            if median_skip=='seasonal':
                with torch.no_grad():
                    h=torch.arange(self.pred_len)
                    self.skip_projection.weight[h,self.seq_len-24+h%24]=1
            self.skip_gate=nn.Parameter(torch.tensor(math.atanh(.2)))
            self.skip_dropout=nn.Dropout(skip_dropout)
        self.encoder_kind=encoder_kind
        self.model_name='hadt_detail';self.output_mode='quantile'

    def forward(self,x,return_aux=False,intervention=None):
        result=super().forward(x,return_aux=return_aux,intervention=intervention)
        if self.skip_projection is None:return result
        q,aux=result if return_aux else (result,None)
        norm,mean,std=self.revin.normalize(x)
        if (intervention or getattr(self,'ablation',None))=='no_revin':
            norm=x;std=torch.ones_like(std)
        b,l,c=norm.shape
        residual=self.skip_projection(self.skip_dropout(norm.transpose(1,2).reshape(b*c,l)))
        residual=(.5*torch.tanh(self.skip_gate))*residual.reshape(b,c,self.pred_len).permute(0,2,1)
        residual=residual*std
        # Common shift preserves all quantile gaps and ordering.
        q=q+residual.unsqueeze(-1)
        if return_aux:
            aux['median_skip']=residual
            return q,aux
        return q
