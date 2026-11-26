"""Historical direct quantile control; not DLinear."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from utils.quantile_heads import ordered_from_raw, initialize_raw_projection

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
        if intervention is not None:
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
        result = self.revin.denormalize(q,mean,std)
        return (result,{}) if return_aux else result



class LinearNQ(DirectQuantileControl):
    def __init__(self, backbone):
        super().__init__(backbone, patch=False)
