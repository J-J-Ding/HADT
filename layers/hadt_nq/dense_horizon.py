import math
import torch
from torch import nn
from torch.nn import functional as F
from layers.hadt_nq.horizon_deformable import HorizonDeformableAttention

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
