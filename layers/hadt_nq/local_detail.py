"""Local temporal mixing and alternative grids for the ETTh1 validation study."""
import torch
from torch import nn
from layers.hadt_nq.horizon_deformable import HorizonDeformableAttention


class LocalResidualBlock(nn.Module):
    def __init__(self, width, hidden, dropout, kernel=3, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, kernel, padding=(kernel//2)*dilation,
                                  dilation=dilation, groups=width)
        self.features = nn.Sequential(nn.Linear(width,hidden),nn.GELU(),
                                      nn.Dropout(dropout),nn.Linear(hidden,width))
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((width,),0.1))

    def forward(self,x):
        z=self.norm(x)
        z=z+self.temporal(z.transpose(1,2)).transpose(1,2)
        return x+self.scale*self.dropout(self.features(z))


class RecentGridAttention(HorizonDeformableAttention):
    """Denser references near the latest history, followed by learned offsets."""
    def __init__(self, grid_power=2.0, **kwargs):
        super().__init__(**kwargs)
        self.grid_power=float(grid_power)

    def _forward_chunk(self,query_base,key,value,reference,disable_offsets=False):
        maximum=max(key.shape[1]-1,1)
        u=reference/maximum
        grid=maximum*(1-(1-u).pow(self.grid_power))
        return super()._forward_chunk(query_base,key,value,grid,disable_offsets)
