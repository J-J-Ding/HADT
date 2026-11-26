import math
import torch
from torch import nn
from torch.nn import functional as F
from models.linear_nq import DirectQuantileControl
from utils.quantile_heads import initialize_raw_projection
from layers.hadt_nq.dense_horizon import DenseHorizonAttention

def make_paper_model(config,pred_len,method):
    from experiments.runner import _build_model
    model=_build_model(config,pred_len)
    if method=='hadt':
        return model
    if method in ('linear','patch'):
        from models.linear_nq import LinearNQ
        from models.patch_nq import PatchNQ
        return LinearNQ(model) if method=='linear' else PatchNQ(model)
    if method=='fixed':
        from models.fixed_nq import FixedNQ
        return FixedNQ(model)
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
        from models.dense_nq import DenseNQ
        return DenseNQ(model,config)
    else:
        raise ValueError(f'Unknown paper method: {method}')
    return model
