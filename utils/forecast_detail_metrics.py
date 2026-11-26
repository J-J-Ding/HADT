"""Validation-only diagnostics for detail preservation; jump thresholds use train rows."""
import numpy as np
import torch


def training_jump_thresholds(bundle):
    values=bundle.train_loader.dataset.values[:bundle.split_bounds['train'][1]]
    return np.maximum(np.quantile(np.abs(np.diff(values,axis=0)),.95,axis=0),1e-6)


@torch.no_grad()
def evaluate_details(model,loader,device,thresholds,scaler):
    model.eval();count=0;pinball=mae=delta_error=tv_pred=tv_true=jump_error=0.
    delta_count=jump_count=0;cover=width=0.
    tau=torch.tensor(model.quantiles,device=device,dtype=torch.float64).view(1,1,1,-1)
    threshold=torch.tensor(thresholds,device=device).view(1,1,-1)
    factor=torch.tensor(scaler.scale_/np.where(scaler.data_range_==0,1,scaler.data_range_),device=device).view(1,1,-1)
    for x,y in loader:
        x=x.to(device);y=y.to(device).double();q=model(x).double()
        if not torch.isfinite(q).all():raise ValueError('Non-finite detail predictions')
        e=y.unsqueeze(-1)-q
        pinball+=float(torch.maximum(tau*e,(tau-1)*e).sum())
        median=q[...,3];mae+=float((median-y).abs().sum());count+=y.numel()
        dy=y[:,1:]-y[:,:-1];dp=median[:,1:]-median[:,:-1]
        delta_error+=float((dp-dy).abs().sum());delta_count+=dy.numel()
        tv_pred+=float(dp.abs().sum());tv_true+=float(dy.abs().sum())
        mask=dy.abs()>=threshold
        jump_error+=float(((median-y)[:,1:].abs()*mask).sum());jump_count+=int(mask.sum())
        cover+=float(((y>=q[...,0])&(y<=q[...,6])).sum())
        width+=float(((q[...,6]-q[...,0])*factor).sum())
    if not count:raise ValueError('Empty evaluation population')
    return dict(pinball=pinball/(count*7),median_mae=mae/count,delta_mae=delta_error/max(delta_count,1),
                jump_mae=jump_error/max(jump_count,1),jump_count=jump_count,roughness_ratio=tv_pred/max(tv_true,1e-12),
                PICP90=cover/count,PINAW90=width/count,elements=count)
