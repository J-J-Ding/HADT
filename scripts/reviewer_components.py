"""Isolated reviewer-study components; frozen project modules remain untouched."""
import csv
import math
import types
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from layers.hadt_nq.horizon_deformable import HorizonDeformableAttention
from utils.losses import forecasting_loss
from utils.metrics import IntervalMetricAccumulator
from utils.provenance import atomic_json


def interior_chunk(self,query_base,key,value,reference,disable_offsets=False):
    reference=torch.linspace(.5,key.shape[1]-1.5,self.n_points,
        device=key.device,dtype=key.dtype).view(1,1,1,self.n_points)
    return HorizonDeformableAttention._forward_chunk(self,query_base,key,value,reference,disable_offsets)


def install_interior_grid(model):
    model.decoder._forward_chunk=types.MethodType(interior_chunk,model.decoder)
    return model


class CalibrationObjective:
    def __init__(self,weight,temperature=.1):
        self.weight=float(weight);self.temperature=float(temperature)
        self.epoch=0;self.rows=[];self.sums=np.zeros(4)

    def __call__(self,prediction,target,quantiles):
        pinball=forecasting_loss(prediction,target,quantiles)
        tau=prediction.new_tensor(quantiles).view(1,-1)
        coverage=torch.sigmoid((prediction-target.unsqueeze(-1))/self.temperature).mean((0,1))
        calibration=((coverage-tau)**2).mean()
        total=pinball+self.weight*calibration
        if not torch.isfinite(total):raise RuntimeError('Non-finite calibration objective')
        n=target.numel()
        self.sums+=np.asarray([pinball.detach().item()*n,calibration.detach().item()*n,total.detach().item()*n,n])
        return total


class ObjectiveLoggingLoader:
    def __init__(self,loader,objective,run):
        self.loader=loader;self.objective=objective;self.run=Path(run)
    def __len__(self):return len(self.loader)
    def __iter__(self):
        o=self.objective;o.epoch+=1;o.sums[:]=0
        for batch in self.loader:yield batch
        if o.sums[-1]:
            row=dict(epoch=o.epoch,train_pinball=o.sums[0]/o.sums[3],
                train_calibration=o.sums[1]/o.sums[3],train_objective=o.sums[2]/o.sums[3])
            p=self.run/'objective_components.csv'
            previous=list(csv.DictReader(p.open())) if p.exists() else []
            previous=[r for r in previous if int(r['epoch'])<o.epoch]
            with p.open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader();w.writerows(previous);w.writerow(row)


@torch.no_grad()
def evaluate(model,loader,device,bundle=None,run=None,description='Validation calibration'):
    model.eval();counts=None;denominator=0;loss_sum=elements=0
    accumulator=IntervalMetricAccumulator(bundle.scaler) if bundle is not None else None
    for x,y in tqdm(loader,desc=description,mininterval=2,dynamic_ncols=True):
        x=x.to(device);y=y.to(device);q=model(x)
        if not torch.isfinite(q).all():raise RuntimeError('Non-finite review prediction')
        hit=(y.unsqueeze(-1)<=q).sum((0,1)).double()
        counts=hit if counts is None else counts+hit
        denominator+=y.shape[0]*y.shape[1]
        loss_sum+=float(forecasting_loss(q,y,model.quantiles))*y.numel();elements+=y.numel()
        if accumulator is not None:
            accumulator.update(q.float().cpu().numpy(),y.float().cpu().numpy(),np.arange(y.shape[-1]),model.quantiles)
    if not elements:raise ValueError('Empty reviewer evaluation population')
    coverage=counts.cpu().numpy()/denominator
    ace=float(np.abs(coverage-np.asarray(model.quantiles)[None,:]).mean())
    metrics=accumulator.summary() if accumulator is not None else {'QuantileLoss_standardized':loss_sum/elements}
    metrics['ACE']=ace
    if run is not None:
        run=Path(run);atomic_json(run/'metrics.json',metrics)
        np.savez_compressed(run/'quantile_calibration.npz',coverage=coverage,quantiles=model.quantiles,count_per_variable=denominator)
        if accumulator is not None:accumulator.horizon_frame().to_csv(run/'horizon_metrics.csv',index=False)
    return metrics


@torch.no_grad()
def attention_diagnostics(model,bundle,device,run):
    ds=bundle.test_loader.dataset;indices=sorted({0,len(ds)//2,len(ds)-1})
    x=torch.stack([ds[i][0] for i in indices]).to(device)
    _,aux=model(x,return_aux=True)
    positions=aux['positions'];offsets=aux['offsets'];attention=aux['attention'];k=positions.shape[-1]
    entropy=-(attention*attention.clamp_min(1e-12).log()).sum(-1)/math.log(k)
    floor=positions.floor().sort(-1).values
    neighborhoods=1+(floor[...,1:]!=floor[...,:-1]).sum(-1)
    n=model.n_patches
    stats=dict(boundary_rate=float(((positions<=1e-6)|(positions>=n-1-1e-6)).float().mean()),
        mean_distinct_floor_neighbors=float(neighborhoods.float().mean()),
        mean_absolute_offset=float(offsets.abs().mean()),normalized_entropy=float(entropy.mean()),
        right_reference_absolute_offset=float(offsets[...,-1].abs().mean()),
        right_reference_attention=float(attention[...,-1].mean()),sample_indices=indices,
        population='All seven variables, all future steps, three fixed test origins; per-seed diagnostic only.')
    atomic_json(Path(run)/'attention_diagnostics.json',stats)
    np.savez_compressed(Path(run)/'attention_diagnostics.npz',positions=positions.cpu().numpy(),
        offsets=offsets.cpu().numpy(),attention=attention.cpu().numpy(),entropy_by_head=entropy.mean((0,1,2)).cpu().numpy())
    return stats


def smoke_checks():
    torch.set_num_threads(3);torch.manual_seed(2024)
    raw=torch.randn(2,8,3,7,requires_grad=True);y=torch.randn(2,8,3)
    from utils.losses import DEFAULT_QUANTILES
    zero=CalibrationObjective(0)(raw,y,DEFAULT_QUANTILES)
    assert torch.equal(zero,forecasting_loss(raw,y,DEFAULT_QUANTILES))
    value=CalibrationObjective(1)(raw,y,DEFAULT_QUANTILES);value.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
    n=41;features=(torch.arange(n,dtype=torch.float32)**2).view(1,n,1,1)
    derivatives=[]
    for last in [40.,39.5]:
        pos=torch.tensor([[[[last]]]],requires_grad=True)
        HorizonDeformableAttention._linear_sample(features,pos).sum().backward();derivatives.append(float(pos.grad))
    assert derivatives[0]==0 and derivatives[1]>0
    import yaml
    from utils.model_registry import build_model
    cfg=yaml.safe_load(Path('configs/optimized_etth1.yaml').read_text())
    for variant in ['full','points_4','points_12','interior_grid']:
        import copy
        c=copy.deepcopy(cfg)
        if variant.startswith('points_'):c['model']['params']['n_points']=int(variant.split('_')[-1])
        model=build_model(c,96,3)
        if variant=='interior_grid':install_interior_grid(model)
        x=torch.randn(2,336,3);q=model(x)
        assert q.shape==(2,96,3,7) and (q[...,1:]-q[...,:-1]>=0).all()
        CalibrationObjective(.1)(q,torch.randn(2,96,3),model.quantiles).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Exercise the existing Trainer/objective/checkpoint/evaluation interfaces on CPU.
    import tempfile
    from types import SimpleNamespace
    from torch.utils.data import DataLoader,TensorDataset
    from engine.trainer import Trainer
    from engine.checkpoint import load_checkpoint
    from utils.logger import create_logger
    from data_provider.scaler import StandardScaler
    with tempfile.TemporaryDirectory(prefix='hadt_reviewer_smoke_') as directory:
        import copy
        c=copy.deepcopy(cfg);c['training']['epochs']=1
        model=build_model(c,96,3).cpu()
        data=TensorDataset(torch.randn(4,336,3),torch.randn(4,96,3))
        loader=DataLoader(data,batch_size=2,shuffle=False)
        objective=CalibrationObjective(.1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        trainer=Trainer(model,optimizer,scheduler,torch.device('cpu'),c,directory,
            create_logger(Path(directory)/'smoke.log'),train_objective=objective)
        cp,best,elapsed=trainer.fit(ObjectiveLoggingLoader(loader,objective,directory),loader)
        load_checkpoint(cp,model,map_location='cpu')
        components=list(csv.DictReader((Path(directory)/'objective_components.csv').open()))
        assert len(components)==1
        r={k:float(v) for k,v in components[0].items()}
        assert abs(r['train_objective']-r['train_pinball']-.1*r['train_calibration'])<1e-6
        bundle=SimpleNamespace(scaler=StandardScaler().fit(np.random.randn(50,3)),test_loader=loader)
        result=evaluate(model,loader,torch.device('cpu'),bundle,directory,'CPU integration check')
        assert np.isfinite(result['ACE']) and result['CrossingRate']==0
        attention_diagnostics(model,bundle,torch.device('cpu'),directory)
    return dict(status='passed',lambda_zero_exact=True,calibration_gradient_finite=True,
        boundary_derivatives=derivatives,model_variants_checked=4,output_shape_and_order=True,
        trainer_objective_checkpoint_and_evaluation_integration=True)
