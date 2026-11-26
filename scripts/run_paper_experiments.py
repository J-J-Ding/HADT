"""Deadline-bounded reproducible benchmark; only completed runs enter paper tables."""
import argparse,copy,hashlib,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data_provider.data_factory import build_data_bundle
from engine.trainer import Trainer
from engine.checkpoint import load_checkpoint
from engine.evaluator import evaluate_test
from experiments.paper_controls import make_paper_model
from utils.config import load_config
from utils.logger import create_logger
from utils.seed import set_seed

ROOT=Path('outputs/paper_study')
SEEDS=[2024,2025,2026]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest():
    paths=[*Path('models').rglob('*.py'),*Path('layers').rglob('*.py'),*Path('utils').rglob('*.py'),*Path('experiments').rglob('*.py'),Path('engine/trainer.py'),Path('engine/evaluator.py'),
           Path('utils/losses.py'),Path('utils/metrics.py'),
           *Path('data_provider').glob('*.py')]
    return hashlib.sha256(''.join(str(p)+digest(p) for p in sorted(paths)).encode()).hexdigest()


def configuration(dataset,method):
    c=load_config('configs/traffic_gpu3.yaml' if dataset=='Traffic' else 'configs/etth1_gpu3.yaml')
    c['model']['e_layers']=3
    c['experiment']['run_name']=f'paper_study/{method}'
    c['visualization']={'enabled':False}
    c['ablation']=None
    c['paper']={'method':method,'protocol':'paper-v1','quantile_controls':'local adaptations, shared ordered head'}
    return c


def immutable_config(c):
    return {k:v for k,v in c.items() if k not in ('experiment','visualization','paper','data_scaler','feature_columns')}


def import_pilot(dataset,method,horizon,seed,c,data_sha):
    if method!='hadt' or horizon!=96 or seed!=2024 or dataset not in ('ETTh1','Traffic'):return None
    run=Path('outputs/ETTh1_analysis/full/ETTh1/H96/seed2024' if dataset=='ETTh1' else 'outputs/Traffic_depth_comparison/layers3/Traffic/H96/seed2024')
    if not all((run/name).exists() for name in ['resolved_config.json','metrics.json','best_checkpoint.pt']):
        return None
    proof=Path('outputs/Traffic_depth_comparison/manifest.json' if dataset=='Traffic' else 'outputs/ETTh1_analysis/RESULTS.md')
    if not proof.exists():return None
    saved=json.loads((run/'resolved_config.json').read_text())
    if immutable_config(saved['config'])!=immutable_config(c):return None
    if dataset=='Traffic':
        old=json.loads(Path('outputs/Traffic_depth_comparison/manifest.json').read_text())['dataset_sha256']
    else:
        import re
        text=Path('outputs/ETTh1_analysis/RESULTS.md').read_text()
        old=re.search(r'SHA256：`([0-9a-f]{64})`',text).group(1)
    if old!=data_sha:raise RuntimeError('Pilot dataset changed')
    m=json.loads((run/'metrics.json').read_text())
    return dict(m,method=method,horizon=horizon,source_dir=str(run),source_kind='verified_prior_run',
                data_sha256=data_sha,code_sha256=None,
                provenance_note='Historical source hash was not captured; configuration/data/checkpoint origin verified.')


@torch.no_grad()
def samples(model,bundle,device,horizon,path,method):
    ds=bundle.test_loader.dataset
    ids=sorted(set([0,len(ds)//2,len(ds)-1]))
    channels=np.unique(np.linspace(0,bundle.n_vars-1,min(8,bundle.n_vars),dtype=int))
    x=torch.stack([ds[i][0][:,channels] for i in ids]).to(device)
    y=np.stack([ds[i][1][:,channels].numpy() for i in ids])
    model.eval();pred=[];auxiliary=[]
    for a in x:
        if method=='hadt':
            q,aux=model(a[None],return_aux=True)
            auxiliary.append({k:v.cpu().numpy() for k,v in aux.items() if torch.is_tensor(v)})
        else:q=model(a[None])
        pred.append(q.cpu().numpy())
    q=np.concatenate(pred)
    scale=bundle.scaler.scale_[channels];mean=bundle.scaler.mean_[channels]
    np.savez_compressed(path/'forecast_samples.npz',history=x.cpu().numpy()*scale+mean,target=y*scale+mean,
                        quantiles=q*scale[None,None,:,None]+mean[None,None,:,None],
                        sample_indices=ids,channel_indices=channels,feature_columns=np.array(bundle.feature_columns)[channels])
    if auxiliary:
        np.savez_compressed(path/'intermediate_steps.npz',**{k:np.concatenate([a[k] for a in auxiliary]) for k in auxiliary[0]},
                            sample_indices=ids,channel_indices=channels)


def run(dataset,method,horizon,seed,device,source_hash):
    c=configuration(dataset,method)
    data_path=Path(c['data']['root_path'])/c['data']['files'][dataset]
    data_sha=digest(data_path)
    imported=import_pilot(dataset,method,horizon,seed,c,data_sha)
    if imported:return imported
    path=ROOT/method/dataset/f'H{horizon}'/f'seed{seed}'
    signature=dict(config=copy.deepcopy(c),data_sha256=data_sha,code_sha256=source_hash,seed=seed,horizon=horizon)
    expected=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    marker=path/'provenance.json'
    if marker.exists() and json.loads(marker.read_text())['fingerprint']!=expected:
        raise RuntimeError(f'Cache fingerprint mismatch {path}')
    if (path/'complete.json').exists():return json.loads((path/'complete.json').read_text())
    set_seed(seed)
    bundle=build_data_bundle(c,dataset,horizon)
    c['data_scaler']=bundle.scaler.state_dict();c['feature_columns']=bundle.feature_columns
    path.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(dict(fingerprint=expected,**signature),indent=2))
    log=create_logger(path/'train.log')
    model=make_paper_model(c,horizon,method).to(device)
    nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    resolved=dict(dataset=dataset,pred_len=horizon,seed=seed,n_vars=bundle.n_vars,n_rows=bundle.n_rows,
                  feature_columns=bundle.feature_columns,split_bounds=bundle.split_bounds,
                  scaler=bundle.scaler.state_dict(),config=c,method=method,n_params=nparams)
    (path/'resolved_config.json').write_text(json.dumps(resolved,indent=2))
    log.info(f'Start method={method} dataset={dataset} H={horizon} seed={seed} GPU={torch.cuda.get_device_name(0)} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    t=c['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(t['learning_rate']),weight_decay=float(t['weight_decay']))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=t['scheduler_factor'],
               patience=t['scheduler_patience'],min_lr=t['min_learning_rate'])
    trainer=Trainer(model,optimizer,scheduler,device,c,path,log)
    checkpoint,best,seconds=trainer.fit(bundle.train_loader,bundle.val_loader)
    cp=load_checkpoint(checkpoint,model,map_location=device)
    metrics=evaluate_test(model,bundle.test_loader,device,bundle.scaler,c['evaluation']['channel_chunk_size'],
                          c['model']['quantiles'],path)
    samples(model,bundle,device,horizon,path,method)
    if digest(data_path)!=data_sha or code_digest()!=source_hash:raise RuntimeError('Inputs changed during training')
    record=dict(metrics,dataset=dataset,method=method,horizon=horizon,seed=seed,best_val_pinball=best,
                training_seconds=seconds,n_params=nparams,best_epoch=cp['epoch'],source_dir=str(path),
                source_kind='paper_run',data_sha256=data_sha,code_sha256=source_hash)
    (path/'complete.json').write_text(json.dumps(record,indent=2))
    del model,bundle,trainer,optimizer,scheduler,cp
    torch.cuda.empty_cache()
    return record


def queue():
    jobs=[]
    # Complete three-seed groups first. Primary tables never substitute unrun cells.
    for dataset in ['ETTh1','ETTh2']:
        for method in ['hadt','linear','patch','dense','fixed']:
            for seed in SEEDS:jobs.append((dataset,method,96,seed,'primary'))
    for method in ['shared','independent','no_revin']:
        for seed in SEEDS:jobs.append(('ETTh1',method,96,seed,'ablation'))
    for dataset in ['ETTh1','ETTh2']:
        for method in ['hadt','linear','dense']:
            for seed in SEEDS:jobs.append((dataset,method,192,seed,'horizon_extension'))
    jobs.extend([('Traffic','hadt',96,2024,'traffic_case'),('Traffic','linear',96,2024,'traffic_case')])
    jobs.append(('Traffic','dense',96,2024,'optional_traffic_case'))
    return jobs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stop-before-utc',default='2099-12-31T23:59:00+00:00')
    args=parser.parse_args()
    cutoff=datetime.fromisoformat(args.stop_before_utc).timestamp()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('PAPER_EXPECTED_GPU','3') or not torch.cuda.is_available():raise RuntimeError('GPU3 required')
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True,exist_ok=True)
    source_hash=code_digest()
    (ROOT/'protocol.json').write_text(json.dumps(dict(source_hash=source_hash,queue=queue(),cutoff_utc=args.stop_before_utc,
                                quantiles=[.05,.1,.25,.5,.75,.9,.95],primary_seeds=SEEDS,GPU=os.environ.get('PAPER_EXPECTED_GPU','3')),indent=2))
    records=[];timings=[]
    for i,(dataset,method,horizon,seed,group) in enumerate(queue()):
        remaining=cutoff-time.time()
        estimate=120 if dataset!='Traffic' else (200 if method=='linear' else 1600)
        if remaining<estimate+60:
            print(f'DEADLINE stop before {dataset}/{method}/H{horizon}/seed{seed}; remaining={remaining:.0f}s',flush=True)
            break
        if code_digest()!=source_hash:raise RuntimeError('Model/trainer code changed during experiment')
        start=time.time()
        print(f'JOB {i+1}/{len(queue())} {dataset} {method} H{horizon} seed{seed} group={group}',flush=True)
        record=run(dataset,method,horizon,seed,torch.device('cuda:0'),source_hash)
        record['study_group']=group
        records.append(record)
        (ROOT/'records.json').write_text(json.dumps(records,indent=2))
        import pandas as pd
        pd.DataFrame(records).to_csv(ROOT/'records.csv',index=False)
        print(f'DONE {dataset} {method} H{horizon} seed{seed} QL={record["QuantileLoss_standardized"]:.6f} elapsed={time.time()-start:.1f}s',flush=True)
    (ROOT/'queue_finished.json').write_text(json.dumps(dict(completed=len(records),utc=datetime.now(timezone.utc).isoformat()),indent=2))


if __name__=='__main__':main()
