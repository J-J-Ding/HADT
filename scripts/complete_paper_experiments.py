"""Frozen four-GPU completion study; validation-only baseline LR selection, then full tests."""
import argparse,copy,fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import atomic_json,sha256,source_manifest

MODELS=('dlinear','patchtst','timemixer','itransformer')
VARIANTS=('fixed','shared','dense','no_skip','fixed_no_skip','independent')
HORIZONS=(96,192,336,720); SEEDS=(2024,2025,2026)
ROOT=Path('outputs/paper_completion_20261126')

def build(config,horizon,n_vars,variant):
    import torch
    from torch import nn
    from utils.model_registry import build_model
    model=build_model(config,horizon,n_vars)
    if variant in ('fixed','fixed_no_skip'):model.ablation='no_offsets'
    if variant=='shared':
        class SharedEmbedding(nn.Module):
            def __init__(self,original):
                super().__init__();self.weight=nn.Parameter(original.weight[:1].detach().clone())
                self.num_embeddings=original.num_embeddings
            def forward(self,index):return self.weight.expand(len(index),-1)
        model.decoder.horizon_embedding=SharedEmbedding(model.decoder.horizon_embedding)
    if variant=='dense':
        from layers.hadt_nq.dense_horizon import DenseHorizonAttention
        old=model.decoder
        decoder=DenseHorizonAttention(d_model=old.d_model,n_heads=old.n_heads,n_points=model.n_patches,
           max_horizon=old.horizon_embedding.num_embeddings,max_offset=old.max_offset,
           horizon_chunk_size=old.horizon_chunk_size,dropout=config['model']['params']['dropout'],
           gradient_checkpointing=old.gradient_checkpointing)
        decoder.load_state_dict({k:v for k,v in old.state_dict().items() if not k.startswith('offset_projection.')})
        model.decoder=decoder;model.n_points=model.n_patches
    if variant in ('no_skip','fixed_no_skip'):
        model.skip_projection=None
        if hasattr(model,'skip_gate'):del model.skip_gate
        if hasattr(model,'skip_dropout'):del model.skip_dropout
    if variant=='independent':
        layer=nn.Linear(model.d_model,7)
        nn.init.normal_(layer.weight,std=.001)
        with torch.no_grad():layer.bias.copy_(torch.tensor([-1.6503,-1.1002,-.5501,0,.5501,1.1002,1.6503]))
        model.quantile_head=layer
    return model

def config_for(job):
    import yaml
    if job['kind']=='baseline':
        cfg=yaml.safe_load(Path(f"configs/baselines/{job['name']}.yaml").read_text())
        if 'dropout' in cfg['model']['params']:cfg['model']['params']['dropout']=.2
        cfg['training'].update(batch_size=32,weight_decay=.001,learning_rate=job['lr'],epochs=job['epochs'])
        cfg['evaluation']['batch_size']=32
    else:
        cfg=yaml.safe_load(Path('configs/optimized_etth1.yaml').read_text())
        cfg['training']['epochs']=job['epochs']
    cfg['runtime'].update(device='cuda:0',torch_threads=3)
    cfg['visualization']['enabled']=False
    return cfg

def path_for(root,job):
    name=job['name'] if job['kind']=='baseline' else 'hadt_'+job['name']
    if job['phase']=='screen':name+=f"_lr{job['lr']}"
    return root/job['phase']/name/f"H{job['horizon']}"/f"seed{job['seed']}"

def verify(root):
    p=json.loads((root/'protocol.json').read_text())
    assert p['source']==source_manifest(),'Model source changed during frozen completion study'
    assert p['driver']==sha256(__file__),'Completion driver changed'
    for path,digest in p['config_sha256'].items():assert sha256(path)==digest
    assert sha256('dataset/ETT-small/ETTh1.csv')==p['data_sha256']
    return p

def run_job(root,job):
    import numpy as np,pandas as pd,torch
    from data_provider.data_factory import build_data_bundle
    from utils.seed import set_seed
    from utils.logger import create_logger
    from engine.trainer import Trainer
    from engine.checkpoint import load_checkpoint
    from engine.evaluator import evaluate_test
    from utils.forecast_detail_metrics import evaluate_details,training_jump_thresholds
    verify(root);run=path_for(root,job);run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        r=json.loads((run/'complete.json').read_text());assert r['job']==job
        for f,d in r['artifacts'].items():assert sha256(run/f)==d
        return r
    set_seed(job['seed']);torch.set_num_threads(3)
    cfg=config_for(job);bundle=build_data_bundle(cfg,'ETTh1',job['horizon'])
    cfg['data_scaler']=bundle.scaler.state_dict();cfg['feature_columns']=bundle.feature_columns
    model=build(cfg,job['horizon'],bundle.n_vars,job['name'] if job['kind']=='ablation' else 'baseline').to('cuda:0')
    atomic_json(run/'resolved_config.json',dict(config=cfg,job=job,split_bounds=bundle.split_bounds,
        feature_columns=bundle.feature_columns,scaler=bundle.scaler.state_dict(),
        native_config=getattr(model,'native_config',None),
        n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    trainer=Trainer(model,optimizer,scheduler,torch.device('cuda:0'),cfg,run,create_logger(run/'train.log'))
    cp,best,elapsed=trainer.fit(bundle.train_loader,bundle.val_loader)
    checkpoint=load_checkpoint(cp,model,map_location='cuda:0')
    result=dict(job=job,best_val=float(best),best_epoch=checkpoint['epoch'],epochs_run=len(pd.read_csv(run/'training_history.csv')),
        training_seconds=elapsed,source_dir=str(run.resolve()),test_used_for_selection=False)
    if job['phase']!='screen':
        metrics=evaluate_test(model,bundle.test_loader,torch.device('cuda:0'),bundle.scaler,8,model.quantiles,run)
        details=evaluate_details(model,bundle.test_loader,torch.device('cuda:0'),training_jump_thresholds(bundle),bundle.scaler)
        if job['name']=='independent':
            metrics={k:v for k,v in metrics.items() if k in ('QuantileLoss_standardized','CrossingRate')}
            details={k:v for k,v in details.items() if not k.startswith(('PICP','PINAW'))}
            atomic_json(run/'metrics.json',metrics)
            # Invalid endpoints are never published as valid prediction intervals.
            (run/'horizon_metrics.csv').unlink(missing_ok=True)
        result.update(metrics=metrics,details=details)
    files=['best_checkpoint.pt','resolved_config.json','training_history.csv']
    if job['phase']!='screen':files.append('metrics.json')
    result['artifacts']={f:sha256(run/f) for f in files}
    verify(root);atomic_json(run/'complete.json',result);return result

def worker(root,phase):
    while True:
        with (root/f'{phase}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=root/f'{phase}_queue.json';q=json.loads(p.read_text());i=q['next']
            if i>=len(q['jobs']):return
            job=q['jobs'][i];q['next']+=1;atomic_json(p,q)
        state=root/'workers'/f"gpu{os.environ['CUDA_VISIBLE_DEVICES']}.json"
        atomic_json(state,dict(status='running',job=job,pid=os.getpid()))
        try:run_job(root,job)
        except BaseException as e:
            atomic_json(state,dict(status='failed',job=job,error=str(e)));raise
        atomic_json(state,dict(status='waiting',last_job=job))

def stage(root,phase,jobs):
    atomic_json(root/f'{phase}_queue.json',dict(jobs=jobs,next=0));children=[]
    for gpu in (0,1,2,3):
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/'workers'/f'{phase}_gpu{gpu}.log').open('w')
        proc=subprocess.Popen([sys.executable,__file__,'--worker','--phase',phase,'--root',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((proc,log))
    try:
        while any(p.poll() is None for p,l in children):
            if any(p.poll() not in (None,0) for p,l in children):raise RuntimeError('Completion worker failed; inspect workers logs')
            done=sum((path_for(root,j)/'complete.json').exists() for j in jobs)
            atomic_json(root/'progress.json',dict(status='running',phase=phase,completed=done,total=len(jobs)))
            time.sleep(5)
        if any(p.returncode for p,l in children):raise RuntimeError('Completion stage failed')
    finally:
        for p,l in children:
            if p.poll() is None:p.terminate()
            p.wait();l.close()
    rows=[json.loads((path_for(root,j)/'complete.json').read_text()) for j in jobs]
    atomic_json(root/f'{phase}_records.json',rows)
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--phase')
    args=parser.parse_args();root=args.root
    if args.worker:worker(root,args.phase);return
    root.mkdir(parents=True,exist_ok=True);(root/'workers').mkdir(exist_ok=True)
    protocol=dict(source=source_manifest(),driver=sha256(__file__),
        config_sha256={str(p):sha256(p) for p in [Path('configs/optimized_etth1.yaml')]+[Path(f'configs/baselines/{m}.yaml') for m in MODELS]},
        data_sha256=sha256('dataset/ETT-small/ETTh1.csv'),dataset='ETTh1',history=336,
        baseline_lr_grid=[1e-4,3e-4,1e-3],selection='Minimum H96 seed2024 8-epoch validation loss; freeze before any confirmation test.',
        common_training='batch32; AdamW wd0.001; neural dropout0.2; cap30; patience7; pinball; best validation checkpoint',
        ablations=list(VARIANTS),ablation_horizons=[96,720],baseline_horizons=list(HORIZONS),seeds=list(SEEDS),gpus=[0,1,2,3])
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol)
    # First fill the causal evidence gap in the final architecture.
    ablations=[dict(phase='ablation',kind='ablation',name=m,horizon=h,seed=s,epochs=30) for h in (96,720) for s in SEEDS for m in VARIANTS]
    stage(root,'ablation',ablations)
    screen=[dict(phase='screen',kind='baseline',name=m,horizon=96,seed=2024,epochs=8,lr=lr) for m in MODELS for lr in (1e-4,3e-4,1e-3)]
    rows=stage(root,'screen',screen)
    selected={m:min([r for r in rows if r['job']['name']==m],key=lambda r:(r['best_val'],r['job']['lr']))['job']['lr'] for m in MODELS}
    selection=dict(learning_rates=selected,test_used=False,screen_records_sha256=sha256(root/'screen_records.json'))
    if (root/'selection.json').exists():assert json.loads((root/'selection.json').read_text())==selection
    else:atomic_json(root/'selection.json',selection)
    confirms=[dict(phase='baseline',kind='baseline',name=m,horizon=h,seed=s,epochs=30,lr=selected[m]) for h in HORIZONS for s in SEEDS for m in MODELS]
    stage(root,'baseline',confirms)
    atomic_json(root/'progress.json',dict(status='complete',ablation_runs=36,baseline_screen_runs=12,baseline_test_runs=48))
    print('Completed 96 training jobs, including 84 full-test evaluations.',flush=True)

if __name__=='__main__':main()
