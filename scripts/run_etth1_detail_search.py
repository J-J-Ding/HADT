"""Four-GPU validation-only optimization; tests are evaluated after frozen selection."""
import argparse
import copy
from functools import partial
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import atomic_json,sha256,source_manifest
from experiments.detail_search import HORIZONS,SEEDS,candidates,configuration,scores


def verify_protocol(root):
    p=json.loads((root/'protocol.json').read_text())
    if p['source']!=source_manifest() or p['driver_sha256']!=sha256(__file__):
        raise RuntimeError('Frozen search source changed; use a new root and do not re-sign old records')
    if p['data_sha256']!=sha256('dataset/ETT-small/ETTh1.csv'):
        raise RuntimeError('Frozen data changed')
    return p


def run_job(root,job):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader,Subset
    from data_provider.data_factory import build_data_bundle
    from utils.seed import set_seed
    from utils.model_registry import build_model
    from utils.logger import create_logger
    from utils.detail_objective import detail_objective
    from utils.forecast_detail_metrics import evaluate_details,training_jump_thresholds
    from engine.trainer import Trainer
    from engine.checkpoint import load_checkpoint
    verify_protocol(root)
    stage,candidate,history,horizon,seed=job['stage'],job['candidate'],job['history'],job['horizon'],job['seed']
    run=root/stage/f'L{history}'/candidate['id']/f'H{horizon}'/f'seed{seed}'
    run.mkdir(parents=True,exist_ok=True)
    marker=run/'validation_complete.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['job']!=job:raise ValueError('Cached job conflict')
        for f,h in record['artifact_sha256'].items():
            if sha256(run/f)!=h:raise ValueError('Cached validation artifact conflict')
        return record
    set_seed(seed);torch.set_num_threads(3);device=torch.device('cuda:0')
    config=configuration(candidate,history,job['epochs'],stage)
    bundle=build_data_bundle(config,'ETTh1',horizon)
    config['data_scaler']=bundle.scaler.state_dict();config['feature_columns']=bundle.feature_columns
    model=build_model(config,horizon,bundle.n_vars).to(device)
    atomic_json(run/'resolved_config.json',dict(config=config,job=job,n_vars=bundle.n_vars,
        feature_columns=bundle.feature_columns,split_bounds=bundle.split_bounds,scaler=bundle.scaler.state_dict(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),device=os.environ['CUDA_VISIBLE_DEVICES']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    objective=partial(detail_objective,weight=candidate['difference_weight'])
    trainer=Trainer(model,optimizer,scheduler,device,config,run,create_logger(run/'train.log'),train_objective=objective)
    cp,best,elapsed=trainer.fit(bundle.train_loader,bundle.val_loader)
    checkpoint=load_checkpoint(cp,model,map_location=device)
    thresholds=training_jump_thresholds(bundle)
    validation=evaluate_details(model,bundle.val_loader,device,thresholds,bundle.scaler)
    assert abs(validation['pinball']-best)<1e-6
    ds=bundle.train_loader.dataset
    starts=ds.target_starts
    indices=np.flatnonzero((starts>=512)&(starts<=bundle.split_bounds['train'][1]-720)&(starts%8==0))[:1024]
    diagnostic=DataLoader(Subset(ds,indices.tolist()),batch_size=32,shuffle=False)
    train=evaluate_details(model,diagnostic,device,thresholds,bundle.scaler)
    record=dict(job=job,stage=stage,candidate=candidate['id'],history=history,horizon=horizon,seed=seed,
        source_dir=str(run.resolve()),validation=validation,train_diagnostic=train,
        train_val_gap=validation['pinball']-train['pinball'],best_epoch=checkpoint['epoch'],
        epochs_run=len(pd.read_csv(run/'training_history.csv')),training_seconds=elapsed,
        jump_thresholds=thresholds.tolist(),train_diagnostic_origins=ds.target_starts[indices].tolist(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),test_used=False,
        artifact_sha256={f:sha256(run/f) for f in ['best_checkpoint.pt','resolved_config.json','training_history.csv']})
    verify_protocol(root);atomic_json(marker,record)
    return record


def claim(root,stage):
    lock=root/f'{stage}_queue.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        path=root/f'{stage}_queue.json';queue=json.loads(path.read_text())
        index=queue['next']
        if index>=len(queue['jobs']):return None
        job=queue['jobs'][index];queue['next']+=1;atomic_json(path,queue)
        return index,job


def worker(root,stage):
    import torch
    if Path(sys.prefix).name!='djjTF' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('djjTF and one isolated GPU required')
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];status=root/'workers'/f'{gpu}.json'
    while True:
        task=claim(root,stage)
        if task is None:break
        index,job=task
        atomic_json(status,dict(status='running',stage=stage,index=index,job=job,pid=os.getpid()))
        try:record=run_job(root,job)
        except BaseException as exc:
            atomic_json(status,dict(status='failed',stage=stage,index=index,job=job,error=str(exc)));raise
        atomic_json(status,dict(status='waiting',stage=stage,index=index,last_completed=record['source_dir']))
    atomic_json(status,dict(status='complete',stage=stage))


def execute_stage(root,stage,jobs,gpus):
    verify_protocol(root)
    path=root/f'{stage}_queue.json'
    if path.exists() and json.loads(path.read_text())['jobs']!=jobs:raise ValueError('Frozen queue conflict')
    atomic_json(path,dict(jobs=jobs,next=0));processes=[]
    for gpu in gpus:
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/'workers'/f'{stage}_gpu{gpu}.log').open('w')
        p=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--worker','--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((p,log,gpu))
    try:
        while any(p.poll() is None for p,l,g in processes):
            failed=[g for p,l,g in processes if p.poll() not in (None,0)]
            if failed:raise RuntimeError(f'Failed GPU workers {failed}; inspect logs')
            completed=len(list((root/stage).rglob('validation_complete.json'))) if (root/stage).exists() else 0
            atomic_json(root/'progress.json',dict(status='running',stage=stage,completed=completed,total=len(jobs)))
            time.sleep(5)
        if any(p.returncode for p,l,g in processes):raise RuntimeError('Stage worker failed')
    except BaseException:
        for p,l,g in processes:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for p,l,g in processes:p.wait();l.close()
    records=[json.loads(p.read_text()) for p in sorted((root/stage).rglob('validation_complete.json'))]
    if len(records)!=len(jobs):raise ValueError('Incomplete stage')
    atomic_json(root/f'{stage}_records.json',records)
    return records


def jobs_for(stage,choices,histories,seeds,epochs):
    return [dict(stage=stage,candidate=c,history=l,horizon=h,seed=s,epochs=epochs)
            for c in choices for l in histories for h in HORIZONS for s in seeds]


def freeze_json(path,value):
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError(f'Frozen selection differs: {path}')
    atomic_json(path,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/etth1_detail_search'))
    parser.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3]);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--stage',choices=['pilot','screen','confirm']);args=parser.parse_args();root=args.root
    if args.worker:worker(root,args.stage);return
    root.mkdir(parents=True,exist_ok=True);(root/'workers').mkdir(exist_ok=True)
    protocol=dict(dataset='ETTh1',horizons=list(HORIZONS),history_choices=[96,336,512],seeds=list(SEEDS),
        pilot_epochs=6,screen_epochs=6,confirmation_epochs=30,candidates=candidates(),
        source=source_manifest(),driver_sha256=sha256(__file__),data_sha256=sha256('dataset/ETT-small/ETTh1.csv'),
        selection='validation_only; equal horizon relative .7 pinball+.2 delta MAE+.1 jump MAE; accuracy and detail options frozen before test',gpus=args.gpus)
    freeze_json(root/'protocol.json',protocol)
    choices=candidates();ref=choices[0]
    pilot=execute_stage(root,'pilot',jobs_for('pilot',[ref],[96,336,512],[2024],6),args.gpus)
    groups={str(l):[r for r in pilot if r['history']==l] for l in (96,336,512)}
    ranking=scores(groups,groups['336']);history=int(ranking[0]['candidate'])
    freeze_json(root/'history_selection.json',dict(history=history,ranking=ranking,test_used=False))
    screen=execute_stage(root,'screen',jobs_for('screen',choices,[history],[2024],6),args.gpus)
    groups={c['id']:[r for r in screen if r['candidate']==c['id']] for c in choices}
    ranking=scores(groups,groups['reference'])
    shortlist=list(dict.fromkeys([r['candidate'] for r in ranking[:3]]+['reference']))
    freeze_json(root/'shortlist.json',dict(history=history,ranking=ranking,candidates=shortlist,test_used=False))
    confirmed=execute_stage(root,'confirm',jobs_for('confirm',[c for c in choices if c['id'] in shortlist],[history],SEEDS,30),args.gpus)
    groups={c:[r for r in confirmed if r['candidate']==c] for c in shortlist};ranking=scores(groups,groups['reference'])
    accuracy=min(ranking,key=lambda r:(r['mean_pinball'],r['candidate']))
    import numpy as np
    eligible=[]
    for row in ranking:
        coverage=all(np.mean([r['validation']['PICP90'] for r in groups[row['candidate']] if r['horizon']==h])>=
                     np.mean([r['validation']['PICP90'] for r in groups['reference'] if r['horizon']==h])-.02 for h in HORIZONS)
        if row['mean_pinball']<=accuracy['mean_pinball']*1.02 and coverage:eligible.append(row)
    detail=min(eligible,key=lambda r:(r['score'],r['candidate'])) if eligible else accuracy
    selection=dict(history=history,accuracy=accuracy['candidate'],detail=detail['candidate'],recommended=detail['candidate'],
        detail_constraints_satisfied=bool(eligible),ranking=ranking,test_used=False,scope='Best within the prespecified finite validation search, not a global optimum')
    freeze_json(root/'selection.json',selection)
    # The export/evaluation program is invoked only after this frozen selection exists.
    atomic_json(root/'progress.json',dict(status='selection_complete',selection=selection,completed_confirmations=len(confirmed)))
    print('Validation search complete:',root/'selection.json',flush=True)


if __name__=='__main__':main()
