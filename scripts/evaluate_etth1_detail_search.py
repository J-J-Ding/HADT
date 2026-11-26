"""Evaluate only the validation-frozen configurations on the full test splits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import atomic_json,sha256
from scripts.run_etth1_detail_search import verify_protocol
from experiments.detail_search import HORIZONS,SEEDS


def evaluate_job(root,job):
    import numpy as np
    import torch
    from data_provider.data_factory import build_data_bundle
    from utils.model_registry import build_model
    from utils.forecast_detail_metrics import evaluate_details
    from utils.seed import set_seed
    from engine.checkpoint import load_checkpoint
    from engine.evaluator import evaluate_test
    from engine.visualization import create_run_figures
    from scripts.plot_run_forecasts import render_run
    verify_protocol(root)
    selection=json.loads((root/'selection.json').read_text())
    allowed={'reference',selection['accuracy'],selection['detail']}
    horizon_path=root/'horizon_selection.json'
    if horizon_path.exists():
        choices=json.loads(horizon_path.read_text())['choices']
        allowed.update(v[k] for v in choices.values() for k in ('accuracy','detail'))
    if job['candidate'] not in allowed:
        raise ValueError('Test candidate was not frozen before testing')
    run=root/'confirm'/f'L{selection["history"]}'/job['candidate']/f'H{job["horizon"]}'/f'seed{job["seed"]}'
    validation=json.loads((run/'validation_complete.json').read_text())
    signature=dict(selection_sha256=sha256(root/'selection.json'),validation_sha256=sha256(run/'validation_complete.json'),horizon_selection_sha256=sha256(horizon_path) if horizon_path.exists() else None)
    marker=run/'test_complete.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['signature']!=signature:raise ValueError('Test cache conflict')
        for f,h in record['artifact_sha256'].items():
            if sha256(run/f)!=h:raise ValueError('Test artifact conflict')
        return record
    for f,h in validation['artifact_sha256'].items():
        if sha256(run/f)!=h:raise ValueError('Validation checkpoint/config/history conflict')
    set_seed(job['seed']);torch.set_num_threads(3);device=torch.device('cuda:0')
    config=json.loads((run/'resolved_config.json').read_text())['config']
    bundle=build_data_bundle(config,'ETTh1',job['horizon'])
    model=build_model(config,job['horizon'],bundle.n_vars).to(device)
    cp=load_checkpoint(run/'best_checkpoint.pt',model,map_location=device)
    test=evaluate_test(model,bundle.test_loader,device,bundle.scaler,8,model.quantiles,run)
    detail=evaluate_details(model,bundle.test_loader,device,np.array(validation['jump_thresholds']),bundle.scaler)
    assert abs(detail['pinball']-test['QuantileLoss_standardized'])<1e-6
    create_run_figures(model,bundle,device,run,test,max_channels=7)
    record=dict(candidate=job['candidate'],candidate_id=job['candidate'],dataset='ETTh1',history=selection['history'],horizon=job['horizon'],seed=job['seed'],
                source_dir=str(run.resolve()),test=test,test_detail=detail,validation=validation['validation'],
                train_diagnostic=validation['train_diagnostic'],train_val_gap=validation['train_val_gap'],
                best_epoch=cp['epoch'],epochs_run=validation['epochs_run'],signature=signature,
                test_used_for_selection=False)
    render_run(record)
    artifacts=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and (p.suffix in ('.png','.pdf','.html','.npz') or p.name in ('metrics.json','horizon_metrics.csv','best_checkpoint.pt'))]
    record['artifact_sha256']={f:sha256(run/f) for f in artifacts}
    verify_protocol(root);atomic_json(marker,record)
    return record


def worker(root):
    while True:
        with (root/'test_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=root/'test_queue.json';queue=json.loads(path.read_text());index=queue['next']
            if index>=len(queue['jobs']):return
            job=queue['jobs'][index];queue['next']+=1;atomic_json(path,queue)
        evaluate_job(root,job)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/etth1_detail_search'))
    p.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3]);p.add_argument('--worker',action='store_true')
    p.add_argument('--selected-only',action='store_true',help='For a standalone retrain root without a reference run')
    p.add_argument('--horizons',type=int,nargs='+',choices=HORIZONS,default=list(HORIZONS))
    args=p.parse_args();root=args.root
    if args.worker:worker(root);return
    verify_protocol(root);selection=json.loads((root/'selection.json').read_text())
    chosen=list(dict.fromkeys(([] if args.selected_only else ['reference'])+[selection['accuracy'],selection['detail']]))
    if (root/'horizon_selection.json').exists():
        horizon_choices=json.loads((root/'horizon_selection.json').read_text())['choices']
        chosen=list(dict.fromkeys(chosen+[v[k] for v in horizon_choices.values() for k in ('accuracy','detail')]))
    jobs=[dict(candidate=c,horizon=h,seed=s) for c in chosen for h in args.horizons for s in SEEDS]
    atomic_json(root/'test_queue.json',dict(jobs=jobs,next=0))
    children=[]
    for gpu in args.gpus:
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/'workers'/f'test_gpu{gpu}.log').open('w')
        process=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((process,log))
    try:
        while any(proc.poll() is None for proc,log in children):
            if any(proc.poll() not in (None,0) for proc,log in children):raise RuntimeError('Test worker failed')
            atomic_json(root/'test_progress.json',dict(status='running',completed=len(list((root/'confirm').rglob('test_complete.json'))),total=len(jobs)))
            time.sleep(5)
        if any(proc.returncode for proc,log in children):raise RuntimeError('Test worker failed')
    except BaseException:
        for proc,log in children:
            if proc.poll() is None:proc.terminate()
        raise
    finally:
        for proc,log in children:proc.wait();log.close()
    records=[json.loads(p.read_text()) for p in sorted((root/'confirm').rglob('test_complete.json'))]
    if len(records)!=len(jobs):raise ValueError('Incomplete final tests')
    atomic_json(root/'test_records.json',records)
    from scripts.plot_run_forecasts import write_gallery
    write_gallery(records,root/'index.html')
    atomic_json(root/'test_progress.json',dict(status='complete',completed=len(records),total=len(jobs)))
    print('Frozen selection test evaluation complete',flush=True)


if __name__=='__main__':main()
