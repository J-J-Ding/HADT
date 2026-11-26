"""Four-GPU single-seed ETTh1 convergence checks, point then quantile per model."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import atomic_json, sha256

MODELS=('dlinear','patchtst','itransformer','timemixer')


def worker(args):
    import torch
    from utils.config import load_config, validate_config
    from experiments.runner import run_single_experiment
    if Path(sys.prefix).name!='djjTF' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('djjTF and one isolated CUDA device required per worker')
    root=Path('outputs')/args.run_name;status=root/'workers'/f'{args.model}.json'
    results=[]
    try:
        for mode in ('point','quantile'):
            atomic_json(status,dict(status='running',model=args.model,gpu=args.gpu,mode=mode,completed_modes=len(results),pid=os.getpid()))
            config=load_config(f'configs/baselines/{args.model}.yaml')
            config['model']['output_mode']=mode
            config['experiment'].update(run_name=args.run_name,scope='single_seed_convergence_30_epoch_cap_not_tuned_benchmark')
            config['training'].update(epochs=30,early_stopping_patience=7)
            validate_config(config)
            result=run_single_experiment(config,'ETTh1',96,2024)
            results.append(result)
            atomic_json(root/'workers'/f'{args.model}_results.json',results)
        atomic_json(status,dict(status='complete',model=args.model,gpu=args.gpu,completed_modes=2,pid=os.getpid()))
    except BaseException as exc:
        atomic_json(status,dict(status='failed',model=args.model,gpu=args.gpu,error=str(exc),completed_modes=len(results),pid=os.getpid()))
        raise


def summarize(root):
    import pandas as pd
    import torch
    from scripts.plot_run_forecasts import write_gallery
    records=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/ETTh1/H96/seed2024/complete.json'))]
    rows=[]
    for r in records:
        run=Path(r['source_dir']);m=json.loads((run/'metrics.json').read_text())
        hist=pd.read_csv(run/'training_history.csv')
        checkpoint=torch.load(run/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        row=dict(model=r['model_name'],mode=r['output_mode'],source='new_training',source_dir=str(run),
                 epochs_run=len(hist),best_epoch=int(checkpoint['epoch']),
                 stop_reason='epoch_cap' if len(hist)==30 else 'early_stopping',
                 **{k:v for k,v in m.items() if isinstance(v,(int,float))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'training_summary.csv',index=False)
    qrows=[r for r in rows if r['mode']=='quantile']
    # Reuse only explicitly labeled historical HADT references. Their policies
    # (batch size, width and MLP learning rate) differ from the new controls.
    historical=[('HADT-Transformer (historical)',Path('outputs/architecture_study/confirm/transformer3/ETTh1/H96/seed2024')),
                ('HADT-MLP (historical)',Path('outputs/architecture_study/confirm/mlp64_lr3e4/ETTh1/H96/seed2024'))]
    for label,run in historical:
        if not (run/'metrics.json').exists():continue
        cp=torch.load(run/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        m=json.loads((run/'metrics.json').read_text());hist=pd.read_csv(run/'training_history.csv')
        qrows.append(dict(model=label,mode='quantile',source='historical_reference_different_config',source_dir=str(run),
                         epochs_run=len(hist),best_epoch=int(cp['epoch']),
                         **{k:v for k,v in m.items() if isinstance(v,(int,float))}))
    pd.DataFrame(qrows).to_csv(root/'quantile_comparison.csv',index=False)
    write_gallery(records,root/'index.html')
    atomic_json(root/'progress.json',dict(status='complete' if len(records)==8 else 'incomplete',completed=len(records),total=8,
                  scope='Single seed 2024; ETTh1/H96; untuned 30-epoch cap with validation early stopping; historical references labeled separately'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='baseline_etth1_30ep_seed2024')
    parser.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.worker:
        worker(args);return
    if len(set(args.gpus))!=4:raise ValueError('Four different GPU IDs required')
    root=Path('outputs')/args.run_name;root.mkdir(parents=True,exist_ok=True)
    (root/'workers').mkdir(exist_ok=True)
    processes=[]
    for gpu,model in zip(args.gpus,MODELS):
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/'workers'/f'{model}.log').open('w')
        child=subprocess.Popen([sys.executable,__file__,'--worker','--model',model,'--gpu',str(gpu),'--run-name',args.run_name],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((model,gpu,child,log))
    atomic_json(root/'launch.json',dict(configs=[dict(model=m,gpu=g,pid=p.pid) for m,g,p,l in processes],epochs=30,patience=7,dataset='ETTh1',horizon=96,seed=2024))
    try:
        while any(p.poll() is None for m,g,p,l in processes):
            statuses=[]
            for m,g,p,l in processes:
                path=root/'workers'/f'{m}.json'
                statuses.append(json.loads(path.read_text()) if path.exists() else dict(model=m,gpu=g,status='starting'))
            atomic_json(root/'progress.json',dict(status='running',workers=statuses,total_modes=8))
            time.sleep(5)
    except BaseException:
        for m,g,p,l in processes:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for m,g,p,l in processes:l.close()
    summarize(root)
    failed=[m for m,g,p,l in processes if p.returncode!=0]
    if failed:raise RuntimeError(f'Failed workers: {failed}; inspect {root}/workers')
    print('Completed four point + four quantile training runs:',root,flush=True)


if __name__=='__main__':main()
