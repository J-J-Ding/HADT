"""Complete final-model and external-baseline comparisons on six additional benchmark datasets."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import source_manifest,sha256,atomic_json
from scripts.complete_paper_experiments import build,config_for,MODELS,HORIZONS,SEEDS
ROOT=Path('outputs/paper_seven_datasets_20261126')
BASE=Path('outputs/paper_completion_20261126')
DATASETS=('ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity')
EPOCH_CAPS={d:10 if d in ('Traffic','Weather','Electricity') else 30 for d in DATASETS}
FILES={d:f'dataset/ETT-small/{d}.csv' for d in DATASETS if d.startswith('ETT')}
FILES.update(Traffic='dataset/traffic/traffic.csv',Weather='dataset/weather/weather.csv',Electricity='dataset/electricity/electricity.csv')

def path_for(root,job):
    return root/job['dataset']/job['name']/f"H{job['horizon']}"/f"seed{job['seed']}"

def verify(root):
    p=json.loads((root/'protocol_v2.json').read_text())
    assert source_manifest()==p['source']
    assert sha256(__file__)==p['driver']
    assert sha256('scripts/paper_resume_training.py')==p['resume_driver_sha256']
    assert sha256(root/'epoch_cap_amendment.json')==p['amendment_sha256']
    assert sha256('scripts/complete_paper_experiments.py')==p['helper_driver_sha256']
    assert sha256(BASE/'selection.json')==p['baseline_selection_sha256']
    for f,digest in p['data_sha256'].items():assert sha256(f)==digest
    for f,digest in p['config_sha256'].items():assert sha256(f)==digest

def run_job(root,job):
    import torch,numpy as np,pandas as pd
    from utils.seed import set_seed
    from utils.logger import create_logger
    from data_provider.data_factory import build_data_bundle
    from engine.trainer import Trainer
    from engine.checkpoint import load_checkpoint
    from engine.evaluator import evaluate_test
    from utils.forecast_detail_metrics import training_jump_thresholds,evaluate_details
    verify(root);run=path_for(root,job);run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        r=json.loads((run/'complete.json').read_text());assert r['job']==job
        for f,d in r['artifacts'].items():assert sha256(run/f)==d
        return r
    set_seed(job['seed']);torch.set_num_threads(3)
    cfg=config_for(job)
    if job['dataset'] in ('Traffic','Electricity'):
        if job['name']=='timemixer':batch=2 if job['dataset']=='Traffic' else 4
        elif job['name'] in ('patchtst','itransformer'):batch=8
        else:batch=32
        cfg['training']['batch_size']=batch
        cfg['evaluation']['batch_size']=min(batch,16)
    cfg['evaluation']['channel_chunk_size']=32
    if job['kind']=='full' and job['dataset'] in ('Traffic','Electricity','Weather'):
        cfg['training'].update(channel_policy='sample',channel_sample_size=16)
        cfg['evaluation']['channel_policy']='chunk'
    bundle=build_data_bundle(cfg,job['dataset'],job['horizon'])
    cfg['data_scaler']=bundle.scaler.state_dict();cfg['feature_columns']=bundle.feature_columns
    model=build(cfg,job['horizon'],bundle.n_vars,'full' if job['kind']=='full' else 'baseline').to('cuda:0')
    atomic_json(run/'resolved_config.json',dict(job=job,config=cfg,split_bounds=bundle.split_bounds,
        scaler=bundle.scaler.state_dict(),feature_columns=bundle.feature_columns,n_vars=bundle.n_vars,
        native_config=getattr(model,'native_config',None),
        n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    trainer=Trainer(model,optimizer,scheduler,torch.device('cuda:0'),cfg,run,create_logger(run/'train.log'))
    from scripts.paper_resume_training import fit_with_resume
    cp,best,elapsed=fit_with_resume(trainer,bundle.train_loader,bundle.val_loader,job['seed'])
    checkpoint=load_checkpoint(cp,model,map_location='cuda:0')
    metrics=evaluate_test(model,bundle.test_loader,torch.device('cuda:0'),bundle.scaler,32,model.quantiles,run)
    # Local-change diagnostics are an ETTh1 study; avoid a redundant high-dimensional test pass.
    details={}
    if bundle.n_vars<=16:
        details=evaluate_details(model,bundle.test_loader,torch.device('cuda:0'),training_jump_thresholds(bundle),bundle.scaler)
        assert abs(details['pinball']-metrics['QuantileLoss_standardized'])<1e-6
    record=dict(job=job,source_dir=str(run.resolve()),best_val=best,best_epoch=checkpoint['epoch'],
        epochs_run=len(pd.read_csv(run/'training_history.csv')),training_seconds=elapsed,
        metrics=metrics,details=details,test_used_for_selection=False)
    record['artifacts']={f:sha256(run/f) for f in ['best_checkpoint.pt','resolved_config.json','training_history.csv','metrics.json','horizon_metrics.csv']}
    verify(root);atomic_json(run/'complete.json',record)
    return record

def worker(root):
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];status=root/'workers'/f'gpu{gpu}.json'
    while True:
        with (root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            q=json.loads((root/'queue.json').read_text());i=q['next']
            if i>=len(q['jobs']):return
            job=q['jobs'][i]
            # All ETT runs must finish before any new large-dataset run starts.
            if not job['dataset'].startswith('ETT'):
                remaining=[j for j in q['jobs'] if j['dataset'].startswith('ETT') and not (path_for(root,j)/'complete.json').exists()]
                if remaining:
                    atomic_json(status,dict(status='waiting_for_ett',remaining=len(remaining),pid=os.getpid()))
                    time.sleep(5)
                    continue
            q['next']+=1;atomic_json(root/'queue.json',q)
        atomic_json(status,dict(status='running',job=job,pid=os.getpid()))
        try:run_job(root,job)
        except BaseException as exc:
            atomic_json(status,dict(status='failed',job=job,error=str(exc)));raise
        atomic_json(status,dict(status='waiting',last_job=job))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--worker',action='store_true')
    args=p.parse_args();root=args.root
    if args.worker:worker(root);return
    root.mkdir(parents=True,exist_ok=True);(root/'workers').mkdir(exist_ok=True)
    while True:
        status=json.loads((BASE/'progress.json').read_text())
        if status.get('status')=='complete':break
        states=[json.loads(f.read_text()) for f in (BASE/'workers').glob('gpu*.json')]
        if any(r.get('status')=='failed' for r in states):raise RuntimeError('ETTh1 stage failed; fix the original stage first')
        atomic_json(root/'progress.json',dict(status='waiting_for_etth1',upstream=status,datasets=list(DATASETS),additional_training_jobs=360))
        time.sleep(10)
    selection=json.loads((BASE/'selection.json').read_text())['learning_rates']
    protocol=dict(source=source_manifest(),driver=sha256(__file__),
        resume_driver_sha256=sha256('scripts/paper_resume_training.py'),
        amendment_sha256=sha256(root/'epoch_cap_amendment.json'),
        previous_protocol_sha256=sha256(root/'protocol.json'),
        helper_driver_sha256=sha256('scripts/complete_paper_experiments.py'),
        config_sha256={str(f):sha256(f) for f in [Path('configs/optimized_etth1.yaml')]+[Path(f'configs/baselines/{m}.yaml') for m in MODELS]},
        data_sha256={f:sha256(f) for f in FILES.values()},
        baseline_selection_sha256=sha256(BASE/'selection.json'),datasets=list(DATASETS),history=336,
        horizons=list(HORIZONS),seeds=list(SEEDS),baseline_learning_rates=selection,
        selection='All model structures and learning rates frozen on ETTh1 validation; no retuning on other datasets or test data.',
        full_model='HADT-NQ: unified linear_skip',gpus=[0,1,2,3],epoch_caps=EPOCH_CAPS,early_stopping=7,
        default_batch_size=32,resource_policy='On Traffic/Electricity: PatchTST/iTransformer train batch8; TimeMixer batch2/4; HADT and DLinear batch32. Evaluation at most16. HADT samples16 channels for C>16; other backbones preserve full native variable context. HADT evaluation chunks32 over ALL variables.',
        high_dim_memory_check_sha256=sha256('outputs/paper_revision_20261126/high_dim_memory_checks.json'))
    if (root/'protocol_v2.json').exists():assert json.loads((root/'protocol_v2.json').read_text())==protocol
    else:atomic_json(root/'protocol_v2.json',protocol)
    jobs=[]
    # Keep the dataset/model matrix complete, using dynamic scheduling across four GPUs.
    for d in DATASETS:
        for h in HORIZONS:
            for s in SEEDS:
                for m in ['hadt',*MODELS]:
                    j=dict(dataset=d,horizon=h,seed=s,name=m,kind='full' if m=='hadt' else 'baseline',epochs=EPOCH_CAPS[d])
                    if m!='hadt':j['lr']=selection[m]
                    jobs.append(j)
    # Schedule costly full-context models early to reduce a long tail.
    jobs.sort(key=lambda j:(0 if j['dataset'].startswith('ETT') else 1, 0 if j['dataset'] in ('Traffic','Electricity') and j['name']=='timemixer' else 1, -j['horizon'],j['dataset'],j['seed'],j['name']))
    atomic_json(root/'queue.json',dict(jobs=jobs,next=0));children=[]
    for gpu in (0,1,2,3):
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/'workers'/f'gpu{gpu}.log').open('w')
        proc=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((proc,log))
    try:
        while any(p.poll() is None for p,l in children):
            if any(p.poll() not in (None,0) for p,l in children):raise RuntimeError('Seven-dataset worker failed; inspect logs')
            done=sum((path_for(root,j)/'complete.json').exists() for j in jobs)
            atomic_json(root/'progress.json',dict(status='running',completed=done,total=360,datasets=list(DATASETS)))
            time.sleep(5)
        if any(p.returncode for p,l in children):raise RuntimeError('Seven-dataset training failed')
    finally:
        for p,l in children:
            if p.poll() is None:p.terminate()
            p.wait();l.close()
    records=[json.loads((path_for(root,j)/'complete.json').read_text()) for j in jobs]
    atomic_json(root/'records.json',records)
    atomic_json(root/'progress.json',dict(status='complete',completed=360,total=360,datasets=list(DATASETS)))
    print('Six additional datasets complete: 360 new full-test runs.',flush=True)

if __name__=='__main__':main()
