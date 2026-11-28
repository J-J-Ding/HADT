"""Four-GPU V4 queue, gated on completed primary comparisons; no core edits."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.provenance import atomic_json,sha256,source_manifest
OUT=ROOT/'outputs/reviewer_revision_20261128'


def read(p):return json.loads(Path(p).read_text())
def run_path(job):
    name=job['id'] if job['phase']=='validation_pilot' else job['variant']
    return OUT/job['phase']/name/f"H{job['horizon']}"/f"seed{job['seed']}"


def verify():
    p=read(OUT/'protocol.json')
    assert source_manifest()==p['source'],'Frozen core source changed'
    for f,digest in p['files'].items():assert sha256(f)==digest,f


def choose_calibration():
    selection=OUT/'selection.json'
    if selection.exists():return read(selection)
    jobs=[j for j in read(OUT/'jobs.json')['jobs'] if j['phase']=='validation_pilot']
    if not all((run_path(j)/'complete.json').exists() for j in jobs):return None
    baseline=read(ROOT/'outputs/etth1_detail_search/confirm/L336/linear_skip/H96/seed2024/test_complete.json')['validation']['pinball']
    candidates=[]
    for j in jobs:
        r=read(run_path(j)/'complete.json');metrics=r['validation']
        candidates.append(dict(weight=j['changes']['calibration_lambda'],ace=metrics['ACE'],
            pinball=metrics['QuantileLoss_standardized'],record=str(run_path(j)/'complete.json'),record_sha256=sha256(run_path(j)/'complete.json')))
    eligible=[r for r in candidates if r['pinball']<=1.02*baseline]
    chosen=min(eligible or candidates,key=lambda r:(r['ace'],r['pinball'],r['weight']))
    result=dict(status='frozen',weight=chosen['weight'],temperature=.1,baseline_validation_pinball=baseline,
        pinball_constraint_satisfied=bool(eligible),chosen=chosen,candidates=candidates,
        rule='Validation-only ACE minimum within 2% pinball constraint; fallback ACE diagnostic if all violate; pinball then smaller weight break ties.',test_used_for_selection=False)
    atomic_json(selection,result);return result


def run_job(job):
    import torch
    import yaml
    from utils.seed import set_seed
    from utils.model_registry import build_model
    from utils.logger import create_logger
    from engine.trainer import Trainer
    from engine.checkpoint import load_checkpoint
    from data_provider.data_factory import build_data_bundle
    from utils.forecast_detail_metrics import evaluate_details,training_jump_thresholds
    from scripts.paper_resume_training import fit_with_resume
    from scripts.reviewer_components import install_interior_grid,CalibrationObjective,ObjectiveLoggingLoader,evaluate,attention_diagnostics
    verify();run=run_path(job);run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        r=read(run/'complete.json');assert r['job']==job
        for f,digest in r['artifacts'].items():assert sha256(run/f)==digest
        for f,digest in r.get('external_artifacts',{}).items():assert sha256(f)==digest
        return
    set_seed(job['seed']);torch.set_num_threads(3);device=torch.device('cuda:0')
    reuse=job['phase']=='diagnostic_reuse'
    if reuse:
        source=Path(job['source_record']);source_record=read(source);source_run=source.parent
        hashes=source_record['artifact_sha256'] if job['variant']=='full' else source_record['artifacts']
        assert sha256(source_run/'best_checkpoint.pt')==hashes['best_checkpoint.pt']
        resolved=read(source_run/'resolved_config.json');cfg=resolved['config']
    else:
        cfg=yaml.safe_load((ROOT/'configs/optimized_etth1.yaml').read_text())
        cfg['training']['epochs']=job['epochs'];cfg['visualization']['enabled']=False
        for key in ('n_points','max_offset'):
            if key in job['changes']:cfg['model']['params'][key]=job['changes'][key]
        cfg['reviewer_study']=dict(job=job,selection=None)
        if job['variant']=='calibration':
            if job['phase']=='formal':
                selected=choose_calibration();assert selected is not None
                weight=selected['weight'];cfg['reviewer_study']['selection']=dict(file=str(OUT/'selection.json'),sha256=sha256(OUT/'selection.json'))
            else:weight=job['changes']['calibration_lambda']
            cfg['reviewer_study'].update(calibration_weight=weight,temperature=.1)
    bundle=build_data_bundle(cfg,'ETTh1',job['horizon'])
    cfg=copy.deepcopy(cfg);cfg['data_scaler']=bundle.scaler.state_dict();cfg['feature_columns']=bundle.feature_columns
    model=build_model(cfg,job['horizon'],bundle.n_vars).to(device)
    if job['variant']=='interior_grid':install_interior_grid(model)
    if reuse and job['variant']=='fixed':model.ablation='no_offsets'
    resolved=dict(job=job,config=cfg,split_bounds=bundle.split_bounds,scaler=bundle.scaler.state_dict(),
        feature_columns=bundle.feature_columns,n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        training_history_note='For calibration, legacy train_pinball column denotes combined objective; objective_components.csv contains separated terms.')
    if (run/'resolved_config.json').exists():assert read(run/'resolved_config.json')==resolved,'Resume configuration changed'
    else:atomic_json(run/'resolved_config.json',resolved)
    result=dict(job=job,test_used_for_selection=False)
    if reuse:
        checkpoint=load_checkpoint(source_run/'best_checkpoint.pt',model,map_location=device)
        result.update(best_epoch=checkpoint['epoch'],training_seconds=0,
            external_artifacts={str(source):sha256(source),str(source_run/'best_checkpoint.pt'):sha256(source_run/'best_checkpoint.pt')})
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
        objective=CalibrationObjective(weight,.1) if job['variant']=='calibration' else None
        trainer=Trainer(model,optimizer,scheduler,device,cfg,run,create_logger(run/'train.log'),train_objective=objective)
        train_loader=bundle.train_loader
        if objective is not None:
            if (run/'best_checkpoint.pt').exists():objective.epoch=int(torch.load(run/'best_checkpoint.pt',map_location='cpu',weights_only=False)['epoch'])
            train_loader=ObjectiveLoggingLoader(train_loader,objective,run)
        cp,best,elapsed=fit_with_resume(trainer,train_loader,bundle.val_loader,job['seed'])
        checkpoint=load_checkpoint(cp,model,map_location=device)
        result.update(best_val=best,best_epoch=checkpoint['epoch'],training_seconds=elapsed)
    if job['phase']=='validation_pilot':
        result['validation']=evaluate(model,bundle.val_loader,device)
        atomic_json(run/'validation.json',result['validation'])
    else:
        metrics=evaluate(model,bundle.test_loader,device,bundle,run,'Test reviewer')
        details=evaluate_details(model,bundle.test_loader,device,training_jump_thresholds(bundle),bundle.scaler)
        assert abs(details['pinball']-metrics['QuantileLoss_standardized'])<1e-6
        if reuse:
            original=source_record['test'] if job['variant']=='full' else source_record['metrics']
            assert abs(original['QuantileLoss_standardized']-metrics['QuantileLoss_standardized'])<1e-6
        result.update(metrics=metrics,details=details,diagnostics=attention_diagnostics(model,bundle,device,run))
    files=['resolved_config.json']
    files+=['validation.json'] if job['phase']=='validation_pilot' else ['metrics.json','horizon_metrics.csv','quantile_calibration.npz','attention_diagnostics.json','attention_diagnostics.npz']
    if not reuse:files+=['best_checkpoint.pt','training_history.csv']
    if (run/'objective_components.csv').exists():files+=['objective_components.csv']
    result['artifacts']={f:sha256(run/f) for f in files}
    verify();atomic_json(run/'complete.json',result)


def worker():
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];status=OUT/'workers'/f'gpu{gpu}.json'
    while True:
        with (OUT/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);queue=read(OUT/'queue.json');i=queue['next']
            if i>=len(queue['jobs']):return
            job=queue['jobs'][i]
            if job.get('requires') and choose_calibration() is None:
                atomic_json(status,dict(status='waiting_for_calibration_selection',pid=os.getpid()))
                time.sleep(3);continue
            queue['next']+=1;atomic_json(OUT/'queue.json',queue)
        atomic_json(status,dict(status='running',job=job,pid=os.getpid()))
        try:run_job(job)
        except BaseException as error:
            atomic_json(status,dict(status='failed',job=job,error=str(error),pid=os.getpid()));raise
        atomic_json(status,dict(status='waiting',last_job=job,pid=os.getpid()))


def main():
    os.chdir(ROOT);p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--smoke-only',action='store_true');args=p.parse_args()
    if args.worker:worker();return
    from scripts.reviewer_components import smoke_checks
    if args.smoke_only:
        atomic_json(OUT/'smoke_checks.json',smoke_checks());print('Reviewer CPU checks passed.');return
    lock=(OUT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    while read(ROOT/'outputs/paper_seven_datasets_20261126/progress.json').get('status')!='complete':
        atomic_json(OUT/'progress.json',dict(status='waiting_for_primary',formal_completed=0,formal_total=36));time.sleep(30)
    assert read(ROOT/'outputs/paper_completion_20261126/progress.json')['status']=='complete'
    manifest=read(OUT/'planned_jobs.json')
    assert sha256(ROOT/manifest['spec'])==manifest['spec_sha256'],'V4 spec changed after task planning'
    protocol=dict(source=source_manifest(),files={str(f):sha256(f) for f in [
        ROOT/'scripts/reviewer_experiments.py',ROOT/'scripts/reviewer_components.py',ROOT/'scripts/paper_resume_training.py',
        ROOT/'configs/optimized_etth1.yaml',ROOT/'dataset/ETT-small/ETTh1.csv',OUT/'planned_jobs.json',ROOT/manifest['spec']]},
        spec=manifest['spec'],spec_sha256=manifest['spec_sha256'],formal=36,pilots=3,reused_diagnostic_evaluations=12)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:atomic_json(OUT/'protocol.json',protocol)
    if not (OUT/'smoke_checks.json').exists():atomic_json(OUT/'smoke_checks.json',smoke_checks())
    planned=manifest['jobs']
    jobs=[j for j in planned if j['phase']=='validation_pilot']
    jobs+=[j for j in planned if j['phase']=='formal' and j['variant']!='calibration']
    jobs+=[dict(id=f"reuse_{r['variant']}_H{r['horizon']}_seed{r['seed']}",phase='diagnostic_reuse',
        variant=r['variant'],horizon=r['horizon'],seed=r['seed'],dataset='ETTh1',source_record=r['record'],requires=[]) for r in manifest['reuse']]
    jobs+=[j for j in planned if j['phase']=='formal' and j['variant']=='calibration']
    assert len(jobs)==51
    atomic_json(OUT/'jobs.json',dict(jobs=jobs));atomic_json(OUT/'queue.json',dict(jobs=jobs,next=0))
    children=[]
    try:
        (OUT/'workers').mkdir(exist_ok=True)
        for gpu in range(4):
            env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
            log=(OUT/'workers'/f'gpu{gpu}.log').open('a')
            proc=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((proc,log))
        while any(p.poll() is None for p,_ in children):
            if any(p.poll() not in (None,0) for p,_ in children):raise RuntimeError('Reviewer worker failed')
            finished=[j for j in jobs if (run_path(j)/'complete.json').exists()]
            atomic_json(OUT/'progress.json',dict(status='running',formal_completed=sum(j['phase']=='formal' for j in finished),formal_total=36,
                pilots_completed=sum(j['phase']=='validation_pilot' for j in finished),pilots_total=3,
                reused_diagnostics_completed=sum(j['phase']=='diagnostic_reuse' for j in finished),reused_diagnostics_total=12))
            time.sleep(5)
        assert all(p.returncode==0 for p,_ in children)
        assert all((run_path(j)/'complete.json').exists() for j in jobs)
        atomic_json(OUT/'records.json',[read(run_path(j)/'complete.json') for j in jobs])
        atomic_json(OUT/'progress.json',dict(status='complete',formal_completed=36,formal_total=36,pilots_completed=3,reused_diagnostics_completed=12))
    finally:
        for p,log in children:
            if p.poll() is None:p.terminate()
            p.wait();log.close()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--worker' not in sys.argv:atomic_json(OUT/'progress.json',dict(status='failed',error=str(error)))
        raise
