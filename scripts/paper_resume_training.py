"""Resume paper runs from saved validation-best epochs without editing frozen core code."""
import csv,inspect,random,textwrap,types,time
from pathlib import Path
import numpy as np
import torch
import engine.trainer as trainer_module
from engine.checkpoint import load_checkpoint
from utils.provenance import atomic_json,sha256

def fit_with_resume(trainer,train_loader,val_loader,seed):
    run=Path(trainer.run_dir);checkpoint_path=run/'best_checkpoint.pt';history_path=run/'training_history.csv'
    prefix=[];best=float('inf');start=1;elapsed=0.;resume=None
    if checkpoint_path.exists() and history_path.exists():
        with history_path.open() as stream:history=list(csv.DictReader(stream))
        cp=load_checkpoint(checkpoint_path,trainer.model,map_location=trainer.device)
        epoch=int(cp['epoch'])
        if epoch>trainer.epochs:raise RuntimeError('Saved best epoch exceeds amended cap; cannot reuse it.')
        if history and int(history[-1]['epoch'])>=trainer.epochs:
            # At the cap, use the best checkpoint within the completed budget.
            return checkpoint_path,float(cp['best_val_loss']),float(history[-1]['elapsed_seconds'])
        for row in history:
            if int(row['epoch'])<=epoch:
                prefix.append({k:int(v) if k=='epoch' else float(v) for k,v in row.items()})
        if not prefix or prefix[-1]['epoch']!=epoch:raise ValueError('Checkpoint and history do not agree')
        trainer.optimizer.load_state_dict(cp['optimizer_state'])
        if cp.get('scheduler_state') is not None:trainer.scheduler.load_state_dict(cp['scheduler_state'])
        if cp.get('amp_scaler_state'):trainer.scaler.load_state_dict(cp['amp_scaler_state'])
        best=float(cp['best_val_loss']);start=epoch+1;elapsed=float(prefix[-1]['elapsed_seconds'])
        resume=dict(checkpoint_sha256=sha256(checkpoint_path),retained_epochs=epoch,start_epoch=start,cap=trainer.epochs)
        rng=cp.get('rng_state')
        if rng:
            random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            if torch.cuda.is_available():torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
            resume['rng_policy']='restored checkpoint RNG state'
        else:
            from utils.seed import set_seed
            resume_seed=int(seed)+100000*epoch;set_seed(resume_seed)
            resume.update(rng_policy='Legacy checkpoint has no RNG state; deterministic reseed, not bitwise uninterrupted continuation.',resume_seed=resume_seed)
        atomic_json(run/'resume_metadata.json',resume)
        if cp.get('config',{}).get('training',{}).get('epochs')!=trainer.epochs:
            cp['epoch_cap_amendment']=dict(previous_cap=cp['config']['training']['epochs'],
                active_cap=trainer.epochs,previous_checkpoint_sha256=resume['checkpoint_sha256'],
                model_weights_unchanged=True)
            cp['config']=trainer.config
            cp['rng_state']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            temp=checkpoint_path.with_name('amended_checkpoint.tmp.pt')
            torch.save(cp,temp);temp.replace(checkpoint_path)
        trainer.logger.info('Resuming from best epoch %s, continuing at %s/%s',epoch,start,trainer.epochs)
    # Keep the exact core epoch body/objective; only restore loop state and save RNGs.
    source=textwrap.dedent(inspect.getsource(trainer_module.Trainer.fit))
    changes={
        'best_val = math.inf':'best_val = _resume_best',
        'start_time = time.time()':'start_time = time.time() - _resume_elapsed',
        'history = []':'history = list(_resume_history)',
        'range(1, self.epochs + 1)':'range(_resume_start, self.epochs + 1)',
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise RuntimeError('Frozen Trainer.fit changed; resume patch requires review: '+old)
        source=source.replace(old,new)
    original_save=trainer_module.save_checkpoint
    def save_with_rng(path,*args,**kwargs):
        temp=Path(path).with_name('resume_checkpoint.tmp.pt')
        original_save(temp,*args,**kwargs)
        cp=torch.load(temp,map_location='cpu',weights_only=False)
        cp['rng_state']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(cp,temp);temp.replace(path)
    namespace=dict(vars(trainer_module))
    namespace.update(_resume_best=best,_resume_elapsed=elapsed,_resume_history=prefix,_resume_start=start,save_checkpoint=save_with_rng)
    exec(compile(source,str(Path(__file__).resolve())+':resumed_fit','exec'),namespace)
    return types.MethodType(namespace['fit'],trainer)(train_loader,val_loader)
