"""Exploratory K sensitivity on ETTh1 only; primary K=8 is never retuned."""
import _thread,argparse,json,os,sys,threading,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import scripts.run_paper_experiments as study

if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('PAPER_EXPECTED_GPU','3'):raise RuntimeError('GPU3 required')
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument('--stop-before-utc',default='2099-12-31T23:59:00+00:00')
    args=parser.parse_args()
    cutoff=datetime.fromisoformat(args.stop_before_utc).timestamp()
    timer=threading.Timer(max(.1,cutoff-time.time()),_thread.interrupt_main);timer.daemon=True;timer.start()
    original=study.configuration
    h=study.code_digest();records=[]
    destination=Path('outputs/paper_sensitivity');destination.mkdir(parents=True,exist_ok=True)
    try:
        for k in [4,12]:
            def configuration(dataset,method,k=k):
                c=original(dataset,method)
                c['model']['n_points']=k
                c['experiment']['run_name']=f'paper_sensitivity/K{k}/{method}'
                c['paper']['sensitivity_K']=k
                return c
            study.configuration=configuration
            study.ROOT=destination/f'K{k}'
            for seed in [2024,2025,2026]:
                if cutoff-time.time()<180:break
                print('K sensitivity',k,seed,flush=True)
                r=study.run('ETTh1','hadt',96,seed,torch.device('cuda:0'),h)
                r['sensitivity_K']=k;r['study_group']='sampling_sensitivity';records.append(r)
                (destination/'records.json').write_text(json.dumps(records,indent=2))
                print('DONE K',k,seed,r['QuantileLoss_standardized'],flush=True)
    except KeyboardInterrupt:
        print('Sensitivity cutoff: unfinished runs excluded.',flush=True)
    finally:timer.cancel()
