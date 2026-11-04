"""Time-bounded Traffic repetitions; partial groups are never reported as three-seed means."""
import _thread,json,os,sys,threading,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.run_paper_experiments import run,code_digest

if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('GPU3 required')
    torch.set_num_threads(4)
    cutoff=datetime.fromisoformat('2026-10-30T16:05:00+00:00').timestamp()
    timer=threading.Timer(max(.1,cutoff-time.time()),_thread.interrupt_main);timer.daemon=True;timer.start()
    rows=[];target=Path('outputs/paper_study/traffic_repeat_records.json');h=code_digest()
    try:
        for method,seed in [('linear',2025),('linear',2026),('hadt',2025),('hadt',2026)]:
            if method=='hadt':
                while not Path('outputs/paper_study/dense/Traffic/H96/seed2024/complete.json').exists():
                    time.sleep(3)
                if cutoff-time.time()<1550:
                    print('Not enough time for a full Traffic run; stop before',method,seed,flush=True);break
            print('REPEAT Traffic',method,seed,flush=True)
            r=run('Traffic',method,96,seed,torch.device('cuda:0'),h)
            r['study_group']='traffic_repetition';rows.append(r)
            target.write_text(json.dumps(rows,indent=2))
            print('DONE',method,seed,r['QuantileLoss_standardized'],flush=True)
    except KeyboardInterrupt:
        print('Traffic repetition deadline reached; incomplete run excluded.',flush=True)
    finally:timer.cancel()
