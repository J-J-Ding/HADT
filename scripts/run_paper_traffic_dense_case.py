from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.run_paper_experiments import run,code_digest

if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('GPU3 required')
    torch.set_num_threads(4)
    print('TRAFFIC dense case',flush=True)
    r=run('Traffic','dense',96,2024,torch.device('cuda:0'),code_digest())
    r['study_group']='traffic_case'
    Path('outputs/paper_study/traffic_dense_records.json').write_text(json.dumps([r],indent=2))
    print('DONE dense',r['QuantileLoss_standardized'],flush=True)
