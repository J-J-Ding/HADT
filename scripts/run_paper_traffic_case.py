"""Independent cheap Traffic control; its records are merged by paper aggregation."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.run_paper_experiments import run,code_digest

if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('GPU3 required')
    torch.set_num_threads(4)
    h=code_digest();records=[]
    for method in ['hadt','linear']:
        print('TRAFFIC CASE',method,flush=True)
        r=run('Traffic',method,96,2024,torch.device('cuda:0'),h)
        r['study_group']='traffic_case';records.append(r)
        Path('outputs/paper_study/traffic_case_records.json').write_text(json.dumps(records,indent=2))
        print('DONE',method,r['QuantileLoss_standardized'],flush=True)
