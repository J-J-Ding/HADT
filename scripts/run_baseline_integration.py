"""Finite ETTh1/H96 integration checks, explicitly excluded from paper benchmarks."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from utils.config import load_config, validate_config
from utils.provenance import atomic_json
from experiments.runner import run_single_experiment
from scripts.plot_run_forecasts import write_gallery


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--matrix',default='configs/baseline_comparison.yaml')
    parser.add_argument('--run-name',default='baseline_integration_v1')
    args=parser.parse_args()
    if os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES')!='3':
        raise RuntimeError('Set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3')
    if Path(sys.prefix).name!='djjTF' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('djjTF and exactly one visible CUDA device required')
    matrix=yaml.safe_load(Path(args.matrix).read_text())
    rows=[];root=Path('outputs')/args.run_name;root.mkdir(parents=True,exist_ok=True)
    for config_path in matrix['configs']:
        c=load_config(config_path)
        c['training']['epochs']=matrix['integration_epochs']
        c['experiment'].update(run_name=args.run_name,scope='integration_only_one_epoch_full_ETTh1_splits_not_paper_benchmark')
        validate_config(c)
        result=run_single_experiment(c,matrix['dataset'],matrix['pred_len'],matrix['seeds'][0])
        rows.append(result)
        atomic_json(root/'integration_summary.json',dict(scope=c['experiment']['scope'],completed=len(rows),total=len(matrix['configs']),results=rows))
    records=[json.loads(p.read_text()) for p in sorted(root.rglob('complete.json'))]
    write_gallery(records,root/'index.html')
    print('Integration complete:',root/'index.html',flush=True)


if __name__=='__main__':main()
