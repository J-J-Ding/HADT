"""Retrain an exported optimized configuration with its declared auxiliary objective."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from utils.provenance import source_manifest,sha256
from scripts.run_etth1_detail_search import execute_stage,jobs_for,freeze_json
from experiments.detail_search import configuration,HORIZONS,SEEDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/optimized_etth1.yaml'))
    p.add_argument('--root',type=Path,default=Path('outputs/optimized_etth1_retrain'))
    p.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3]);args=p.parse_args()
    config=yaml.safe_load(args.config.read_text());candidate=config['optimization']['candidate']
    length=config['data']['seq_len'];epochs=config['training']['epochs']
    expected=configuration(candidate,length,epochs,'confirm')
    if config['model']!=expected['model']:
        raise ValueError('Model fields differ from the exported candidate; update optimization.candidate consistently')
    for key in ('batch_size','learning_rate','weight_decay','difference_weight'):
        if config['training'][key]!=expected['training'][key]:raise ValueError(f'Unsupported inconsistent override: {key}')
    args.root.mkdir(parents=True,exist_ok=True);(args.root/'workers').mkdir(exist_ok=True)
    protocol=dict(source=source_manifest(),driver_sha256=sha256('scripts/run_etth1_detail_search.py'),
                  data_sha256=sha256('dataset/ETT-small/ETTh1.csv'),config_sha256=sha256(args.config),
                  scope='Reproduce a previously validation-selected configuration; no new model selection')
    freeze_json(args.root/'protocol.json',protocol)
    horizons=config['experiment']['horizons']
    jobs=[j for j in jobs_for('confirm',[candidate],[length],SEEDS,epochs) if j['horizon'] in horizons]
    execute_stage(args.root,'confirm',jobs,args.gpus)
    freeze_json(args.root/'selection.json',dict(history=length,accuracy=candidate['id'],detail=candidate['id'],
                recommended=candidate['id'],test_used=False,scope='Previously selected configuration retrain'))
    subprocess.run([sys.executable,'scripts/evaluate_etth1_detail_search.py','--root',str(args.root),
                    '--selected-only','--horizons',*[str(h) for h in horizons],'--gpus',*[str(g) for g in args.gpus]],check=True)


if __name__=='__main__':main()
