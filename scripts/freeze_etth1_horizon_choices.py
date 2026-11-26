"""Freeze per-horizon validation choices before any test, without changing global selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.detail_search import HORIZONS,SEEDS,scores
from scripts.run_etth1_detail_search import verify_protocol,freeze_json
from utils.provenance import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/etth1_detail_search'));args=p.parse_args();root=args.root
    verify_protocol(root);records=json.loads((root/'confirm_records.json').read_text())
    global_choice=json.loads((root/'selection.json').read_text());choices={}
    for h in HORIZONS:
        rows=[]
        baseline=[r for r in records if r['candidate']=='reference' and r['horizon']==h]
        for name in sorted({r['candidate'] for r in records}):
            group=[r for r in records if r['candidate']==name and r['horizon']==h]
            assert len(group)==3 and {r['seed'] for r in group}==set(SEEDS)
            means={m:float(np.mean([r['validation'][m] for r in group])) for m in ['pinball','delta_mae','jump_mae','PICP90']}
            ratios={m:means[m]/np.mean([r['validation'][m] for r in baseline]) for m in ['pinball','delta_mae','jump_mae']}
            rows.append(dict(candidate=name,**means,score=float(.7*ratios['pinball']+.2*ratios['delta_mae']+.1*ratios['jump_mae'])))
        accuracy=min(rows,key=lambda r:(r['pinball'],r['candidate']))
        coverage=np.mean([r['validation']['PICP90'] for r in baseline])
        eligible=[r for r in rows if r['pinball']<=accuracy['pinball']*1.02 and r['PICP90']>=coverage-.02]
        detail=min(eligible,key=lambda r:(r['score'],r['candidate'])) if eligible else accuracy
        choices[str(h)]=dict(accuracy=accuracy['candidate'],detail=detail['candidate'],constraints_satisfied=bool(eligible),ranking=rows)
    result=dict(history=global_choice['history'],choices=choices,test_used=False,
                timing='Additional per-horizon deployment rule declared during confirmation and frozen before first test; original global selection unchanged',
                global_selection_sha256=sha256(root/'selection.json'))
    destination=root/'horizon_selection.json'
    if not destination.exists() and list((root/'confirm').rglob('test_complete.json')):
        raise RuntimeError('Cannot define new horizon choices after testing')
    freeze_json(destination,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
