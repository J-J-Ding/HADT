"""Compare existing HADT checkpoints against the pre-refactor Git snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from experiments.runner import _build_model
from experiments.architecture_search import build_candidate
from utils.provenance import atomic_json

SNAPSHOT='1fbada44c526599f37bdbfac3f4b3fd72291eb02'
CHILD=r'''
import sys,torch
from experiments.runner import _build_model
from models.architecture_search import build_candidate
r=torch.load(sys.argv[1],weights_only=False,map_location='cpu')
torch.set_num_threads(2)
m=build_candidate(r['config'],96,r['candidate']) if r['candidate'] else _build_model(r['config'],96)
m.load_state_dict(r['state'],strict=True);m.eval()
with torch.no_grad():q,aux=m(r['x'],return_aux=True)
torch.save(dict(q=q,aux=aux,keys=list(m.state_dict())),sys.argv[2])
'''


def verify(output):
    torch.set_num_threads(2)
    cases=[('transformer3',Path('outputs/architecture_study/transfer/transformer3/ETTh2/H96/seed2024/best_checkpoint.pt'),None),
           ('mlp64_lr3e4',Path('outputs/architecture_study/confirm/mlp64_lr3e4/ETTh1/H96/seed2024/best_checkpoint.pt'),
            json.loads(Path('configs/selected_architecture.json').read_text())['candidate'])]
    rows=[]
    with tempfile.TemporaryDirectory(prefix='hadt-legacy-parity-') as tmp:
        root=Path(tmp);archive=root/'snapshot.tar';tree=root/'old';tree.mkdir()
        subprocess.run(['git','archive','--format=tar','--output',str(archive),SNAPSHOT,
                        'models','utils','engine','experiments','data_provider','losses','metrics','configs'],check=True)
        with tarfile.open(archive) as tar:tar.extractall(tree)
        for name,path,candidate in cases:
            cp=torch.load(path,map_location='cpu',weights_only=False)
            torch.manual_seed(731);x=torch.randn(2,cp['config']['data']['seq_len'],3)
            model=build_candidate(cp['config'],96,candidate) if candidate else _build_model(cp['config'],96)
            model.load_state_dict(cp['model_state'],strict=True);model.eval()
            with torch.no_grad():q,aux=model(x,return_aux=True)
            input_file=root/(name+'-input.pt');result_file=root/(name+'-output.pt')
            torch.save(dict(config=cp['config'],candidate=candidate,state=cp['model_state'],x=x),input_file)
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            subprocess.run([sys.executable,'-c',CHILD,str(input_file),str(result_file)],cwd=tree,env=env,check=True,capture_output=True,text=True)
            old=torch.load(result_file,weights_only=False)
            assert list(model.state_dict())==old['keys']
            assert torch.equal(q,old['q'])
            assert set(aux)==set(old['aux'])
            for key,value in aux.items():
                if torch.is_tensor(value):assert torch.equal(value,old['aux'][key]),key
                else:assert value==old['aux'][key]
            row=dict(model=name,checkpoint=str(path),snapshot=SNAPSHOT,strict_loading=True,
                     state_keys_identical=True,prediction_and_aux_bitwise_equal=True,passed=True)
            rows.append(row);print(json.dumps(row),flush=True)
    atomic_json(output,dict(cases=rows,passed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/baseline_refactor_validation/legacy_regression.json'))
    verify(parser.parse_args().output)
