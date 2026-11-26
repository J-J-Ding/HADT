"""Compare vendored point backbones to pinned upstream in isolated subprocesses.

Usage: python scripts/verify_baseline_parity.py --upstream /tmp/hadt-baseline-upstream
The upstream folders are read-only; no project imports are injected into them.
An unused optional Reformer import in iTransformer is stubbed only in the child
when unavailable; its FullAttention path never invokes that class.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.config import load_config
from utils.model_registry import build_model
from utils.provenance import atomic_json

CHILD = r'''
import importlib,json,sys,types
from pathlib import Path
import torch
# This optional dependency belongs to unused ReformerLayer, not iTransformer.
try:
    import reformer_pytorch
    optional_stub=False
except ImportError:
    dummy=types.ModuleType('reformer_pytorch')
    class UnusedReformer:
        def __init__(self,*a,**k):raise RuntimeError('Unused Reformer must never execute')
    dummy.LSHSelfAttention=UnusedReformer
    sys.modules['reformer_pytorch']=dummy
    optional_stub=True
payload=torch.load(sys.argv[1],weights_only=False,map_location='cpu')
torch.set_num_threads(2)
module=importlib.import_module(sys.argv[3])
m=module.Model(types.SimpleNamespace(**payload['native_config'])).eval()
m.load_state_dict(payload['state'],strict=True)
with torch.no_grad():
    x=payload['x'].clone()
    y=m(x,None,None,None) if payload['multi'] else m(x)
    if isinstance(y,tuple):y=y[0]
torch.save({'output':y,'optional_import_stub':optional_stub},sys.argv[2])
'''


def verify(upstream, output):
    manifest=json.loads(Path('third_party/BASELINE_SOURCES.json').read_text())
    torch.set_num_threads(2);rows=[]
    cases=[('dlinear',{}),('dlinear',{'individual':True}),('patchtst',{}),
           ('patchtst',{'decomposition':True,'individual':True}),
           ('itransformer',{}),('itransformer',{'use_norm':False,'output_attention':True}),
           ('timemixer',{}),('timemixer',{'channel_independence':0,'down_sampling_method':'max'})]
    with tempfile.TemporaryDirectory(prefix='hadt-parity-') as tmp:
        for index,(name,extra) in enumerate(cases):
            repo=upstream/name
            revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            if revision!=manifest[name]['commit']:
                raise ValueError(f'Wrong upstream commit: {name}')
            c=load_config(f'configs/baselines/{name}.yaml');c['data']['seq_len']=96
            c['model']['params'].update(extra);c['model']['output_mode']='point'
            torch.manual_seed(817+index);model=build_model(c,24,5).eval();x=torch.randn(2,96,5)
            with torch.no_grad():expected=model(x)
            payload=Path(tmp)/f'{index}-input.pt';result=Path(tmp)/f'{index}-output.pt'
            torch.save(dict(native_config=model.native_config,state=model.point_model.state_dict(),x=x,multi=model.multi_argument),payload)
            cwd=repo/'PatchTST_supervised' if name=='patchtst' else repo
            module={'patchtst':'models.PatchTST','dlinear':'models.DLinear','itransformer':'model.iTransformer','timemixer':'models.TimeMixer'}[name]
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            subprocess.run([sys.executable,'-c',CHILD,str(payload),str(result),module],cwd=cwd,env=env,check=True,capture_output=True,text=True)
            reference=torch.load(result,weights_only=False)
            observed=reference['output'];diff=(expected-observed).abs().max().item()
            torch.testing.assert_close(expected,observed,atol=1e-6,rtol=1e-5)
            row=dict(model=name,configuration_overrides=extra,upstream_commit=revision,
                     strict_state_loading=True,shape=list(expected.shape),max_absolute_error=diff,
                     bitwise_equal=torch.equal(expected,observed),reference_optional_reformer_stub=reference['optional_import_stub'],passed=True)
            rows.append(row);print(json.dumps(row),flush=True)
    atomic_json(output,dict(scope='Native point backbones only; isolated pinned official implementations with identical weights/inputs. Optional unused Reformer import stub documented in script.',cases=rows,passed=True))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/baseline_refactor_validation/official_parity.json'))
    args=parser.parse_args();verify(args.upstream.resolve(),args.output)
