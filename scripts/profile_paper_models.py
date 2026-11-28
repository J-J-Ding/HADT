"""CPU-only parameter/core-FLOP accounting for the actual ETTh1 configurations."""
from pathlib import Path
import sys
import json
import csv
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.model_registry import build_model
from utils.provenance import sha256,atomic_json


def main():
    torch.set_num_threads(3)
    torch.manual_seed(2024)
    rows=[]
    for h in (96,192,336,720):
        for name in ('hadt','dlinear','patchtst','timemixer','itransformer'):
            run=(Path('outputs/etth1_detail_search/confirm/L336/linear_skip') if name=='hadt'
                 else Path('outputs/paper_completion_20261126/baseline')/name)/f'H{h}'/'seed2024'
            resolved=json.loads((run/'resolved_config.json').read_text())
            cfg=resolved['config'];model=build_model(cfg,h,7).cpu().eval()
            parameters=sum(p.numel() for p in model.parameters())
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable==resolved.get('n_parameters',resolved.get('trainable_parameters'))
            x=torch.randn(1,336,7)
            with torch.no_grad(),FlopCounterMode(display=False) as counter:
                q=model(x)
            assert q.shape==(1,h,7,7)
            measured=int(counter.get_total_flops())
            # HADT implements score/aggregation using pointwise products + sums,
            # rather than mm/bmm, so the default profiler does not count them.
            # For BC,H,K,D: QK dot products and weighted V sums use ~4*BC*H*K*D;
            # two linear interpolations use ~6*BC*H*K*D. Count both explicitly.
            attention=interpolation=0
            if name=='hadt':
                d=model.d_model;k=model.decoder.n_points
                attention=4*7*h*k*d
                interpolation=6*7*h*k*d
            total=measured+attention+interpolation
            rows.append(dict(model=name,history=336,horizon=h,variables=7,batch_size=1,
                parameters=parameters,trainable_parameters=trainable,
                torch_matmul_convolution_flops=measured,
                added_sampled_attention_flops=attention,added_interpolation_flops=interpolation,
                estimated_core_flops=total,estimated_core_macs=total/2,
                gflops=total/1e9,config_sha256=sha256(run/'resolved_config.json'),
                counted_operators={str(k):int(v) for k,v in counter.get_flop_counts()['Global'].items()}))
            print(name,h,parameters,f'{total/1e9:.6f} GFLOPs',flush=True)
    out=Path('outputs/model_complexity');out.mkdir(parents=True,exist_ok=True)
    policy=dict(input='[1,336,7], ETTh1; H independently 96/192/336/720; output [1,H,7,7]',
        configs='Actual saved final ETTh1 model configurations; full seven-variable inference, no channel sampling.',
        metric='Estimated core forward FLOPs; multiply-add counted as 2 FLOPs.',
        measured='PyTorch FlopCounterMode matrix multiplication, convolution, and registered attention operators.',
        manual_hadt='Add 4*B*C*H*K*D for sampled attention dot/aggregation and 6*B*C*H*K*D for key/value interpolation.',
        excluded='Backward/optimizer, data loading, memory indexing/copy, normalization, nonlinear activation, softmax, and other miscellaneous scalar/elementwise operations. This is not a complete instruction count or speed measurement.',
        device='CPU for profiling only; ongoing GPU training unchanged.',torch_version=torch.__version__)
    atomic_json(out/'etth1_complexity.json',dict(policy=policy,results=rows))
    fields=[k for k in rows[0] if k!='counted_operators']
    with (out/'etth1_complexity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)


if __name__=='__main__':main()
