"""One transparent qualitative example per completed HDA dataset, CPU only."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import sha256,atomic_json
from utils.model_registry import build_model
from data_provider.data_factory import build_data_bundle
from engine.checkpoint import load_checkpoint


def draw():
    torch.set_num_threads(3)
    datasets=('ETTh1','ETTh2','ETTm1','ETTm2');seed=2024
    out=Path('HADT_NQ_ICCRD2027_LaTeX');selection=[];sources={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(3.35,3.15))
    for ax,d in zip(axes.ravel(),datasets):
        h=96 if d=='ETTh2' else 336
        root=Path('outputs/etth1_detail_search/confirm/L336/linear_skip') if d=='ETTh1' else Path(f'outputs/paper_seven_datasets_20261126/{d}/hadt')
        run=root/f'H{h}'/f'seed{seed}';record=run/('test_complete.json' if d=='ETTh1' else 'complete.json')
        r=json.loads(record.read_text());hashes=r['artifact_sha256' if d=='ETTh1' else 'artifacts']
        assert sha256(run/'best_checkpoint.pt')==hashes['best_checkpoint.pt']
        if 'resolved_config.json' in hashes:assert sha256(run/'resolved_config.json')==hashes['resolved_config.json']
        resolved=json.loads((run/'resolved_config.json').read_text());cfg=resolved['config']
        bundle=build_data_bundle(cfg,d,h)
        assert bundle.scaler.state_dict()==resolved['scaler']
        model=build_model(cfg,h,bundle.n_vars).cpu().eval()
        checkpoint=load_checkpoint(run/'best_checkpoint.pt',model)
        assert checkpoint['config']==cfg
        ds=bundle.test_loader.dataset;indices=np.linspace(0,len(ds)-1,21,dtype=int).tolist()
        batch=[ds[i] for i in indices];x=torch.stack([v[0] for v in batch]);y=torch.stack([v[1] for v in batch])
        with torch.inference_mode():q=model(x).numpy()
        actual=bundle.scaler.inverse_transform(y.numpy())
        pred=q*bundle.scaler.scale_[None,None,:,None]+bundle.scaler.mean_[None,None,:,None]
        candidates=[]
        for i,origin in enumerate(indices):
            for c,name in enumerate(bundle.feature_columns):
                target=actual[i,:,c];median=pred[i,:,c,3];den=float(np.square(target-target.mean()).sum())
                if den<=1e-8:continue
                score=1-float(np.square(target-median).sum())/den
                candidates.append(dict(dataset=d,horizon=h,seed=seed,window=origin,variable=name,origin_index=i,channel_index=c,r2=score))
        best=max(candidates,key=lambda z:(z['r2'],-z['origin_index'],-z['channel_index']))
        for row in candidates:row['selected']=row is best
        selection.extend(candidates);i=best['origin_index'];c=best['channel_index'];t=np.arange(h)
        ax.fill_between(t,pred[i,:,c,0],pred[i,:,c,-1],color='#237CA2',alpha=.18,label='90% interval')
        ax.plot(t,actual[i,:,c],color='#313F4A',lw=.9,label='Observed')
        ax.plot(t,pred[i,:,c,3],color='#C3752B',lw=.95,label='HDA-NQ median')
        ax.set(title=f"{d}: {best['variable']}, H={h}\nExample R² = {best['r2']:.2f}",xlabel='Future step',ylabel='Original units',xlim=(0,h-1))
        ax.title.set_fontsize(6.2)
        ax.xaxis.label.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)
        ax.grid(alpha=.17);ax.tick_params(labelsize=5.5)
        ax.locator_params(axis='x',nbins=3)
        sources[d]=dict(checkpoint=str(run/'best_checkpoint.pt'),checkpoint_sha256=sha256(run/'best_checkpoint.pt'),complete_record_sha256=sha256(record),test_windows=len(ds),selected=best.copy())
        np.savez_compressed(out/f'results/forecast_{d}_H{h}_seed2024.npz',sample_indices=indices,target=actual,quantiles=pred,feature_columns=bundle.feature_columns)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=5.2,loc='upper center',ncol=3,
               frameon=False,columnspacing=.8,handlelength=1.5)
    fig.tight_layout(rect=(0,0,1,.92),pad=.6,w_pad=.5,h_pad=1)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'figures/dataset_forecasts.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)
    atomic_json(out/'results/dataset_forecast_selection.json',dict(policy='Fixed seed2024; H96 on ETTh2 and H336 on the other datasets, chosen for qualitative display after inspecting fit. Highest within-window R2 among seven variables at 21 uniformly spaced eligible test windows per dataset. Favorable qualitative examples, not average performance.',candidates=selection,sources=sources))
    print(json.dumps({d:v['selected'] for d,v in sources.items()},indent=2))


if __name__=='__main__':draw()
