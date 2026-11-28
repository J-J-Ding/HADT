"""Plot existing measured records only; no training or test-based model selection."""
import json
import hashlib
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'outputs/etth1_detail_search/confirm/L336'
OUT=ROOT/'HADT_NQ_ICCRD2027_LaTeX/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
sources={}


def record(path):
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def save(fig, name):
    fig.tight_layout()
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)


OUT.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,4,figsize=(14,3.5),sharex=True,sharey=True)
levels=np.array([.5,.8,.9])*100
for ax,h in zip(axes,(96,192,336,720)):
    ax.plot([45,95],[45,95],ls='--',color='#9AA4AC',lw=1,label='Nominal coverage')
    for candidate,label,color in [('reference','Previous MLP','#6D7B88'),('linear_skip','Unified linear_skip','#216F9C')]:
        values=[]
        for seed in (2024,2025,2026):
            p=record(RUNS/candidate/f'H{h}'/f'seed{seed}'/'metrics.json')
            r=json.loads(p.read_text());values.append([r[f'PICP{l}']*100 for l in (50,80,90)])
        values=np.asarray(values)
        ax.errorbar(levels,values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,
                    color=color,label=label,lw=1.5)
    ax.set(title=f'H = {h}',xlabel='Nominal interval coverage (%)',xticks=levels,xlim=(46,94),ylim=(42,95))
    ax.grid(alpha=.2)
axes[0].set_ylabel('Empirical test coverage (%)')
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('ETTh1, L = 336: full-test interval coverage; mean ± sample SD over 3 seeds',fontsize=12)
save(fig,'paper_latest_calibration')

path=record(RUNS/'linear_skip/H96/seed2024/figures/intermediate_steps.npz')
with np.load(path) as data:
    # Display one fixed origin/variable/head; it is a case study, not a population estimate.
    origin=0;channel=6;head=0
    positions=data['positions'][origin,channel,:,head,:]
    attention=data['attention'][origin,channel,:,head,:]
    offsets=data['offsets'][origin,channel,:,head,:]
    n=data['encoded'].shape[2];k=positions.shape[-1]
    sample_index=int(data['sample_indices'][origin])
    assert int(data['channel_indices'][channel])==6
fig,axes=plt.subplots(1,3,figsize=(14,4),gridspec_kw={'width_ratios':[1.15,1,1]})
ax=axes[0]
for r in np.linspace(0,n-1,k):ax.axvline(r,lw=.6,ls='--',color='#ADB6BD')
points=ax.scatter(positions.ravel(),np.repeat(np.arange(1,97),k),c=attention.ravel(),
                 s=5+45*attention.ravel(),cmap='viridis',vmin=0,vmax=1,alpha=.85)
ax.set(xlabel='Historical patch index (older → recent)',ylabel='Future step',title='Learned sampling locations',xlim=(-1,n),ylim=(96.5,.5))
fig.colorbar(points,ax=ax,label='Attention weight',fraction=.045,pad=.02)
for ax,arr,title,cmap,vmin,vmax in [
    (axes[1],offsets,'Predicted offsets (patch units)','coolwarm',-1.5,1.5),
    (axes[2],attention,'Attention over the 8 samples','viridis',0,1)]:
    m=ax.imshow(arr,aspect='auto',origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,
                extent=(.5,k+.5,96.5,.5))
    ax.set(xlabel='Reference sample index',ylabel='Future step',title=title,xticks=np.arange(1,k+1))
    fig.colorbar(m,ax=ax,fraction=.045,pad=.02)
fig.suptitle(f'Actual saved case: linear_skip · ETTh1 H96 · seed 2024 · test window {sample_index} · OT · head 0',fontsize=11)
save(fig,'paper_latest_sampling')

# Aggregate all saved cases, rather than generalizing from the displayed head.
diagnostics=[]
for h in (96,192,336,720):
    for seed in (2024,2025,2026):
        p=record(RUNS/'linear_skip'/f'H{h}'/f'seed{seed}'/'figures/intermediate_steps.npz')
        with np.load(p) as data:
            a=data['attention'];p=data['positions']
            entropy=-(a*np.log(np.clip(a,1e-30,1))).sum(-1)/np.log(a.shape[-1])
            for head in range(a.shape[-2]):
                diagnostics.append(dict(horizon=h,seed=seed,head=head,
                    normalized_entropy=float(entropy[...,head].mean()),
                    max_attention=float(a[...,head,:].max(-1).mean()),
                    last_reference_mass=float(a[...,head,-1].mean()),
                    boundary_fraction=float(((p[...,head,:]==0)|(p[...,head,:]==40)).mean())))
fig,axes=plt.subplots(1,3,figsize=(13,3.6))
for ax,key,title in zip(axes,('normalized_entropy','last_reference_mass','boundary_fraction'),
    ('Normalized attention entropy','Attention mass on last reference','Fraction of positions at a boundary')):
    matrix=np.array([[np.mean([r[key] for r in diagnostics if r['horizon']==h and r['head']==head])
                      for head in range(8)] for h in (96,192,336,720)])
    im=ax.imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=1)
    ax.set(title=title,xlabel='Head index',xticks=np.arange(8),yticks=np.arange(4),yticklabels=(96,192,336,720))
    ax.set_ylabel('Forecast length H')
    for i in range(4):
        for j in range(8):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                 color='white' if matrix[i,j]<.55 else '#182631')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
fig.suptitle('linear_skip: 3 fixed test windows × 7 variables × 3 seeds; saved-case diagnostics, not full-test averages',fontsize=10)
save(fig,'paper_latest_attention_diagnostics')
p=ROOT/'docs/validation/paper_attention_diagnostics.csv'
with p.open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)

manifest={'scope':'Measured existing records; no new training, no test-based configuration selection.',
          'calibration':'Complete test intervals at 50/80/90%, 2 configurations × 4 horizons × 3 seeds; SD across seeds.',
          'sampling':'One fixed saved test window, OT, head 0, seed 2024, H96; not causal attribution or full-test average.',
          'attention_diagnostics':'All saved first/middle/last windows, all 7 variables, 3 seeds, all 4 horizons; not full-test averages.',
          'source_sha256':sources}
p=ROOT/'docs/validation/paper_evidence_figures.json'
p.write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Wrote calibration and sampling figures; audited {len(sources)} input files.')
