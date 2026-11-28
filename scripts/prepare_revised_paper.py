"""Generate publication tables and figures from complete existing three-seed groups."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'HADT_NQ_ICCRD2027_LaTeX'; R=P/'results'; F=P/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
sources={}
def read(path):
    path=Path(path);sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path
def write(name,s): (R/name).write_text(s+'\n')
def table(name,caption,label,header,rows,alignment,wide=False):
    env='table*' if wide else 'table'
    write(name,'\n'.join([rf'\begin{{{env}}}[t]',r'\centering\small',
       r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+alignment+'@{}}',
       r'\toprule',' & '.join(header)+r' \\',r'\midrule',*rows,
       r'\bottomrule',r'\end{tabular}',rf'\end{{{env}}}']))
def save(fig,name):
    fig.tight_layout()
    for ext in ('pdf','png','svg'):fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=210)
    plt.close(fig)
def pm(row,key,digits=5,scale=1):
    return '$'+f'{row[key+"_mean"]*scale:.{digits}f}'+r'\pm'+f'{row[key+"_std"]*scale:.{digits}f}'+'$'

fig,axes=plt.subplots(2,2,figsize=(3.35,3.0),sharex=True,sharey=True)
for ax,h in zip(axes.ravel(),(96,192,336,720)):
    ax.plot([45,95],[45,95],ls='--',color='#9BA6AF',lw=.9)
    for c,label,col in [('linear_skip','HDA-NQ','#237CA2')]:
        values=[]
        for seed in (2024,2025,2026):
            p=read(ROOT/f'outputs/etth1_detail_search/confirm/L336/{c}/H{h}/seed{seed}/metrics.json')
            m=json.loads(p.read_text());values.append([m[f'PICP{v}']*100 for v in (50,80,90)])
        v=np.asarray(values)
        ax.errorbar([50,80,90],v.mean(0),yerr=v.std(0,ddof=1),capsize=2,marker='o',ms=3,label=label,color=col,lw=1)
    ax.set(title=f'H = {h}',xticks=(50,80,90),xlim=(46,94),ylim=(42,95));ax.grid(alpha=.18)
axes[0,0].legend(fontsize=5.5)
for ax in axes[:,0]:ax.set_ylabel('Empirical coverage (%)')
for ax in axes[-1,:]:ax.set_xlabel('Nominal coverage (%)')
for ax in axes.ravel():
    ax.tick_params(labelsize=6)
    ax.title.set_fontsize(7)
    ax.xaxis.label.set_fontsize(6.5)
    ax.yaxis.label.set_fontsize(6.5)
save(fig,'revised_calibration')
# High-fit qualitative examples, selected transparently at a fixed seed.
fig,axes=plt.subplots(2,2,figsize=(7.15,4.3))
selection_rows=[]
for ax,h in zip(axes.ravel(),(96,192,336,720)):
    p=read(ROOT/f'outputs/etth1_detail_search/confirm/L336/linear_skip/H{h}/seed2024/figures/forecast_samples.npz')
    with np.load(p) as f:
        candidates=[]
        for origin in range(f['target'].shape[0]):
            for channel,name in enumerate(f['feature_columns']):
                y=f['target'][origin,:,channel];median=f['quantiles'][origin,:,channel,3]
                denominator=float(np.square(y-y.mean()).sum())
                if denominator<=1e-8:continue
                score=1-float(np.square(y-median).sum())/denominator
                candidates.append(dict(horizon=h,seed=2024,window=int(f['sample_indices'][origin]),
                    variable=str(name),origin_index=origin,channel_index=channel,r2=score,
                    source=str(p.relative_to(ROOT))))
        chosen=max(candidates,key=lambda r:(r['r2'],-r['origin_index'],-r['channel_index']))
        for r in candidates:r['selected']=r is chosen
        selection_rows.extend(candidates)
        i,c=chosen['origin_index'],chosen['channel_index']
        target=f['target'][i,:,c];q=f['quantiles'][i,:,c];x=np.arange(1,len(target)+1)
        assert len(target)==h and q.shape==(h,7) and np.isfinite(q).all()
        ax.fill_between(x,q[:,0],q[:,-1],color='#247CA3',alpha=.14,label='90% interval')
        ax.plot(x,target,color='#2A3037',lw=.8,label='Observed')
        ax.plot(x,q[:,3],color='#CA6B29',lw=.9,label='Median')
        ax.set(title=f"{chosen['variable']}, H = {h} (example R² = {chosen['r2']:.2f})",
               xlabel='Future step',ylabel='Original units',xlim=(1,h))
        ax.grid(alpha=.15)
axes[0,0].legend(fontsize=6,ncol=3,loc='upper right')
save(fig,'revised_forecasts')
pd.DataFrame(selection_rows).to_csv(R/'forecast_example_selection.csv',index=False)
write('forecast_example_selection.json',json.dumps(dict(
    policy='Qualitative display only: fixed model linear_skip and seed2024; highest within-window R2 over saved first/middle/last windows and all seven variables, independently for each H. Full named horizons are plotted without smoothing.',
    selected=[r for r in selection_rows if r['selected']],
    test_used_for_model_selection=False,not_representative_population_estimate=True),indent=2))

# Compact publication diagnostic: actual historical mass and multi-case head entropy.
example=next(r for r in selection_rows if r['selected'] and r['horizon']==96)
path=read(ROOT/'outputs/etth1_detail_search/confirm/L336/linear_skip/H96/seed2024/figures/intermediate_steps.npz')
with np.load(path) as data:
    pos=data['positions'][example['origin_index'],example['channel_index']]
    att=data['attention'][example['origin_index'],example['channel_index']]
    n=data['encoded'].shape[2];mass=np.zeros((96,n))
    for head in range(pos.shape[1]):
        for sample in range(pos.shape[2]):
            location=pos[:,head,sample];left=np.floor(location).astype(int)
            right=np.minimum(left+1,n-1);weight=location-left
            np.add.at(mass,(np.arange(96),left),att[:,head,sample]*(1-weight)/pos.shape[1])
            np.add.at(mass,(np.arange(96),right),att[:,head,sample]*weight/pos.shape[1])
    assert np.allclose(mass.sum(1),1,atol=1e-6)
diagnostics=pd.read_csv(read(ROOT/'docs/validation/paper_attention_diagnostics.csv'))
entropy=diagnostics.groupby(['horizon','head']).normalized_entropy.mean().unstack().loc[[96,192,336,720]].to_numpy()
fig,axes=plt.subplots(2,1,figsize=(3.35,4.25),gridspec_kw={'height_ratios':[1.15,1]})
image=axes[0].imshow(mass,aspect='auto',origin='upper',cmap='viridis',extent=(-.5,n-.5,96.5,.5))
axes[0].set(title='(a) Historical attention mass (H = 96)',xlabel='Historical patch (older → recent)',ylabel='Future step',yticks=[1,24,48,72,96])
fig.colorbar(image,ax=axes[0],fraction=.04,pad=.02).set_label('Mass',fontsize=7)
image=axes[1].imshow(entropy,aspect='auto',cmap='viridis',vmin=0,vmax=1)
axes[1].set(title='(b) Normalized head entropy',xlabel='Head index',ylabel='Prediction length H',xticks=np.arange(8),yticks=np.arange(4),yticklabels=[96,192,336,720])
for i in range(4):
    for j in range(8):axes[1].text(j,i,f'{entropy[i,j]:.2f}',ha='center',va='center',fontsize=6.5,color='#21313D')
fig.colorbar(image,ax=axes[1],fraction=.04,pad=.02)
for ax in axes:
    ax.tick_params(labelsize=6.5)
    ax.title.set_fontsize(7.5)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontsize(7)
for cax in fig.axes:cax.tick_params(labelsize=6.5)
save(fig,'revised_attention')
np.savez_compressed(R/'revised_attention_data.npz',historical_mass=mass,entropy=entropy)

fig,ax=plt.subplots(figsize=(7.15,4.0));ax.set(xlim=(0,14.3),ylim=(0,8));ax.axis('off')
pal={'blue':('#EDF5FB','#34698C'),'orange':('#FFF2E2','#A56324'),
     'green':('#EBF5EE','#3D7754'),'purple':('#F3EDFA','#79569A')}
def tx(x,y,s,sz=7,**kw):ax.text(x,y,s,ha='center',va='center',fontsize=sz,linespacing=1.25,**kw)
def box(x,y,w,h,title,body='',style='blue',sz=6.5):
    bg,fg=pal[style];ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.08',fc=bg,ec=fg,lw=.8))
    if title:tx(x+w/2,y+h-.23,title,7,fontweight='bold',color=fg)
    if body:tx(x+w/2,y+(h-.37)/2,body,sz)
def ar(a,b,col='#536674',ls='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=.8,color=col,linestyle=ls,mutation_scale=8))
tx(7.15,7.75,'(a) Channel-independent forecasting path',9,fontweight='bold')
boxes=[(.15,1.7,'History + RevIN',r'$[B,336,C]$'+'\nWindow statistics'),
       (2.30,1.8,'Patch tokens',r'$P16/S8,\ N=41$'+'\nLinear + positions'),
       (4.55,1.9,'Residual MLP',r'$D64,\ E=2$'+'\nToken-wise FFNs'),
       (6.90,2.10,'Horizon decoder',r'$M=8,\ K=8$'+'\nHorizon-conditioned'),
       (9.45,1.85,'Ordered head','Median + 6 gaps\nSeven quantiles'),
       (11.75,2.35,'Reverse + output',r'$[B,H,C,7]$'+'\nRestore history scale')]
for i,(x,w,t,bdy) in enumerate(boxes):
    box(x,5.87,w,1.45,t,bdy,'orange' if i==3 else ('green' if i>=4 else 'blue'))
for (x,w,*_), (x2,*_) in zip(boxes[:-1],boxes[1:]):ar((x+w,6.59),(x2,6.59))
box(3.9,4.82,6.6,.63,'Common history correction','Dropout → Linear(L,H) → gate × window scale','purple')
ar((1.0,5.87),(1.0,5.12),'#79569A','-');ar((1.,5.12),(3.9,5.12),'#79569A','-')
ar((10.5,5.12),(12.9,5.12),'#79569A','-');ar((12.9,5.12),(12.9,5.87),'#79569A','-')
tx(7.15,4.49,'All variables share weights; each future step has a separate query. The correction shifts all quantiles equally.',6.5,color='#576775')
tx(7.15,4.10,'(b) Decoder detail: one future step, one head',9,fontweight='bold')
box(.15,1.92,3.0,1.85,'Build horizon query',r'$u_h=\mathrm{LN}(W_c\mathrm{mean}(Z)+e_h)$'+'\nHistory context + lead identity','orange',6.2)
box(3.65,1.92,3.0,1.85,'Predict bounded positions',r'$p=\mathrm{clip}(r+1.5\tanh(W_\delta u_h))$'+'\nOffsets move reference points','orange',6)
box(7.15,1.92,3.0,1.85,'Interpolate keys / values',r'$\mathcal{I}(F,p)=(1-\lambda)F_\ell+\lambda F_r$'+'\nGather neighboring patches','orange',6.3)
box(10.65,1.92,3.45,1.85,'Attend + refine',r'$\alpha=\mathrm{softmax}(q^\top K_p/\sqrt{d})$'+'\nWeighted values → residual + FFN','orange',6.5)
for aa,bb in [(3.15,3.65),(6.65,7.15),(10.15,10.65)]:ar((aa,2.76),(bb,2.76),'#A56324')
refs=np.linspace(4.1,6.15,8)
for j,r in enumerate(refs):
    dest=r+(.10 if j%2 else -.06);ax.plot([r,dest],[3.2,3.0],color='#A56324',lw=.6)
    ax.scatter([r],[3.2],s=10,facecolors='white',edgecolors='#A56324',lw=.6)
    ax.scatter([dest],[3.0],s=12,color='#A56324')
for j in range(8):
    ax.add_patch(Rectangle((.65+j*.24,3.04),.16,.15,fc=plt.cm.Blues(.3+.07*j),ec='white',lw=.3))
ax.plot([7.8,9.5],[3.1,3.1],color='#60798B',lw=.7)
ax.scatter([7.8,9.5],[3.1,3.1],s=15,color='#34698C');ax.scatter([8.5],[3.1],s=20,color='#A56324',zorder=4)
tx(7.15,1.55,'(c) Ordered outputs: median anchor and cumulative positive gaps',8,fontweight='bold')
box(.15,.13,13.95,1.12,'',
    r'$g^\pm_j=\mathrm{softplus}(a^\pm_j)+10^{-4},\quad q_{.50}=m,\quad q_{\mathrm{lower}}=m-\mathrm{cumsum}(g^-),\quad q_{\mathrm{upper}}=m+\mathrm{cumsum}(g^+)$'+'\n'+
    'Ascending output levels: .05, .10, .25, .50, .75, .90, .95. Common history shifts preserve all gaps.','green',6.6)
save(fig,'revised_architecture')
write('revision_sources.json',json.dumps({'date':'2026-11-26','scope':'Only complete three-seed groups; no new training.',
       'figure_scope':'Final linear_skip model only','source_sha256':sources},indent=2))
print('Publication tables and four figures generated; input hashes recorded.')

import runpy
runpy.run_path('scripts/draw_wide_architecture.py',run_name='__main__')
from scripts.draw_dataset_forecasts import draw as draw_dataset_forecasts
draw_dataset_forecasts()
