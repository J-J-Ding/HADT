"""Plot measured test fit against estimated core FLOPs; area denotes parameters."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.provenance import sha256,atomic_json


def main():
    root=Path('outputs/model_complexity')
    profile=root/'etth1_complexity.json'
    complexity={r['model']:r for r in json.loads(profile.read_text())['results'] if r['horizon']==96}
    labels={'hadt':'HDA-NQ (Ours)','dlinear':'DLinear-NQ','patchtst':'PatchTST-NQ','timemixer':'TimeMixer-NQ','itransformer':'iTransformer-NQ'}
    colors={'hadt':'#D65345','dlinear':'#75A944','patchtst':'#A67ABB','timemixer':'#E9B13D','itransformer':'#268DB1'}
    rows=[];sources={str(profile):sha256(profile)}
    for name in labels:
        values=[]
        for seed in (2024,2025,2026):
            if name=='hadt':
                p=Path(f'outputs/etth1_detail_search/confirm/L336/linear_skip/H96/seed{seed}/test_complete.json')
                r=json.loads(p.read_text());metrics=r['test'];hashes=r['artifact_sha256']
            else:
                p=Path(f'outputs/paper_completion_20261126/baseline/{name}/H96/seed{seed}/complete.json')
                r=json.loads(p.read_text());metrics=r['metrics'];hashes=r['artifacts']
            assert sha256(p.parent/'metrics.json')==hashes['metrics.json']
            sources[str(p)]=sha256(p)
            values.append(metrics['QuantileLoss_standardized'])
        c=complexity[name]
        rows.append(dict(model=name,label=labels[name],gflops=c['gflops'],parameters=c['trainable_parameters'],
            parameter_millions=c['trainable_parameters']/1e6,pinball_mean=float(np.mean(values)),pinball_std=float(np.std(values,ddof=1))))
    frame=pd.DataFrame(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.35,3.3))
    area_scale=600
    offsets={'hadt':(7,-8),'dlinear':(5,-17),'patchtst':(5,8),'timemixer':(-5,16),'itransformer':(6,6)}
    for r in rows:
        name=r['model'];x=r['gflops'];y=r['pinball_mean'];color=colors[name]
        ax.errorbar(x,y,yerr=r['pinball_std'],color=color,fmt='none',capsize=3,lw=1.2,zorder=2)
        ax.scatter(x,y,s=area_scale*r['parameter_millions'],c=color,alpha=.76,
                   edgecolors='#8F281E' if name=='hadt' else 'white',linewidths=1.5,zorder=3)
        dx,dy=offsets[name]
        ax.annotate(f"{labels[name]}\n{r['parameter_millions']:.3f}M",(x,y),xytext=(dx,dy),
                    textcoords='offset points',ha='right' if dx<0 else 'left',va='bottom',
                    fontsize=6,color=color,fontweight='bold' if name=='hadt' else 'normal',zorder=4)
    ax.set_xscale('log');ax.set_xlim(.00055,7);ax.set_ylim(.114,.236)
    ax.set_xlabel('Estimated core forward FLOPs (GFLOPs, log scale)')
    ax.set_ylabel('Test pinball loss (lower is better)')
    ax.set_title('ETTh1: L = 336, H = 96, 7 variables',fontsize=7,fontweight='bold')
    ax.grid(alpha=.17,which='major');ax.set_axisbelow(True)
    handles=[ax.scatter([],[],s=area_scale*v,facecolor='#CED2D6',edgecolor='white') for v in (.1,.4,.8)]
    ax.legend(handles,['0.1M','0.4M','0.8M'],title='Bubble area: parameters',loc='upper left',
              frameon=True,labelspacing=2.0,borderpad=1.0,handleheight=2.2,fontsize=5.5,title_fontsize=6,ncol=3,
              handletextpad=.4,columnspacing=1.1)
    ax.annotate('Preferred direction',xy=(.001,.119),xytext=(.004,.119),
                arrowprops=dict(arrowstyle='->',color='#697783'),fontsize=5.5,color='#697783',va='center')
    fig.subplots_adjust(left=.17,right=.97,bottom=.19,top=.89)
    for ext in ('pdf','svg','png'):fig.savefig(root/f'etth1_H96_efficiency_bubbles.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    frame.to_csv(root/'etth1_H96_efficiency_bubbles.csv',index=False)
    atomic_json(root/'etth1_H96_efficiency_bubbles_sources.json',dict(source_sha256=sources,
        bubble_area='Proportional to trainable parameter count, not memory.',
        error_bars='Sample standard deviation across seeds 2024/2025/2026.',
        scope='ETTh1 H96 only; no generalization to other horizons or datasets.'))
    print(frame.to_string(index=False))


if __name__=='__main__':main()
