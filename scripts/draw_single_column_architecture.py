"""Vertical vector architecture designed at the actual ACM column width."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.35,6.5));ax.set(xlim=(0,10),ylim=(0,18));ax.axis('off')
    def box(y,title,body,color='#EAF2F9',edge='#34698C',height=1.35):
        ax.add_patch(FancyBboxPatch((.35,y),8.25,height,boxstyle='round,pad=.06,rounding_size=.12',fc=color,ec=edge,lw=.8))
        ax.text(4.475,y+height-.27,title,ha='center',va='center',fontsize=7.4,fontweight='bold',color=edge)
        ax.text(4.475,y+.43,body,ha='center',va='center',fontsize=6.7,linespacing=1.25)
    steps=[(16.1,'Observed history + RevIN',r'$[B,336,C]\ \rightarrow\ [B,336,C]$'+'\nObserved-window mean and scale'),
           (14.2,'Overlapping patch embedding',r'$P=16,\ S=8,\ N=41,\ D=64$'+'\nLinear projection + learned positions'),
           (12.3,'Shared residual MLP encoder',r'$[BC,41,64]$; two token-wise blocks'+'\nLayerNorm + FFN(64 → 128 → 64)'),
           (10.4,'Horizon-conditioned deformable decoder',r'$[BC,H,64]$; eight heads, eight samples'+'\nHistory context + future-step embeddings'),
           (8.5,'Median-anchored ordered quantile head',r'$g=\mathrm{softplus}(a)+10^{-4}$'+'\nMedian ± cumulative positive gaps'),
           (6.6,'Reverse RevIN + common correction',r'$\widehat Q=s\widetilde Q+\mu+s\gamma W_R\widetilde X$'+'\nSeven quantiles: [B, H, C, 7]')]
    for i,(y,title,body) in enumerate(steps):
        box(y,title,body,'#FFF2E2' if i==3 else ('#EBF5EE' if i>=4 else '#EAF2F9'))
        if i<len(steps)-1:
            ax.annotate('',xy=(4.475,steps[i+1][0]+1.43),xytext=(4.475,y-.06),arrowprops=dict(arrowstyle='-|>',color='#536674',lw=.9))
    ax.plot([8.65,9.2,9.2,8.65],[16.75,16.75,7.2,7.2],color='#79569A',lw=.9)
    ax.annotate('',xy=(8.65,7.2),xytext=(9.2,7.2),arrowprops=dict(arrowstyle='-|>',color='#79569A',lw=.9))
    ax.text(9.65,12,'History correction: dropout → Linear(L,H) → gate',rotation=90,ha='center',va='center',fontsize=6.5,color='#79569A')
    ax.text(4.475,5.95,'Decoder detail: one future step and one head',ha='center',fontsize=7.1,fontweight='bold')
    for y,title,body in [
        (4.35,'1. Construct the horizon query',r'$u_h=\mathrm{LN}(W_c\mathrm{mean}(Z)+e_h)$'),
        (2.7,'2. Move references and interpolate',r'$p=\mathrm{clip}(r+1.5\tanh(W_\delta u_h))$'+'\nGather and interpolate neighboring K / V'),
        (1.05,'3. Attend and refine',r'$\alpha=\mathrm{softmax}(q^\top K_p/\sqrt{d})$'+'\nWeighted values + residual + FFN')]:
        box(y,title,body,'#FFF2E2','#A56324',1.25)
    ax.text(4.7,.35,'Variables share weights and are processed independently.\nThe correction shifts all quantiles equally.',ha='center',va='center',fontsize=6.2,color='#536674')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    dest=Path('HADT_NQ_ICCRD2027_LaTeX/figures')
    for ext in ('pdf','svg','png'):fig.savefig(dest/f'revised_architecture.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':draw()
