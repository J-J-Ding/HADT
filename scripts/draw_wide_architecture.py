"""Horizontal vector architecture at full ACM text width."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
F=Path('HADT_NQ_ICCRD2027_LaTeX/figures')
def save(fig,name):
    fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=220)
    plt.close(fig)
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
