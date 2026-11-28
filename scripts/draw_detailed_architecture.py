"""Vector schematic of the validation-selected linear_skip model; no measured data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'HADT_NQ_ICCRD2027_LaTeX' / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
COL = {'blue': ('#EDF5FC', '#28628C'), 'orange': ('#FFF3E4', '#A45915'),
       'green': ('#EDF7F0', '#347657'), 'purple': ('#F3EEFA', '#76529B'),
       'gray': ('#F5F7F9', '#546575')}
fig, ax = plt.subplots(figsize=(20, 13.5))
ax.set(xlim=(0, 20), ylim=(0, 13.5)); ax.axis('off')


def text(x, y, s, size=10, color='#22323F', weight='normal', ha='center', **kw):
    return ax.text(x, y, s.replace(chr(92)+'n', chr(10)), fontsize=size, color=color, fontweight=weight,
                   ha=ha, va='center', linespacing=1.45, **kw)


def box(x, y, w, h, title, body='', style='blue', size=10):
    fill, edge = COL[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.025,rounding_size=0.09',
                 facecolor=fill, edgecolor=edge, linewidth=1.25))
    text(x+w/2, y+h-.26, title, 11, edge, 'bold')
    if body:
        text(x+w/2, y+(h-.43)/2, body, size)


def arrow(points, color='#536778', dashed=False, lw=1.3):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                ls='--' if dashed else '-')
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle='--' if dashed else '-', mutation_scale=12))


def panel(y, label, title, subtitle, x=0):
    text(x+.35, y, label, 14, '#243C50', 'bold', ha='left')
    text(x+1.05, y, title, 13, '#243C50', 'bold', ha='left')
    if subtitle:
        text(19.6, y, subtitle, 9, '#677785', ha='right')


text(10, 13.20, 'HADT-NQ: horizon-conditioned historical retrieval with ordered quantile outputs',
     17, weight='bold')
text(10, 12.85, 'Unified validation-selected configuration: linear_skip  |  L = 336, D = 64, two token-wise MLP blocks',
     10, '#536778')
panel(12.40, '(a)', 'End-to-end data flow', 'Solid arrows: tensors   ·   Dashed arrows: normalization statistics')

box(.35, 10.35, 2.15, 1.65, 'History + RevIN',
    r'$X: [B,336,C]$\nTrain-only global scaling\nWindow statistics $\mu,s$\n$\widetilde X: [B,336,C]$', size=9)
box(2.95, 10.35, 2.15, 1.65, 'Patch embedding',
    r'Unfold: $P=16$, $S=8$\n$[BC,41,16]$\nLinear $16\!\rightarrow\!64$ + position\n$T: [BC,41,64]$', size=9)
box(5.55, 10.35, 2.15, 1.65, 'Shared encoder',
    r'Token-wise residual MLP × 2\n$64\!\rightarrow\!128\!\rightarrow\!64$\nFinal LayerNorm\n$Z: [BC,41,64]$', size=9)
box(8.15, 10.35, 2.65, 1.65, 'Deformable decoder',
    r'History context + horizon ID\n8 heads × 8 sampled positions\nBounded offsets + interpolation\n$[BC,H,64]$  ·  see (b)', 'orange', 9)
box(11.25, 10.35, 2.15, 1.65, 'Ordered head',
    r'Median + 6 positive gaps\nCumulative construction\n$[BC,H,7]$\nReshape $[B,H,C,7]$', 'green', 9)
box(13.85, 10.35, 1.85, 1.65, 'Reverse RevIN',
    r'$Q^{base}=s\widetilde Q+\mu$\nHistory-only statistics\nGlobal-standardized\ncoordinates', 'gray', 9)
ax.add_patch(Circle((16.75, 11.17), .22, facecolor='white', edgecolor='#76529B', lw=1.6))
text(16.75, 11.17, '+', 18, '#76529B')
box(17.45, 10.35, 2.15, 1.65, 'Forecasts',
    r'$\widehat Q: [B,H,C,7]$\n$H\in\{96,192,336,720\}$\n50% / 80% / 90% intervals\nOne model trained per $H$', 'green', 9)
for a, b in [(2.5, 2.95), (5.1, 5.55), (7.7, 8.15), (10.8, 11.25), (13.4, 13.85), (15.7, 16.53), (16.97, 17.45)]:
    arrow([(a, 11.17), (b, 11.17)])
# The correction joins AFTER reversal, matching DetailHADTNQ.forward exactly.
box(5.55, 9.20, 8.55, .70, 'Historical correction: Dropout → Linear(336,H) → gate → × s',
    '', 'purple')
text(9.825, 9.39, 'Same learned shift for all seven quantiles; see (d).', 9, '#76529B')
arrow([(1.42, 10.35), (1.42, 9.55), (5.55, 9.55)], '#76529B')
arrow([(14.10, 9.55), (16.75, 9.55), (16.75, 10.95)], '#76529B')
arrow([(1.0, 12.0), (1.0, 12.15), (14.75, 12.15), (14.75, 12.0)], '#7B8992', True, .9)
text(10.65, 12.24, r'Saved $(\mu,s)$', 8, '#677785')

panel(8.73, '(b)', 'Horizon-conditioned deformable attention', 'One variable shown; weights are shared across variables')
box(.35, 5.48, 3.15, 2.75, '1  Construct query',
    r'$c=\mathrm{mean}_{n}(Z)$\n$u_h=\mathrm{LN}(W_c c+e_h)$\n$q_{hm}=(W_Q u_h)_m$\nLearned horizon embedding $e_h$\n$u: [BC,H,64]$', 'orange', 10)
box(3.95, 5.48, 3.15, 2.75, '2  Locate samples',
    r'$r_k=(k-1)(N-1)/(K-1)$\n$\delta_{hmk}=1.5\tanh(W_\delta u_h)_{mk}$\n$p_{hmk}=\mathrm{clip}(r_k+\delta,0,N-1)$\n8 positions per head\nOffsets start at zero', 'orange', 9.5)
box(7.55, 5.48, 3.75, 2.75, '3  Interpolate historical K / V',
    r'$K=W_KZ,\quad V=W_VZ$\n$\ell=\lfloor p\rfloor,\ r=\min(\ell+1,N-1)$\n$\lambda=p-\ell$\n$\mathcal{I}(F,p)=(1-\lambda)F_\ell+\lambda F_r$\nSampled K / V: $[BC,H,8,8,8]$', 'orange', 9.5)
box(11.75, 5.48, 3.55, 2.75, '4  Weight and aggregate',
    r'$\alpha_{hmk}=\mathrm{softmax}_{k}$\n$\quad(q_{hm}^{\mathsf{T}}\mathcal{I}(K_m,p)/\sqrt{8})$\n$o_{hm}=\sum_k\alpha_{hmk}\mathcal{I}(V_m,p)$\nSoftmax over 8 samples\nConcatenate 8 heads + $W_O$', 'orange', 9.5)
box(15.75, 5.48, 3.85, 2.75, '5  Residual refinement',
    r'$v_h=\mathrm{LN}_1(u_h+\mathrm{Drop}(W_O o_h))$\n$z_h=v_h+\mathrm{Drop}(\mathrm{FFN}(\mathrm{LN}_2(v_h)))$\nFFN: $64\!\rightarrow\!128\!\rightarrow\!64$\nDirect predictions for all $H$ steps\n$z: [BC,H,64]$ → ordered head', 'orange', 9.3)
for a, b in [(3.5, 3.95), (7.1, 7.55), (11.3, 11.75), (15.3, 15.75)]:
    arrow([(a, 6.8), (b, 6.8)], '#A45915')
# Schematic glyphs make the operations visible as well as algebraic.
for i in range(8):
    ax.add_patch(Rectangle((.95+i*.24, 7.50), .18, .22,
                 facecolor=plt.cm.Blues(.3+i*.07), edgecolor='white', lw=.5))
for i in range(6):
    x=4.40+i*.42
    ax.add_patch(Circle((x, 7.68), .035, facecolor='white', edgecolor='#A45915', lw=.8))
    ax.add_patch(Circle((x+(.08 if i%2 else -.06), 7.48), .04, facecolor='#C4782A', lw=0))
    ax.plot([x, x+(.08 if i%2 else -.06)], [7.68, 7.48], color='#C4782A', lw=.7)
ax.plot([8.40, 10.40], [7.52, 7.52], color='#7B8992')
ax.scatter([8.40, 10.40], [7.52, 7.52], s=26, color='#28628C')
ax.scatter([9.12], [7.52], s=38, color='#C4782A', zorder=5)
for i, height in enumerate([.08, .16, .27, .10, .20, .07, .15, .09]):
    ax.add_patch(Rectangle((12.25+i*.30, 7.43), .18, height,
                          facecolor='#C4782A', edgecolor='none'))
text(17.65, 7.60, 'query → ⊕ → LN → ⊕ → state', 9, '#A45915')
text(9.0, 8.40, '$Z$ supplies both pooled context and historical keys / values; query projection also feeds attention scores.',
     9, '#536778')
# Simple sampling illustration: explicitly schematic, not an experimental result.
ax.plot([1.0, 18.9], [4.91, 4.91], color='#ABB6BF', lw=1)
ref = np.linspace(1.0, 18.9, 8)
pos = np.clip(ref + np.array([.2, -.3, .35, -.15, .4, -.2, .22, -.12]), 1, 18.9)
ax.scatter(ref, np.full(8, 4.91), s=28, facecolors='white', edgecolors='#7A8791', zorder=4)
ax.scatter(pos, np.full(8, 4.66), s=36, color='#C4782A', zorder=4)
for r, p in zip(ref, pos):
    ax.plot([r, p], [4.91, 4.66], color='#C4782A', lw=.9)
text(.4, 4.34, 'Illustration: one head and one horizon; open circles = uniform references, filled circles = shifted samples.',
     9, '#677785', ha='left')
text(19.6, 4.34, 'Older history → more recent history', 9, '#677785', ha='right')

panel(3.86, '(c)', 'Ordered quantiles', '')
panel(3.86, '(d)', 'Shared historical correction', '', x=10.3)

box(.35, .95, 9.25, 2.50, 'Median-anchored positive-gap head', '', 'green')
text(2.25, 2.63, r'$z_h\rightarrow\mathrm{Linear}(64,7)$\n$m_h,\ a^-_{h,1:3},\ a^+_{h,1:3}$\n$g^\pm_{hj}=\mathrm{softplus}(a^\pm_{hj})+10^{-4}$', 10)
arrow([(4.12, 2.34), (4.65, 2.34)], '#347657')
text(7.05, 2.52, r'$q_{.50}=m_h$\n$q_{.25},q_{.10},q_{.05}=m_h-\mathrm{cumsum}(g^-)$\n$q_{.75},q_{.90},q_{.95}=m_h+\mathrm{cumsum}(g^+)$', 10)
text(5.0, 1.27, 'Reorder lower outputs to ascending levels: .05, .10, .25, .50, .75, .90, .95', 9, '#347657')
quantile_x=[2.1, 3.2, 4.3, 5.0, 5.8, 6.9, 8.1]
ax.plot(quantile_x, np.full(7, 1.68), color='#98B9A4', lw=1.2)
ax.scatter(quantile_x, np.full(7, 1.68), s=[20,25,30,50,30,25,20],
           c=['#84B39A']*3+['#246644']+['#84B39A']*3, zorder=4)

box(10.65, .95, 8.95, 2.50, 'A common correction preserves every quantile gap', '', 'purple')
text(15.125, 2.57, r'$R=\mathrm{Linear}_{L\rightarrow H}(\mathrm{Drop}(\widetilde X))$\n$\gamma=0.5\tanh(g),\qquad\gamma_{init}=0.1$\n$\widehat Q_{bhcq}=Q^{base}_{bhcq}+s_{bc}\gamma R_{bhc}$', 11)
text(15.125, 1.35, r'Zero-initialized projection; no bias. Gate is bounded; correction amplitude is not.\nEquivalent to adding the correction before inverse RevIN.', 9, '#76529B')

text(10, .57, 'Training: mean pinball loss across seven quantiles; validation early stopping. No difference-loss term in this configuration.', 10, '#354B5C')
text(10, .22, 'Notation: BC = flattened batch × variables; N = 41 patches; M = 8 heads; K = 8 samples; head width = 8. Output is globally standardized until inverse data scaling.', 9, '#677785')

OUT.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(left=.008, right=.992, bottom=.008, top=.992)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(OUT / f'paper_architecture_detailed.{ext}', dpi=200,
                bbox_inches='tight', pad_inches=.08)
for name, bottom, top in [('overview', 9.05, 12.65), ('decoder', 4.18, 8.97), ('output', .85, 4.04)]:
    crop=Bbox.from_extents(.15, bottom, 19.85, top).transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT / f'paper_architecture_{name}.pdf', bbox_inches=crop, pad_inches=.06)
plt.close(fig)
print(OUT / 'paper_architecture_detailed.pdf')
