"""Draw a schematic of the implemented HADT-NQ computation (no data)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                           "font.family": "DejaVu Sans", "font.size": 8})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HADT_NQ_ICCRD2027_LaTeX" / "figures"

fig, ax = plt.subplots(figsize=(7.15, 2.45))
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 10), ylim=(0, 3.1))
ax.axis("off")
colors = {"blue": ("#e8f0f7", "#315b7b"),
          "orange": ("#fff0df", "#9b5a15"),
          "green": ("#e9f3ea", "#386743")}

def box(x, y, w, h, title, body, style="blue", title_size=7.8):
    fill, edge = colors[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.035,rounding_size=0.055",
                 facecolor=fill, edgecolor=edge, linewidth=0.85))
    ax.text(x+w/2, y+h-0.21, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=edge)
    ax.text(x+w/2, y+(h-0.28)/2, body, ha="center", va="center",
            fontsize=7.0, linespacing=1.35, color="#18242b")

def arrow(a,b,color="#52606b",ls="-"):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="-|>",
                lw=1.0,color=color,linestyle=ls,mutation_scale=8))

box(.08, 1.70, 1.62, 1.05, "History + RevIN", "Train-only scaling\nNon-affine per window\n$[B,336,C]$")
box(2.03, 1.70, 1.55, 1.05, "Patch embedding", "$P=16$, stride $8$\n$N=41$ tokens\nLearned positions")
box(3.91, 1.70, 1.88, 1.05, "Shared encoder", "Channel-independent\n3 temporal blocks\n$D=128$, $M=8$")
box(6.12, 1.70, 1.77, 1.05, "Horizon decoder", "$K=8$ samples / head\nOffset + attention\nOne query per horizon", "orange")
box(8.22, 1.70, 1.70, 1.05, "Ordered output", "Median + positive gaps\nReverse instance scale\n$[B,H,C,7]$", "green")
for x0, x1 in [(1.70,2.03),(3.58,3.91),(5.79,6.12),(7.89,8.22)]:
    arrow((x0,2.22),(x1,2.22))

ax.text(5, 3.01, "HADT-NQ: direct marginal quantile forecasting",
        fontsize=10, fontweight="bold", ha="center", color="#18242b")

box(.10, .15, 2.27, .92, "Query construction", "$u_h=\mathrm{LN}(W_c\,\mathrm{mean}(Z)+e_h)$\nContext + horizon identity", "orange", 8)
box(2.69, .15, 2.27, .92, "Bounded positions", "$p=\mathrm{clip}(r+1.5\,\tanh(W_\delta u_h))$\nUniform reference grid $r$", "orange", 8)
box(5.28, .15, 2.27, .92, "Sample + attend", "Interpolate keys / values\nSoftmax over $K$ samples / head", "orange", 8)
box(7.87, .15, 2.03, .92, "Decoder state", "Head projection + residual\nLayer norm + FFN residual", "orange", 8)
for x0,x1 in [(2.37,2.69),(4.96,5.28),(7.55,7.87)]:
    arrow((x0,.61),(x1,.61),"#9b5a15")
ax.text(.12, 1.32, "Decoder detail (schematic)", fontsize=7.7,
        fontweight="bold", color="#9b5a15")
arrow((7.0,1.70),(7.0,1.10),"#9b5a15", "--")

fig.subplots_adjust(left=.008,right=.992,bottom=.015,top=.985)
OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/"paper_architecture.pdf", bbox_inches="tight", pad_inches=.035)
fig.savefig(OUT/"paper_architecture.png", dpi=240, bbox_inches="tight", pad_inches=.035)
plt.close(fig)
