"""Clean hand-laid block schematic of the Hopsan plant (Fig. 1).

Replaces the auto-placed (overlapping) .hmf-pose rendering with a deliberate,
non-overlapping servo-loop diagram, colour-coded by physical domain.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mp

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

HYD = ("#cfe8ff", "#1f6fb2")   # hydraulic
MEC = ("#d7f0d7", "#2e7d32")   # mechanical
SIG = ("#ffe6c7", "#c77a1f")   # signal/control

# node: (label, x, y, domain, w, h)
BW, BH = 2.0, 0.95
nodes = {
    "ref":   ("Reference\n(sine)",        0.9, 4.0, SIG),
    "sum":   ("$\\Sigma$\n(error)",        3.0, 4.0, SIG),
    "pi":    ("PI\ncontroller",            5.1, 4.0, SIG),
    "valve": ("4/3 servo\nvalve",          7.3, 4.0, HYD),
    "hose":  ("Hose\n(line loss)",         9.6, 4.0, HYD),
    "cyl":   ("Double-acting\ncylinder",  11.9, 4.0, HYD),
    "load":  ("Load\n(mass + stops)",     14.2, 4.0, MEC),
    "sup":   ("Pump + relief\n(supply)",   5.1, 6.4, HYD),
    "sens":  ("Position\nsensor",          7.3, 1.7, MEC),
    "pq":    ("$P,Q$ sensors\n(manifold)", 8.45, 6.4, SIG),
}

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(-0.4, 15.6); ax.set_ylim(0.6, 7.7); ax.axis("off")


def draw_box(key):
    label, x, y, (fc, ec) = nodes[key]
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=fc, edgecolor=ec, lw=1.8, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, zorder=4, color="#1a1a1a")


def edge(a, b, color="#444", style="-", lw=1.8, rad=0.0, label=None, loff=(0, 0.28)):
    (_, ax1, ay1, _), (_, ax2, ay2, _) = nodes[a], nodes[b]
    # connect from box edge, not centre
    dx, dy = ax2 - ax1, ay2 - ay1
    sx = ax1 + (BW / 2) * (1 if dx > 0.3 else (-1 if dx < -0.3 else 0))
    sy = ay1 + (BH / 2) * (1 if dy > 0.3 else (-1 if dy < -0.3 else 0))
    ex = ax2 - (BW / 2) * (1 if dx > 0.3 else (-1 if dx < -0.3 else 0))
    ey = ay2 - (BH / 2) * (1 if dy > 0.3 else (-1 if dy < -0.3 else 0))
    ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=lw, linestyle=style,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((sx + ex) / 2 + loff[0], (sy + ey) / 2 + loff[1], label,
                ha="center", va="center", fontsize=8, color=color, zorder=5)


for k in nodes:
    draw_box(k)

# forward path
edge("ref", "sum", label="$x_{ref}$")
edge("sum", "pi", label="error")
edge("pi", "valve", label="spool")
edge("valve", "hose", color=HYD[1], lw=2.4, label="$Q_1,Q_2$")
edge("hose", "cyl", color=HYD[1], lw=2.4, label="")
edge("cyl", "load", color=MEC[1], lw=2.2, label="motion", loff=(0, 0.32))
# supply down into valve
edge("sup", "valve", color=HYD[1], lw=2.2, label="supply", loff=(0.55, 0.0))
# P,Q sensors tap the manifold (valve outlet, upstream of the hose)
(_, vx, vy, _) = nodes["valve"]; (_, hx, hy, _) = nodes["hose"]
manx = (vx + BW / 2 + hx - BW / 2) / 2          # midpoint of the valve->hose link
ax.plot([manx], [4.0], marker="o", ms=6, color=SIG[1], zorder=6)
ax.add_patch(FancyArrowPatch((manx, 4.0 + 0.08), (manx, 6.4 - BH / 2),
             arrowstyle="-|>", mutation_scale=12, color=SIG[1], lw=1.6,
             linestyle=":", connectionstyle="arc3,rad=0.0", zorder=2))
# feedback: load -> sensor -> sum
edge("load", "sens", color=MEC[1], style="--", rad=-0.0, label="position", loff=(0.7, 0.0))
# sensor back to summing junction (routed dashed line)
(_, sx, sy, _) = nodes["sens"]; (_, mx, my, _) = nodes["sum"]
ax.add_patch(FancyArrowPatch((sx - BW / 2, sy), (mx, my - BH / 2),
             arrowstyle="-|>", mutation_scale=14, color=SIG[1], lw=1.8,
             linestyle="--", connectionstyle="arc3,rad=0.0", zorder=2))
ax.text(3.2, 1.7, "measured position (feedback)", ha="left", va="center",
        fontsize=8, color=SIG[1])
# + / - on summing junction
ax.text(2.35, 4.45, "$+$", fontsize=11, color=SIG[1]); ax.text(2.7, 3.2, "$-$", fontsize=11, color=SIG[1])

handles = [mp.Patch(facecolor=HYD[0], edgecolor=HYD[1], label="Hydraulic"),
           mp.Patch(facecolor=MEC[0], edgecolor=MEC[1], label="Mechanical"),
           mp.Patch(facecolor=SIG[0], edgecolor=SIG[1], label="Signal / control")]
ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
          frameon=True, bbox_to_anchor=(0.5, 1.10))
fig.savefig(os.path.join(FIG, "fig_hopsan_model.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote clean fig_hopsan_model.png")
