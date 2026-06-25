"""Explanatory diagrams (vector-clean, matplotlib) to make the paper accessible
from first principles:

  fig_principle.png       -- the coupled cylinder: flow->velocity->position,
                             pressure->force, with no position sensor.
  fig_observer_block.png  -- the complementary observer as a signal-flow graph.
  fig_manifold.png        -- manifold-located sensing and the hose line loss.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
plt.rcParams.update({"font.size": 10.5})

HYD = ("#cfe8ff", "#1f6fb2")    # hydraulic
MEC = ("#d7f0d7", "#2e7d32")    # mechanical
SIG = ("#ffe6c7", "#c77a1f")    # signal / control
RED = "#b3202a"


def arrow(ax, p, q, color="#333", lw=2.0, style="-|>", ms=15, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                 color=color, lw=lw, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=5))


# ===================================================================== #
# 1. First-principles coupled cylinder                                  #
# ===================================================================== #
def principle():
    fig, ax = plt.subplots(figsize=(9.2, 4.5)); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)

    # barrel
    bx, by, bw, bh = 2.2, 3.4, 8.6, 2.6
    ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="#eef6ff",
                 edgecolor=HYD[1], lw=2.2, zorder=2))
    # end-stop walls
    ax.add_patch(Rectangle((bx - 0.18, by - 0.2), 0.18, bh + 0.4, facecolor="#555", edgecolor="none"))
    ax.add_patch(Rectangle((bx + bw, by - 0.2), 0.18, bh + 0.4, facecolor="#555", edgecolor="none"))
    # piston
    px = bx + 4.3
    ax.add_patch(Rectangle((px, by + 0.06), 0.55, bh - 0.12, facecolor="#9bc4e8",
                 edgecolor=HYD[1], lw=1.6, zorder=4))
    # rod through rod side to the right
    ax.add_patch(Rectangle((px + 0.55, by + bh / 2 - 0.22), bw - 4.85 + 0.18, 0.44,
                 facecolor="#9bc4e8", edgecolor=HYD[1], lw=1.3, zorder=4))
    # chamber fills
    ax.add_patch(Rectangle((bx, by + 0.06), 4.3, bh - 0.12, facecolor="#cfe8ff", alpha=0.7, zorder=1))
    ax.add_patch(Rectangle((px + 0.55, by + 0.06), bw - 4.85, bh - 0.12, facecolor="#dbeaf7", alpha=0.7, zorder=1))

    ax.text(bx + 2.1, by + bh - 0.45, "chamber 1 (cap)", ha="center", fontsize=9.5, color=HYD[1])
    ax.text(bx + 2.1, by + 0.42, r"$P_1,\ A_1$", ha="center", fontsize=11, color=HYD[1])
    ax.text(bx + 6.7, by + bh - 0.45, "chamber 2 (rod)", ha="center", fontsize=9.5, color=HYD[1])
    ax.text(bx + 6.7, by + 0.42, r"$P_2,\ A_2$", ha="center", fontsize=11, color=HYD[1])

    # flow arrows in/out
    arrow(ax, (bx + 1.0, by - 1.1), (bx + 1.0, by - 0.05), color=HYD[1], lw=2.4)
    ax.text(bx + 1.0, by - 1.4, r"$Q_1$", ha="center", color=HYD[1], fontsize=12)
    arrow(ax, (bx + 7.6, by - 0.05), (bx + 7.6, by - 1.1), color=HYD[1], lw=2.4)
    ax.text(bx + 7.6, by - 1.4, r"$Q_2$", ha="center", color=HYD[1], fontsize=12)

    # velocity + position
    arrow(ax, (px + 0.27, by + bh + 0.35), (px + 1.6, by + bh + 0.35), color=MEC[1], lw=2.2)
    ax.text(px + 0.95, by + bh + 0.62, r"$v=\dot x$", ha="center", color=MEC[1], fontsize=11)
    # position dimension
    ax.annotate("", xy=(px + 0.27, by - 0.5), xytext=(bx, by - 0.5),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.3))
    ax.text(bx + 2.15, by - 0.85, r"$x$ (stroke)", ha="center", color="#444", fontsize=10)
    # external force on rod
    arrow(ax, (bx + bw + 1.5, by + bh / 2), (bx + bw + 0.2, by + bh / 2), color=MEC[1], lw=2.4)
    ax.text(bx + bw + 1.7, by + bh / 2, r"load $F$", ha="left", va="center", color=MEC[1], fontsize=10.5)

    # the two governing relations
    ax.add_patch(FancyBboxPatch((0.5, 7.0), 6.0, 1.5, boxstyle="round,pad=0.1,rounding_size=0.12",
                 facecolor=MEC[0], edgecolor=MEC[1], lw=1.6))
    ax.text(3.5, 7.75, "flow $\\Rightarrow$ motion\n"
            r"$A_1\dot x \approx Q_1 \Rightarrow x=\int\! v\,dt$",
            ha="center", va="center", fontsize=11)
    ax.add_patch(FancyBboxPatch((7.3, 7.0), 6.2, 1.5, boxstyle="round,pad=0.1,rounding_size=0.12",
                 facecolor=SIG[0], edgecolor=SIG[1], lw=1.6))
    ax.text(10.4, 7.75, "pressure $\\Rightarrow$ force\n"
            r"$F = P_1 A_1 - P_2 A_2$",
            ha="center", va="center", fontsize=11)
    ax.text(7.0, 1.05, "only $P_1,P_2,Q_1,Q_2$ are measured — no position sensor",
            ha="center", fontsize=10.5, style="italic", color=RED)
    fig.savefig(os.path.join(FIG, "fig_principle.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_principle.png")


# ===================================================================== #
# 2. Complementary observer signal-flow                                 #
# ===================================================================== #
def observer_block():
    fig, ax = plt.subplots(figsize=(11, 4.7)); ax.axis("off")
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 8)

    def box(x, y, w, h, label, col, fs=9.8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.1",
                     facecolor=col[0], edgecolor=col[1], lw=1.7, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)

    # measured inputs
    box(1.5, 5.6, 2.3, 1.4, "Measured\n$Q_1,Q_2,P_1,P_2$\n$T$ (noisy)", HYD, 9.2)
    # velocity estimate
    box(5.2, 5.6, 2.9, 1.5,
        "Velocity from\ncontinuity (both\nchambers, SNR blend)", SIG, 9.0)
    # integrator
    box(9.0, 5.6, 1.7, 1.2, r"$\int dt$", SIG, 13)
    # x_hat output
    box(12.0, 5.6, 2.0, 1.1, r"$\hat x$  position", MEC, 10.5)
    # force path
    box(9.0, 2.0, 3.0, 1.1, r"$\hat F = P_1A_1 - P_2A_2$", SIG, 10)
    box(12.0, 2.0, 2.0, 1.1, r"$\hat F$  force", MEC, 10.5)
    # end-stop detector
    box(5.2, 2.4, 2.9, 1.5, "End-stop detector\n(low $Q$, high $P$,\nsteady $\\dot P$)", MEC, 9.0)

    # forward path
    arrow(ax, (2.65, 5.6), (3.75, 5.6), color=SIG[1])
    arrow(ax, (6.65, 5.6), (8.15, 5.6), color=SIG[1])
    arrow(ax, (9.85, 5.6), (11.0, 5.6), color=SIG[1])
    # force path
    arrow(ax, (1.9, 4.9), (8.0, 2.4), color=SIG[1], rad=-0.05)
    arrow(ax, (10.5, 2.0), (11.0, 2.0), color=SIG[1])
    # inputs to detector
    arrow(ax, (1.9, 4.9), (4.2, 3.0), color=MEC[1], rad=0.05)
    # homing reset: detector -> integrator output (x_hat node)
    arrow(ax, (6.3, 3.0), (11.4, 5.05), color=RED, rad=-0.18, lw=2.0)
    ax.text(9.2, 3.45, "homing reset  $\\hat x\\!\\leftarrow\\!$ stop",
            ha="center", color=RED, fontsize=9.2, rotation=14)
    # bias calibration: detector -> velocity box
    arrow(ax, (5.2, 3.15), (5.2, 4.85), color=RED, rad=0.0, lw=2.0)
    ax.text(5.32, 4.0, "bias\n$b_i$", ha="left", va="center", color=RED, fontsize=9.2)

    ax.text(7.6, 7.4, "Complementary observer: fast flow-derived velocity, "
            "corrected at the end-stops", ha="center", fontsize=10.5, style="italic", color="#333")
    fig.savefig(os.path.join(FIG, "fig_observer_block.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_observer_block.png")


# ===================================================================== #
# 3. Manifold sensing + hose loss                                       #
# ===================================================================== #
def manifold():
    fig, ax = plt.subplots(figsize=(10.5, 4.0)); ax.axis("off")
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)

    # valve / manifold block
    ax.add_patch(FancyBboxPatch((0.6, 2.2), 2.6, 2.6, boxstyle="round,pad=0.03,rounding_size=0.1",
                 facecolor=HYD[0], edgecolor=HYD[1], lw=1.8))
    ax.text(1.9, 3.5, "Valve\nmanifold", ha="center", va="center", fontsize=10)
    # sensor taps
    ax.add_patch(FancyBboxPatch((0.5, 5.2), 2.8, 1.0, boxstyle="round,pad=0.03,rounding_size=0.1",
                 facecolor=SIG[0], edgecolor=SIG[1], lw=1.6))
    ax.text(1.9, 5.7, "$P,Q$ sensors", ha="center", va="center", fontsize=10)
    arrow(ax, (1.9, 5.15), (1.9, 4.85), color=SIG[1], lw=1.6, ls=":")
    ax.plot([1.9], [4.5], "o", ms=6, color=SIG[1])

    # two hoses
    for yy, lab in [(4.0, "cap line"), (3.0, "rod line")]:
        ax.add_patch(Rectangle((3.2, yy - 0.12), 7.0, 0.24, facecolor="#cfe8ff", edgecolor=HYD[1], lw=1.2))
        # wavy hose hint
    ax.annotate("", xy=(10.2, 5.0), xytext=(3.2, 5.0),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
    ax.text(6.7, 5.25, "hose length $L$  (3/4 in.)", ha="center", color="#444", fontsize=10)

    # cylinder
    ax.add_patch(Rectangle((10.4, 2.5), 3.8, 2.0, facecolor="#eef6ff", edgecolor=HYD[1], lw=2.0))
    ax.add_patch(Rectangle((11.9, 2.55), 0.45, 1.9, facecolor="#9bc4e8", edgecolor=HYD[1], lw=1.4))
    ax.add_patch(Rectangle((12.35, 3.3), 2.0, 0.4, facecolor="#9bc4e8", edgecolor=HYD[1], lw=1.2))
    ax.text(12.3, 4.75, "cylinder", ha="center", fontsize=10, color=HYD[1])

    # pressure labels
    ax.text(3.4, 4.45, r"$P_{\mathrm{manifold}}$ (measured)", ha="left", color=SIG[1], fontsize=9.8)
    ax.text(10.3, 2.15, r"$P_{\mathrm{cyl}} = P_{\mathrm{manifold}} - \Delta p$",
            ha="center", color=RED, fontsize=9.8)
    # drop annotation
    ax.text(6.7, 3.5, r"$\Delta p = Q/K_c \ \propto\ \mu(T)\,L$", ha="center",
            color=RED, fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=1.0))

    ax.text(7.5, 0.7, "position: flow is conserved along the hose $\\Rightarrow$ robust  |  "
            "force: reads $P_{\\mathrm{manifold}} \\Rightarrow$ off by $\\Delta p\\,A$",
            ha="center", fontsize=9.8, style="italic", color="#333")
    fig.savefig(os.path.join(FIG, "fig_manifold.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_manifold.png")


if __name__ == "__main__":
    principle()
    observer_block()
    manifold()
