"""Annotated ISO fluid-power schematic of the Hopsan plant, drawn to resemble the
simulator canvas, with the new manifold-to-cylinder hoses highlighted.

HopsanGUI has no headless image export (its canvas export is an interactive GUI
action), so this is a faithful hand-laid ISO-1219-style rendering of the same
model -- the components and connections match the .hmf one-for-one."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
HYD = "#1f6fb2"; MEC = "#2e7d32"; SIG = "#c77a1f"; RED = "#b3202a"; K = "#222"

fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_xlim(0, 24); ax.set_ylim(0, 15); ax.axis("off")
ax.set_aspect("equal")
# faint canvas grid
for gx in range(0, 25): ax.plot([gx, gx], [0, 15], color="#eef1f4", lw=0.6, zorder=0)
for gy in range(0, 16): ax.plot([0, 24], [gy, gy], color="#eef1f4", lw=0.6, zorder=0)


def line(p, q, c=K, lw=2.0, z=2, ls="-"):
    ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, zorder=z, ls=ls))


def poly(pts, fc="none", ec=K, lw=1.8, z=3):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))


def label(x, y, s, c=K, fs=10, w="normal", style="normal", ha="center"):
    ax.text(x, y, s, color=c, fontsize=fs, fontweight=w, style=style, ha=ha, va="center", zorder=6)


# ---- tank + pump + relief (supply) -------------------------------------
tx, ty = 2.0, 1.6
line((tx - 0.7, ty), (tx + 0.7, ty)); line((tx - 0.7, ty), (tx - 0.7, ty + 0.5))
line((tx + 0.7, ty), (tx + 0.7, ty + 0.5)); line((tx, ty), (tx, ty + 0.5))
label(tx, ty - 0.5, "tank", fs=9)
# pump (circle + triangle)
pcx, pcy = 2.0, 4.2
ax.add_patch(Circle((pcx, pcy), 0.8, facecolor="white", edgecolor=HYD, lw=2.2, zorder=3))
poly([(pcx - 0.45, pcy + 0.45), (pcx - 0.45, pcy - 0.45), (pcx + 0.5, pcy)], fc=HYD, ec=HYD, z=4)
line((tx, ty + 0.5), (pcx, pcy - 0.8), c=HYD)
label(pcx, pcy - 1.25, "pump", c=HYD, fs=9)
# supply rail up
line((pcx, pcy + 0.8), (pcx, 9.0), c=HYD, lw=2.4)
line((pcx, 9.0), (6.2, 9.0), c=HYD, lw=2.4)
# relief valve (box + arrow + spring) teed off the rail
rvx, rvy = 4.0, 6.4
ax.add_patch(Rectangle((rvx - 0.6, rvy - 0.6), 1.2, 1.2, facecolor="white", edgecolor=HYD, lw=2.0, zorder=3))
ax.add_patch(FancyArrowPatch((rvx - 0.35, rvy - 0.35), (rvx + 0.35, rvy + 0.35),
             arrowstyle="-|>", mutation_scale=12, color=HYD, lw=1.6, zorder=4))
line((pcx, 7.4), (rvx, 7.4), c=HYD); line((rvx, 7.4), (rvx, rvy + 0.6), c=HYD)
line((rvx, rvy - 0.6), (rvx, 2.2), c=HYD); line((rvx, 2.2), (tx + 0.7, 2.2), c=HYD); line((tx + 0.7, 2.2), (tx + 0.7, ty + 0.5), c=HYD)
label(rvx + 1.35, rvy, "relief\nvalve", c=HYD, fs=8.5)

# ---- 4/3 servo valve ---------------------------------------------------
vx, vy, vw, vh = 6.2, 8.2, 3.3, 1.6
for i in range(3):
    ax.add_patch(Rectangle((vx + i * vw / 3, vy), vw / 3, vh, facecolor="white", edgecolor=K, lw=1.6, zorder=3))
# centre envelope: blocked (T symbols); side envelopes: arrows
cxm = vx + vw / 2
ax.add_patch(FancyArrowPatch((vx + 0.25, vy + 0.4), (vx + vw / 3 - 0.25, vy + 1.2),
             arrowstyle="-|>", mutation_scale=9, color=K, lw=1.3, zorder=4))
ax.add_patch(FancyArrowPatch((vx + 2 * vw / 3 + 0.25, vy + 1.2), (vx + vw - 0.25, vy + 0.4),
             arrowstyle="-|>", mutation_scale=9, color=K, lw=1.3, zorder=4))
label(cxm, vy + vh + 0.55, "4/3 servo valve", fs=9.5, c=K)
# spool actuators (control)
line((vx, vy + vh / 2), (vx - 0.7, vy + vh / 2)); ax.add_patch(Rectangle((vx - 1.3, vy + vh / 2 - 0.35), 0.6, 0.7, facecolor="white", edgecolor=SIG, lw=1.6, zorder=3))
line((vx + vw, vy + vh / 2), (vx + vw + 0.7, vy + vh / 2)); ax.add_patch(Rectangle((vx + vw + 0.7, vy + vh / 2 - 0.35), 0.6, 0.7, facecolor="white", edgecolor=SIG, lw=1.6, zorder=3))
# P port (supply in, bottom centre), T port (to tank)
line((cxm, vy), (cxm, 9.0), c=HYD, lw=2.4)   # supply rail meets P
label(cxm + 0.3, 8.0, "P", fs=8, c=HYD, ha="left")
line((vx + 0.5, vy), (vx + 0.5, 3.0), c=HYD); line((vx + 0.5, 3.0), (rvx, 3.0), c=HYD)  # T to tank rail
label(vx + 0.8, 7.9, "T", fs=8, c=HYD, ha="left")

# A, B ports at top of valve -> manifold nodes
ax_port = vx + vw / 3 * 0.5; bx_port = vx + vw / 3 * 2.5
line((ax_port, vy + vh), (ax_port, 11.0), c=HYD, lw=2.4)
line((bx_port, vy + vh), (bx_port, 11.0), c=HYD, lw=2.4)
label(ax_port - 0.3, vy + vh + 0.3, "A", fs=8, c=HYD, ha="right")
label(bx_port - 0.3, vy + vh + 0.3, "B", fs=8, c=HYD, ha="right")

# ---- manifold sensor taps (P,Q) ----------------------------------------
for px, lab in [(ax_port, "Q1,P1"), (bx_port, "Q2,P2")]:
    ax.add_patch(Circle((px, 11.0), 0.16, facecolor=SIG, edgecolor=SIG, zorder=6))
for gx in (ax_port, bx_port):
    ax.add_patch(Circle((gx - 0.0, 12.0), 0.42, facecolor="white", edgecolor=SIG, lw=1.7, zorder=5))
    line((gx, 12.0), (gx + 0.28, 12.28), c=SIG, lw=1.4, z=6)
    line((gx, 11.16), (gx, 11.58), c=SIG, lw=1.4)
label((ax_port + bx_port) / 2, 13.0, "$P,Q$ sensors (at manifold)", c=SIG, fs=10, w="bold")

# ---- HOSES (highlighted, new) ------------------------------------------
def hose(x, y0, y1):
    """vertical hose drawn as a coil to read as a flexible line."""
    ys = np.linspace(y0, y1, 60)
    xs = x + 0.16 * np.sin(np.linspace(0, 12 * np.pi, 60))
    ax.add_line(Line2D(xs, ys, color=HYD, lw=2.6, zorder=4))


# horizontal run from valve A/B (top) toward the cylinder on the right
line((ax_port, 11.0), (12.2, 11.0), c=HYD, lw=2.6)
line((bx_port, 11.0), (12.2, 10.0), c=HYD, lw=2.6)
# the flexible-hose stretch
for yy in (11.0, 10.0):
    xs = np.linspace(12.2, 15.4, 80)
    ax.add_line(Line2D(xs, yy + 0.16 * np.sin(np.linspace(0, 26 * np.pi, 80)), color=HYD, lw=2.6, zorder=4))
# highlight box around the hose stretch
ax.add_patch(FancyBboxPatch((12.0, 9.3), 3.7, 2.4, boxstyle="round,pad=0.05,rounding_size=0.15",
             facecolor="none", edgecolor=RED, lw=1.8, ls="--", zorder=5))
label(13.85, 12.05, "hoses to cylinder", c=RED, fs=9.5, w="bold")
label(13.85, 8.95, "line loss $\\Delta p$ + compliance", c=RED, fs=8.5, style="italic")

# ---- cylinder ----------------------------------------------------------
cx0, cy0, cw, ch = 15.6, 8.6, 4.6, 2.8
ax.add_patch(Rectangle((cx0, cy0), cw, ch, facecolor="white", edgecolor=HYD, lw=2.4, zorder=3))
# piston + rod
pistx = cx0 + 1.7
ax.add_patch(Rectangle((pistx, cy0 + 0.08), 0.4, ch - 0.16, facecolor="#9bc4e8", edgecolor=HYD, lw=1.6, zorder=4))
ax.add_patch(Rectangle((pistx + 0.4, cy0 + ch / 2 - 0.18), cw - 1.7 + 0.6, 0.36, facecolor="#9bc4e8", edgecolor=HYD, lw=1.3, zorder=4))
# ports from hoses into cylinder
line((15.4, 11.0), (cx0 + 0.8, 11.0), c=HYD, lw=2.6); line((cx0 + 0.8, 11.0), (cx0 + 0.8, cy0 + ch), c=HYD, lw=2.6)
line((15.4, 10.0), (cx0 + cw - 0.8, 10.0), c=HYD, lw=2.6); line((cx0 + cw - 0.8, 10.0), (cx0 + cw - 0.8, cy0 + ch), c=HYD, lw=2.6)
label(cx0 + cw / 2, cy0 - 0.5, "double-acting cylinder", c=HYD, fs=9.5)
# load mass
ax.add_patch(Rectangle((cx0 + cw + 0.5, cy0 + ch / 2 - 0.7, ), 1.3, 1.4, facecolor="#d7f0d7", edgecolor=MEC, lw=2.0, zorder=3))
line((cx0 + cw + 0.5, cy0 + ch / 2), (pistx + 0.4 + cw - 1.7 + 0.6, cy0 + ch / 2), c=MEC, lw=2.0)
label(cx0 + cw + 1.15, cy0 + ch / 2 - 1.15, "load\n+ stops", c=MEC, fs=8.5)
# position sensor (grey, NOT used by estimator)
ax.add_patch(Circle((cx0 + cw / 2, cy0 - 1.7), 0.45, facecolor="white", edgecolor="#999", lw=1.6, ls="--", zorder=3))
line((cx0 + cw / 2, cy0), (cx0 + cw / 2, cy0 - 1.25), c="#999", lw=1.4, ls="--")
label(cx0 + cw / 2 + 0.7, cy0 - 1.7, "position sensor\n(truth only --\nnot used)", c="#888", fs=7.5, ha="left", style="italic")

# ---- control chain (signal, top-left) ----------------------------------
def sbox(x, y, w, h, s, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                 facecolor="#ffe6c7", edgecolor=SIG, lw=1.7, zorder=3))
    label(x, y, s, c=K, fs=fs)


cyR = 14.3
sbox(2.0, cyR, 2.2, 1.0, "sine ref")
ax.add_patch(Circle((4.7, cyR), 0.45, facecolor="#ffe6c7", edgecolor=SIG, lw=1.7, zorder=3)); label(4.7, cyR, "$\\Sigma$")
sbox(7.2, cyR, 2.0, 1.0, "PI ctrl")
ax.add_patch(FancyArrowPatch((3.1, cyR), (4.25, cyR), arrowstyle="-|>", mutation_scale=12, color=SIG, lw=1.8, zorder=4))
ax.add_patch(FancyArrowPatch((5.15, cyR), (6.2, cyR), arrowstyle="-|>", mutation_scale=12, color=SIG, lw=1.8, zorder=4))
# PI -> valve spool actuator (left)
ax.add_patch(FancyArrowPatch((7.2, cyR - 0.5), (vx - 1.0, vy + vh / 2 + 0.4), arrowstyle="-|>",
             mutation_scale=12, color=SIG, lw=1.8, zorder=4, connectionstyle="arc3,rad=0.25"))
label(4.7, 11.0, "spool cmd", c=SIG, fs=8)
# position feedback -> sum (dashed)
ax.add_patch(FancyArrowPatch((cx0 + cw / 2 - 0.45, cy0 - 1.7), (4.7, cyR - 0.5), arrowstyle="-|>",
             mutation_scale=11, color="#999", lw=1.4, ls=(0, (4, 3)), zorder=2, connectionstyle="arc3,rad=0.28"))

# legend
handles = [Line2D([0], [0], color=HYD, lw=2.5, label="hydraulic"),
           Line2D([0], [0], color=MEC, lw=2.5, label="mechanical"),
           Line2D([0], [0], color=SIG, lw=2.5, label="signal / control"),
           Line2D([0], [0], color=RED, lw=2.0, ls="--", label="new: hoses (line loss)")]
ax.legend(handles=handles, loc="lower left", fontsize=9, ncol=2, frameon=True, bbox_to_anchor=(0.0, -0.02))
ax.set_title("Hopsan plant schematic (ISO fluid-power symbols): "
             "manifold-sensed, hoses to the cylinder", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_hopsan_canvas.png"), dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote fig_hopsan_canvas.png")
