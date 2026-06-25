"""Conceptual diagram: the two distinct end-stop corrections.

HOMING RESET  : at a seated stop the absolute position is known, so set x_hat to
                the stop -> removes the *accumulated* position error (now).
BIAS CALIB.   : at a seated stop the true velocity is 0, so measured flow = bias
                + known leakage -> solve for the flow-meter bias -> removes the
                *drift source* (the slope of future error growth).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
plt.rcParams.update({"font.size": 10})

fig = plt.figure(figsize=(11.5, 4.6))
gsA = fig.add_axes([0.02, 0.05, 0.40, 0.9]); gsA.axis("off"); gsA.set_xlim(0, 10); gsA.set_ylim(0, 10)
gsB = fig.add_axes([0.50, 0.14, 0.47, 0.78])

# ---------------- Panel A: the seated condition ----------------
gsA.set_title("(a) At a seated end-stop", fontsize=11, fontweight="bold", loc="left")
# cylinder barrel
gsA.add_patch(Rectangle((1.2, 5.4), 4.4, 1.7, facecolor="#cfe8ff", edgecolor="#1f6fb2", lw=1.6))
# piston (pushed fully right, against the stop)
gsA.add_patch(Rectangle((4.3, 5.5), 0.5, 1.5, facecolor="#9bc4e8", edgecolor="#1f6fb2", lw=1.4))
gsA.add_patch(Rectangle((4.8, 6.05), 2.0, 0.4, facecolor="#9bc4e8", edgecolor="#1f6fb2", lw=1.2))  # rod
# hard stop (wall)
gsA.add_patch(Rectangle((6.8, 4.9), 0.35, 2.7, facecolor="#555", edgecolor="k"))
gsA.text(6.97, 7.8, "hard stop", ha="center", fontsize=8, color="#333")
gsA.annotate("", xy=(6.8, 6.25), xytext=(5.6, 6.25),
             arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2))
gsA.text(3.4, 4.7, r"piston seated:  $v=0$,  $x=L$ (known)", ha="center", fontsize=9.5)

# two callouts
gsA.add_patch(FancyBboxPatch((0.3, 2.2), 9.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15",
             facecolor="#ffe6c7", edgecolor="#c77a1f", lw=1.5))
gsA.text(0.6, 3.35, "HOMING RESET", fontsize=10, fontweight="bold", color="#c77a1f")
gsA.text(0.6, 2.75, r"$x$ is known $\Rightarrow$ set $\hat{x}\leftarrow L$.  Removes the"
         "\n" r"accumulated position error right now.", fontsize=8.8, va="center")

gsA.add_patch(FancyBboxPatch((0.3, 0.2), 9.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15",
             facecolor="#d7f0d7", edgecolor="#2e7d32", lw=1.5))
gsA.text(0.6, 1.35, "BIAS CALIBRATION", fontsize=10, fontweight="bold", color="#2e7d32")
gsA.text(0.6, 0.75, r"$v=0\Rightarrow Q_{meas}=b+q_{leak}$, solve $b=Q_{meas}-q_{leak}$."
         "\n" r"Removes the drift source (the future error slope).", fontsize=8.8, va="center")

# ---------------- Panel B: effect on position error vs time ----------------
seats = [0, 18, 36, 54]      # homing events (s)
T = 70.0
t = np.linspace(0, T, 1400)
slope_full = 0.40            # cm/s drift rate at full (uncalibrated) bias
slope_cal = 0.045            # cm/s after calibration (residual)

# no reset, no calibration: monotone climb
e_none = slope_full * t
# reset only: sawtooth at full slope, zeroed at each seat
e_reset = np.zeros_like(t)
for i, ti in enumerate(t):
    last = max([s for s in seats if s <= ti])
    e_reset[i] = slope_full * (ti - last)
# reset + calibration: slope drops to residual after the first seat
e_both = np.zeros_like(t)
for i, ti in enumerate(t):
    last = max([s for s in seats if s <= ti])
    sl = slope_full if last == 0 else slope_cal
    e_both[i] = sl * (ti - last)

gsB.plot(t, e_none, color="#c0392b", lw=2, label="no reset, no calibration")
gsB.plot(t, e_reset, color="#1f6fb2", lw=2, label="reset only (full drift slope)")
gsB.plot(t, e_both, color="#2e7d32", lw=2, label="reset + calibration (reduced slope)")
gsB.axhline(5, color="r", ls="--", lw=1.3); gsB.text(T*0.5, 5.4, "±5 cm budget", color="r", fontsize=8)
for s in seats:
    gsB.axvline(s, color="0.6", ls=":", lw=1)
gsB.text(seats[1], gsB.get_ylim()[1] if False else 13.5, "", )
gsB.annotate("homing events\n(stop seatings)", xy=(18, 0.3), xytext=(24, 3.0),
             fontsize=8, color="#555", arrowprops=dict(arrowstyle="-|>", color="#888", lw=1))
gsB.annotate("reset = vertical drop\n(zeros accumulated error)", xy=(36, 0.2), xytext=(38, 2.4),
             fontsize=8, color="#1f6fb2")
gsB.annotate("calibration = smaller slope\n(slower drift between seats)", xy=(46, 0.045*10),
             xytext=(40, 9.5), fontsize=8, color="#2e7d32",
             arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1))
gsB.set_xlabel("time [s]"); gsB.set_ylabel("|position error| [cm]")
gsB.set_title("(b) Effect on position-error growth (illustrative)", fontsize=11, fontweight="bold", loc="left")
gsB.set_ylim(0, 14); gsB.set_xlim(0, T); gsB.grid(alpha=0.3); gsB.legend(loc="upper left", fontsize=8.5)

fig.savefig(os.path.join(FIG, "fig_homing_concept.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig_homing_concept.png")
