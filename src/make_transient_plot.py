"""Hopsan-rendered pressure transient across the manifold-to-cylinder hose.

We run the closed-loop plant with a long, cold (high-viscosity) hose so the line
effects are visible, log at high rate, and zoom on a direction reversal / stop
seating. Two things show up that the lumped-line elements produce: (i) a standing
offset between the manifold pressure (what the sensor reads) and the true cylinder
pressure, equal to the friction drop Delta p; and (ii) fast pressure ringing at
the hydraulic natural frequency when the piston slams a stop, riding on the line
compliance. This is the transient the estimator's force channel must live with."""
import os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# high log rate to resolve the transient; long cold hose for visible line loss
cfg = replace(Config(), sens=replace(Config().sens, sample_rate=5000.0))
rng = np.random.default_rng(3)
mm = draw_mismatch(cfg, rng)
tr = simulate_truth(cfg, 0.0, freq=0.05, mm=mm, hose_len=12.0, hose_dia=0.019)

t = tr["t"]
P1m, P2m = tr["P1"] / 1e5, tr["P2"] / 1e5            # manifold (measured), bar
P1c, P2c = tr["P1_cyl"] / 1e5, tr["P2_cyl"] / 1e5    # cylinder (true), bar
dP1 = (tr["P1"] - tr["P1_cyl"]) / 1e5
dP2 = (tr["P2"] - tr["P2_cyl"]) / 1e5

# pick a zoom window around the largest line-drop event (a fast reversal/seat)
i0 = int(1.0 * cfg.sens.sample_rate)                 # skip settling
k = i0 + int(np.argmax(np.abs(dP1[i0:]) + np.abs(dP2[i0:])))
# asymmetric window: a little steady region before the event, then the transient
a = max(0, k - int(0.45 * cfg.sens.sample_rate))
b = min(t.size, k + int(0.30 * cfg.sens.sample_rate))
ts = t[a:b]

fig, ax = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True, constrained_layout=True)

ax[0].plot(ts, P1m[a:b], color="#1f6fb2", lw=2.0, label=r"$P_1$ manifold (measured)")
ax[0].plot(ts, P1c[a:b], color="#1f6fb2", lw=1.3, ls="--", label=r"$P_1$ cylinder (true)")
ax[0].plot(ts, P2m[a:b], color="#c77a1f", lw=2.0, label=r"$P_2$ manifold (measured)")
ax[0].plot(ts, P2c[a:b], color="#c77a1f", lw=1.3, ls="--", label=r"$P_2$ cylinder (true)")
ax[0].set_ylabel("pressure [bar]")
ax[0].set_title("Pressure transient across a 12 m hose at 0 °C "
                "(cold oil): manifold vs. cylinder", fontsize=11, fontweight="bold")
ax[0].legend(fontsize=8.2, ncol=2, loc="upper right"); ax[0].grid(alpha=0.3)

ax[1].plot(ts, dP1[a:b], color="#1f6fb2", lw=1.8, label=r"$\Delta p_1$ (cap line)")
ax[1].plot(ts, dP2[a:b], color="#c77a1f", lw=1.8, label=r"$\Delta p_2$ (rod line)")
ax[1].axhline(0, color="#888", lw=0.8)
ax[1].set_ylabel("line drop $\\Delta p$ [bar]"); ax[1].set_xlabel("time [s]")
ax[1].set_title("Manifold$-$cylinder pressure difference (the force-channel error)",
                fontsize=10.5)
ax[1].legend(fontsize=9, loc="upper right"); ax[1].grid(alpha=0.3)

fig.savefig(os.path.join(FIG, "fig_transient.png"), dpi=150)
plt.close(fig)
print(f"wrote fig_transient.png; peak |dP1|={np.abs(dP1).max():.1f} bar, "
      f"|dP2|={np.abs(dP2).max():.1f} bar")
