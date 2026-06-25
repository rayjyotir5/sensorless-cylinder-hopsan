"""Visualize the Monte-Carlo ensemble: all 48 error-vs-time traces overlaid,
coloured by oil temperature, against the acceptance bands."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
cfg = Config()
temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
seeds = list(range(8))
Ffs = cfg.force_fullscale

norm = Normalize(vmin=min(temps), vmax=max(temps))
cmap = cm.viridis

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                               constrained_layout=True)

worst_pos = 0.0
worst_frc = 0.0
runs = 0
for T in temps:
    truth = simulate_truth(cfg, T)          # plant depends only on T -> cache
    t = truth["t"]
    x_true = np.clip(truth["x"], 0.0, cfg.cyl.stroke)
    color = cmap(norm(T))
    for sd in seeds:
        rng = np.random.default_rng(sd)
        meas = make_measurements(truth, cfg, rng)
        obs = CylinderObserver(cfg)
        n = t.size
        xh = np.empty(n); Fh = np.empty(n)
        for i in range(n):
            xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                    meas["P1"][i], meas["P2"][i], meas["T"][i])
        pe = (xh - x_true) * 100.0
        fe = (Fh - truth["F_net"]) / Ffs * 100.0
        m = t >= 1.0
        worst_pos = max(worst_pos, np.max(np.abs(pe[m])))
        worst_frc = max(worst_frc, np.max(np.abs(fe[m])))
        ax1.plot(t, pe, color=color, lw=0.6, alpha=0.55)
        ax2.plot(t, fe, color=color, lw=0.6, alpha=0.55)
        runs += 1

# acceptance bands
for ax, lim, lab, unit in ((ax1, POS_ERR_LIMIT*100, "position", "cm"),
                           (ax2, FORCE_ERR_LIMIT_FRAC*100, "force", "% FS")):
    ax.axhspan(-lim, lim, color="green", alpha=0.06)
    ax.axhline(lim, color="r", ls="--", lw=1.4)
    ax.axhline(-lim, color="r", ls="--", lw=1.4, label=f"±{lim:g} {unit} limit")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

ax1.set_ylabel("position error [cm]")
ax1.set_ylim(-POS_ERR_LIMIT*100*1.15, POS_ERR_LIMIT*100*1.15)
ax1.set_title(f"Monte-Carlo ensemble: {runs} runs "
              f"(6 temperatures × 8 seeds) — worst |pos| = {worst_pos:.2f} cm, "
              f"worst |force| = {worst_frc:.2f} % FS")
ax2.set_ylabel("force error [% FS]"); ax2.set_xlabel("time [s]")
ax2.set_ylim(-FORCE_ERR_LIMIT_FRAC*100*1.15, FORCE_ERR_LIMIT_FRAC*100*1.15)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.9, pad=0.015)
cb.set_label("oil temperature [°C]")

fig.savefig(os.path.join(FIG, "fig_mc_ensemble.png"), dpi=140)
plt.close(fig)
print(f"wrote fig_mc_ensemble.png  ({runs} runs, worst pos {worst_pos:.2f} cm, "
      f"force {worst_frc:.2f} %FS)")
