"""Hose-length sweep. The pressure/flow sensors sit at the valve manifold, with a
standard mobile-hydraulics pressure hose (3/4 in. bore) running to the cylinder.
We sweep the hose length at several oil temperatures and ask how the
manifold-located sensing degrades the position and force estimates.

Physics: a hose contributes (i) a friction pressure drop that grows with length
and with viscosity (cold oil), so the manifold pressure the sensor reads exceeds
the true cylinder pressure, and (ii) a line compliance that briefly stores flow.
Position is reconstructed by integrating flow, which is conserved along the hose
in steady state, so it is nearly length-insensitive; force is read straight off
the (manifold) pressure, so it carries the full line-loss offset. The sweep thus
sets a maximum hose length for the force specification, worst at cold start."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "codex-reviews")
cfg = Config()
Ffs = cfg.force_fullscale
HOSE_DIA = 0.019                       # 3/4 in. mobile-hydraulics pressure hose
lengths = [0.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
temps = [0.0, 25.0, 50.0]
seeds = list(range(4))
FREQ = 0.03


def run(T, L, sd):
    rng = np.random.default_rng(sd)
    mm = draw_mismatch(cfg, rng)
    truth = simulate_truth(cfg, T, freq=FREQ, mm=mm, hose_len=L, hose_dia=HOSE_DIA)
    meas = make_measurements(truth, cfg, rng)
    obs = CylinderObserver(cfg)
    n = truth["t"].size; xh = np.empty(n); Fh = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, cfg.cyl.stroke); m = truth["t"] >= 1.0
    pe = float(np.max(np.abs(xh - xt)[m])) * 100
    fe = float(np.percentile(np.abs(Fh - truth["F_net"])[m] / Ffs, 99.5)) * 100
    dP = float(np.max(np.abs(truth["P1"] - truth["P1_cyl"])[m])) / 1e5
    return pe, fe, dP


pos = np.zeros((len(temps), len(lengths)))
force = np.zeros_like(pos)
dpk = np.zeros_like(pos)
for i, T in enumerate(temps):
    for j, L in enumerate(lengths):
        ps, fs, dps = [], [], []
        for sd in seeds:
            a, b, c = run(T, L, sd); ps.append(a); fs.append(b); dps.append(c)
        pos[i, j] = max(ps); force[i, j] = max(fs); dpk[i, j] = max(dps)
    row = "  ".join(f"L={L:4.1f}:{force[i,j]:4.1f}%" for j, L in enumerate(lengths))
    print(f"T={T:4.1f}C  pos worst {pos[i].max():.2f}cm  force[{row}]")

json.dump({"lengths_m": lengths, "temps": temps, "hose_dia": HOSE_DIA,
           "pos_cm": pos.tolist(), "force_pctFS": force.tolist(),
           "dP_line_bar": dpk.tolist()},
          open(os.path.join(OUT, "hose_sweep.json"), "w"), indent=2)

L = np.array(lengths)
colors = ["#1f6fb2", "#c77a1f", "#b3202a"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for i, T in enumerate(temps):
    ax[0].plot(L, pos[i], "o-", color=colors[i], label=f"{T:.0f} °C")
ax[0].axhline(POS_ERR_LIMIT * 100, color="r", ls="--", lw=1.3, label="±5 cm limit")
ax[0].set_xlabel("hose length [m]"); ax[0].set_ylabel("position error [cm]")
ax[0].set_title("Position vs hose length"); ax[0].set_ylim(0, POS_ERR_LIMIT * 100 * 1.2)
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8.5, title="oil temp")
for i, T in enumerate(temps):
    ax[1].plot(L, force[i], "o-", color=colors[i], label=f"{T:.0f} °C")
ax[1].axhline(FORCE_ERR_LIMIT_FRAC * 100, color="r", ls="--", lw=1.3, label="±12 % FS limit")
ax[1].set_xlabel("hose length [m]"); ax[1].set_ylabel("force error [% FS]")
ax[1].set_title("Force vs hose length"); ax[1].set_ylim(0, None)
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8.5, title="oil temp")
fig.suptitle("Manifold-sensed hose-length sweep (100 mm cylinder, 3/4 in. hose, 4 seeds)",
             fontsize=11.5, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_hose_sweep.png"), dpi=140)
plt.close(fig)
print("wrote fig_hose_sweep.png")
