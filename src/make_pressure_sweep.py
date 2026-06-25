"""Sweep supply pressure (fixed 100 mm cylinder) to test how low- vs high-
pressure operation changes position and force estimation. Two pressure-transducer
configurations: a fixed 250-bar transducer, and one matched to each supply
(FS = 1.2*supply). The plant truth depends only on (supply, seed-mismatch), so it
is cached and reused across the two transducer configs.

Reference frequency lowered to 0.03 Hz so the valve is not flow-limited at low
supply (peak demand stays within valve capacity) -- isolating the estimation
question from an actuation limit."""
import os, sys, json
from dataclasses import replace
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
base = Config()
supplies_bar = [60, 100, 150, 210, 280]
seeds = list(range(6))
T = 30.0
FREQ = 0.03


def metrics(truth, meas, cfg):
    obs = CylinderObserver(cfg)
    n = truth["t"].size
    xh = np.empty(n); Fh = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, cfg.cyl.stroke)
    m = truth["t"] >= 1.0
    Ffs = cfg.force_fullscale
    pos = float(np.max(np.abs((xh - xt)[m]))) * 100
    fe = np.abs(Fh - truth["F_net"])[m] / Ffs
    return pos, float(np.percentile(fe, 99.5)) * 100


res = {"supplies": supplies_bar,
       "fixed": {"pos": [], "force": []}, "matched": {"pos": [], "force": []}}
for Ps in supplies_bar:
    Ps_pa = Ps * 1e5
    circ = replace(base.circ, supply_pressure=Ps_pa)
    sens_fixed = base.sens                                  # 250-bar transducer
    sens_match = replace(base.sens, p_fullscale=1.2 * Ps_pa)
    cfg_f = replace(base, circ=circ, sens=sens_fixed)
    cfg_m = replace(base, circ=circ, sens=sens_match)
    pf, ff, pm, fm = [], [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mm = draw_mismatch(cfg_f, rng)
        truth = simulate_truth(cfg_f, T, freq=FREQ, mm=mm)   # truth independent of sens
        # fixed transducer
        rng_f = np.random.default_rng(10_000 + sd)
        p, f = metrics(truth, make_measurements(truth, cfg_f, rng_f), cfg_f)
        pf.append(p); ff.append(f)
        # matched transducer (same truth)
        rng_m = np.random.default_rng(10_000 + sd)
        p, f = metrics(truth, make_measurements(truth, cfg_m, rng_m), cfg_m)
        pm.append(p); fm.append(f)
    res["fixed"]["pos"].append(pf); res["fixed"]["force"].append(ff)
    res["matched"]["pos"].append(pm); res["matched"]["force"].append(fm)
    print(f"  Ps={Ps:3d} bar  pos~{np.median(pf):.2f}cm  "
          f"force fixed~{np.median(ff):.2f}%  matched~{np.median(fm):.2f}%FS")

json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "codex-reviews", "pressure_sweep.json"), "w"), indent=2)

# ---- figure ----
sup = np.array(supplies_bar)
def med(a): return np.median(np.array(a), axis=1)
def band(a): return np.percentile(np.array(a),25,axis=1), np.percentile(np.array(a),75,axis=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
# position
for key, c, lab in [("fixed", "#1f6fb2", "fixed 250-bar transducer"),
                    ("matched", "#2e7d32", "transducer matched to supply")]:
    lo, hi = band(res[key]["pos"])
    ax[0].fill_between(sup, lo, hi, color=c, alpha=0.13)
    ax[0].plot(sup, med(res[key]["pos"]), "o-", color=c, label=lab)
ax[0].axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.5, label="±5 cm limit")
ax[0].set_xlabel("supply pressure [bar]"); ax[0].set_ylabel("position error [cm]")
ax[0].set_title("Position error vs supply pressure"); ax[0].legend(fontsize=8.5)
ax[0].set_ylim(0, POS_ERR_LIMIT*100*1.2); ax[0].grid(alpha=0.3)
# force
for key, c, lab in [("fixed", "#1f6fb2", "fixed 250-bar transducer"),
                    ("matched", "#2e7d32", "transducer matched to supply")]:
    lo, hi = band(res[key]["force"])
    ax[1].fill_between(sup, lo, hi, color=c, alpha=0.13)
    ax[1].plot(sup, med(res[key]["force"]), "o-", color=c, label=lab)
ax[1].axhline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls="--", lw=1.5, label="±12 % FS limit")
ax[1].set_xlabel("supply pressure [bar]"); ax[1].set_ylabel("operational force error [% FS]")
ax[1].set_title("Force error vs supply pressure"); ax[1].legend(fontsize=8.5)
ax[1].set_ylim(0, None); ax[1].grid(alpha=0.3)
fig.suptitle("Supply-pressure sweep (100 mm cylinder, T = 30 °C, 6 seeds)",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_pressure_sweep.png"), dpi=140)
plt.close(fig)
print("wrote fig_pressure_sweep.png")
