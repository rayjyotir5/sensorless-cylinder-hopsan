"""Cylinder-bore sweep: does the result generalise across actuator sizes?
Standard ISO bores 63-160 mm, with area-similarity scaling (rod, mass, dead
volumes, valve max-stroke and controller gains all scaled with bore^2) so each
size is comparably actuated and controlled. Supply 210 bar, T = 30 C, 6 seeds."""
import os, sys, json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, Cylinder, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
bores_mm = [63, 80, 100, 125, 160]
seeds = list(range(6))
T = 30.0
FREQ = 0.04


def observe(truth, meas, cfg):
    obs = CylinderObserver(cfg)
    n = truth["t"].size; xh = np.empty(n); Fh = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, cfg.cyl.stroke); m = truth["t"] >= 1.0
    Ffs = cfg.force_fullscale
    pos = float(np.max(np.abs((xh - xt)[m]))) * 100
    fe = np.abs(Fh - truth["F_net"])[m] / Ffs
    return pos, float(np.percentile(fe, 99.5)) * 100


res = {"bores": bores_mm, "pos_fixed": [], "pos_matched": [], "force": []}
base = Config()
base_qfs = base.sens.q_fullscale
for bore in bores_mm:
    s = bore / 100.0
    cyl = Cylinder(bore=bore/1000.0, rod=0.56*bore/1000.0, stroke=1.0,
                   dead_vol_cap=2.0e-4*s*s, dead_vol_rod=2.0e-4*s*s, mass=200.0*s*s)
    sens_fixed = base.sens                                   # 160 L/min meter (unsized)
    sens_match = replace(base.sens, q_fullscale=base_qfs*s*s)  # meter sized to bore
    cfg_fix = replace(base, cyl=cyl, sens=sens_fixed)
    cfg_mat = replace(base, cyl=cyl, sens=sens_match)
    gain_scale = s*s; xvmax = 0.01*s*s
    pf, pm, ff = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mm = draw_mismatch(cfg_fix, rng)
        truth = simulate_truth(cfg_fix, T, freq=FREQ, mm=mm,
                               gain_scale=gain_scale, xvmax=xvmax)   # truth indep. of sensor FS
        p, f = observe(truth, make_measurements(truth, cfg_fix, np.random.default_rng(30_000+sd)), cfg_fix)
        pf.append(p); ff.append(f)
        p, _ = observe(truth, make_measurements(truth, cfg_mat, np.random.default_rng(30_000+sd)), cfg_mat)
        pm.append(p)
    res["pos_fixed"].append(pf); res["pos_matched"].append(pm); res["force"].append(ff)
    print(f"  bore={bore:3d}mm  A1={cyl.area_cap*1e4:.1f}cm^2  Ffs={cfg_fix.force_fullscale/1e3:.0f}kN  "
          f"pos fixed~{np.median(pf):.2f} matched~{np.median(pm):.2f}cm  force~{np.median(ff):.2f}%FS")

json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "codex-reviews", "bore_sweep.json"), "w"), indent=2)

bo = np.array(bores_mm)
def med(a): return np.median(np.array(a), axis=1)
def band(a): return np.percentile(np.array(a),25,axis=1), np.percentile(np.array(a),75,axis=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for key, c, lab in [("pos_fixed", "#c0392b", "fixed 160 L/min flow meter"),
                    ("pos_matched", "#2e7d32", "flow meter sized to bore")]:
    lo, hi = band(res[key])
    ax[0].fill_between(bo, lo, np.minimum(hi, 8), color=c, alpha=0.13)
    ax[0].plot(bo, np.minimum(med(res[key]), 8.2), "o-", color=c, label=lab)
ax[0].axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.5, label="±5 cm limit")
ax[0].annotate("unsized meter saturates\n(large bore) / noise-floor\n(small bore)",
               xy=(150, 7.2), xytext=(95, 6.4), fontsize=8, color="#c0392b")
ax[0].set_xlabel("bore [mm]"); ax[0].set_ylabel("position error [cm]")
ax[0].set_title("Position error vs bore"); ax[0].legend(fontsize=8.5)
ax[0].set_ylim(0, 8.2); ax[0].grid(alpha=0.3)
lo, hi = band(res["force"])
ax[1].fill_between(bo, lo, hi, color="#c77a1f", alpha=0.15)
ax[1].plot(bo, med(res["force"]), "o-", color="#c77a1f")
ax[1].axhline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls="--", lw=1.5, label="±12 % FS limit")
ax[1].set_xlabel("bore [mm]"); ax[1].set_ylabel("operational force error [% FS]")
ax[1].set_title("Force error vs bore (transducer % FS)"); ax[1].legend(fontsize=9)
ax[1].set_ylim(0, None); ax[1].grid(alpha=0.3)
fig.suptitle("Cylinder-bore sweep, area-similarity scaling (210 bar, T = 30 °C)",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_bore_sweep.png"), dpi=140)
plt.close(fig)
print("wrote fig_bore_sweep.png")
