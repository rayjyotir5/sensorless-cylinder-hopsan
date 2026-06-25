"""Bench-scale, OPEN-LOOP validation sweep, matching the Simscape lab rig
(40 mm bore, 16 mm rod, 0.1 m stroke, 3 kg), driven by an open-loop sine spool
command (no position controller). Sensors are sized to the bench (flow meter to
the small flows, pressure transducer to the 100-bar supply). Because the stroke
is 0.1 m, position error is judged as %-of-stroke (the +/-5 cm industrial target
was 5% of a 1 m stroke -> +/-5% here)."""
import os, sys, json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, Cylinder, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
base = Config()
STROKE = 0.10
POS_LIMIT_FRAC = 0.05      # +/-5% of stroke (= the industrial +/-5 cm / 1 m)
XV, AMP, FREQ = 3e-4, 0.7, 0.08
temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
seeds = list(range(8))

cyl = Cylinder(bore=0.040, rod=0.016, stroke=STROKE,
               dead_vol_cap=1e-5, dead_vol_rod=1e-5, mass=3.0)
circ = replace(base.circ, supply_pressure=100e5)                  # 100 bar bench supply
sens = replace(base.sens, q_fullscale=20.0/60e3, p_fullscale=120e5)  # bench-sized sensors
bench = replace(base, cyl=cyl, circ=circ, sens=sens)
Ffs = bench.force_fullscale


def run(Ti, sd):
    rng = np.random.default_rng(sd)
    mm = draw_mismatch(bench, rng)
    truth = simulate_truth(bench, Ti, freq=FREQ, mm=mm, open_loop=True,
                           spool_amp=AMP, xvmax=XV)
    meas = make_measurements(truth, bench, rng)
    obs = CylinderObserver(bench)
    n = truth["t"].size; xh = np.empty(n); Fh = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0, STROKE); m = truth["t"] >= 1.0
    pe_mm = np.abs(xh - xt)[m] * 1000
    fe = np.abs(Fh - truth["F_net"])[m] / Ffs
    return pe_mm.max(), float(np.percentile(fe, 99.5)) * 100


res = {"temps": temps, "pos_mm": [], "force": []}
for Ti in temps:
    pm, ff = [], []
    for sd in seeds:
        a, b = run(Ti, sd); pm.append(a); ff.append(b)
    res["pos_mm"].append(pm); res["force"].append(ff)
    print(f"  T={Ti:4.1f}C  pos worst {max(pm):.2f}mm ({max(pm)/(STROKE*1000)*100:.1f}% stroke)  "
          f"force {max(ff):.2f}%FS")
worst_mm = max(max(p) for p in res["pos_mm"]); worst_f = max(max(f) for f in res["force"])
print(f"BENCH worst: pos {worst_mm:.2f}mm = {worst_mm/(STROKE*1000)*100:.1f}% stroke "
      f"(limit {POS_LIMIT_FRAC*100:.0f}%), force {worst_f:.2f}%FS")
json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "codex-reviews", "bench_sweep.json"), "w"), indent=2)

T = np.array(temps)
def med(a): return np.median(np.array(a), axis=1)
def hi(a): return np.max(np.array(a), axis=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
ax[0].plot(T, hi(res["pos_mm"])/(STROKE*1000)*100, "o-", color="#1f6fb2", label="worst-case")
ax[0].plot(T, med(res["pos_mm"])/(STROKE*1000)*100, "s--", color="#1f6fb2", alpha=0.6, label="median")
ax[0].axhline(POS_LIMIT_FRAC*100, color="r", ls="--", lw=1.5, label="±5% stroke")
ax[0].set_xlabel("oil temperature [°C]"); ax[0].set_ylabel("position error [% of stroke]")
ax[0].set_title("Bench open-loop: position error"); ax[0].legend(fontsize=8.5)
ax[0].set_ylim(0, POS_LIMIT_FRAC*100*1.3); ax[0].grid(alpha=0.3)
ax[1].plot(T, hi(res["force"]), "o-", color="#c77a1f", label="worst-case 99.5 pct")
ax[1].plot(T, med(res["force"]), "s--", color="#c77a1f", alpha=0.6, label="median")
ax[1].axhline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls="--", lw=1.5, label="±12 % FS")
ax[1].set_xlabel("oil temperature [°C]"); ax[1].set_ylabel("force error [% FS]")
ax[1].set_title("Bench open-loop: force error"); ax[1].legend(fontsize=8.5)
ax[1].set_ylim(0, None); ax[1].grid(alpha=0.3)
fig.suptitle("Bench-scale, open-loop sweep (40 mm bore, 0.1 m stroke, 3 kg, 100 bar)",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_bench_sweep.png"), dpi=140)
plt.close(fig)
print("wrote fig_bench_sweep.png")
