"""External validation against a real experimental rig: Zhou, Meng, Yuan & Qiao,
"Research and Experimental Analysis of Hydraulic Cylinder Position Control
Mechanism Based on Pressure Detection," Machines 2022, 10, 1.

That paper reconstructs cylinder position from pressure/flow by the same
flow-integration principle we use, and reports the relative error between the
flow it predicts from pressure and the flow actually measured on a real
180 mm-bore / 900 mm-stroke advancing mining cylinder, swept over 5-31 MPa
(their Table 8). Because position is the time integral of flow over area, that
flow-coincidence error is, to first order, their position-estimation error.

Here we configure OUR Hopsan plant + observer to their rig (geometry, load,
emulsion bulk modulus, Coulomb/static friction, 0-40 MPa transducer, 400 L/min
flow meter) and sweep the same supply pressures, with the plant's bulk modulus
following the entrained-air beta_eff(P) model so the real low-pressure softening
they cite ("elastic modulus equivalent to a constant above 10 MPa") is present.
We then plot our position error against their experimental envelope."""
import os, sys, json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, Cylinder, Fluid, Friction
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "codex-reviews")

# ---- Zhou et al. rig ----------------------------------------------------
STROKE = 0.900
cyl = Cylinder(bore=0.180, rod=0.120, stroke=STROKE,
               dead_vol_cap=5e-4, dead_vol_rod=5e-4, mass=500.0)
fluid = Fluid(beta_ref=2.01e9)                 # emulsion bulk modulus 2010 MPa
fric = Friction(coulomb=7000.0, viscous_ref=4.0e3)   # f_k=7000, f_s=1.5*=10500 (~Zhou 10000)
base = Config()
XV = 0.03
FREQ = 0.03
T_LAB = 20.0
seeds = list(range(6))
pressures_MPa = [5, 10, 15, 20, 25, 30]

# ---- Zhou Table 8: |Qn - Qs|/Qs * 100 [%] at 5,10,15,20,25,30 MPa -------
ZHOU = {
    "DN10 extend":  [12.32, 11.29, 0.11, 0.88, 2.99, 3.57],
    "DN10 retract": [20.05, 13.82, 0.51, 0.46, 2.47, 4.90],
    "DN20 extend":  [14.29, 0.03,  0.06, 0.81, 0.49, 1.44],
    "DN20 retract": [12.09, 6.71,  0.01, 3.25, 4.75, 3.96],
}
zhou_arr = np.array(list(ZHOU.values()))     # (4, 6)
zhou_lo, zhou_hi = zhou_arr.min(0), zhou_arr.max(0)
zhou_mean = zhou_arr.mean(0)


def run(Ps_MPa, sd):
    Ps = Ps_MPa * 1e6
    circ = replace(base.circ, supply_pressure=Ps)
    sens = replace(base.sens, q_fullscale=400.0 / 60e3, p_fullscale=40e6)
    cfg = replace(base, cyl=cyl, fluid=fluid, fric=fric, circ=circ, sens=sens)
    rng = np.random.default_rng(sd)
    mm = draw_mismatch(cfg, rng)
    # plant bulk modulus follows entrained-air beta_eff at the operating pressure
    truth = simulate_truth(cfg, T_LAB, freq=FREQ, mm=mm, xvmax=XV, p_rep=Ps)
    meas = make_measurements(truth, cfg, rng)
    obs = CylinderObserver(cfg)              # naive constant-beta observer
    n = truth["t"].size; xh = np.empty(n); Fh = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = obs.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, STROKE); m = truth["t"] >= 1.0
    Ffs = cfg.force_fullscale
    pe = float(np.max(np.abs(xh - xt)[m])) / STROKE * 100        # % of stroke
    fe = float(np.percentile(np.abs(Fh - truth["F_net"])[m] / Ffs, 99.5)) * 100
    return pe, fe


pos = np.zeros((len(pressures_MPa), len(seeds)))
force = np.zeros_like(pos)
for i, P in enumerate(pressures_MPa):
    for j, sd in enumerate(seeds):
        pos[i, j], force[i, j] = run(P, sd)
    print(f"  Ps={P:2d} MPa  pos median {np.median(pos[i]):.2f}%  worst {pos[i].max():.2f}%  "
          f"(Zhou {zhou_lo[i]:.1f}-{zhou_hi[i]:.1f}%)   force {np.median(force[i]):.2f}%FS")

json.dump({"pressures_MPa": pressures_MPa,
           "ours_pos_pct_stroke": pos.tolist(), "ours_force_pctFS": force.tolist(),
           "zhou_table8": ZHOU},
          open(os.path.join(OUT, "zhou_validation.json"), "w"), indent=2)

# ---- figure -------------------------------------------------------------
P = np.array(pressures_MPa)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)

ax[0].fill_between(P, zhou_lo, zhou_hi, color="#888888", alpha=0.30,
                   label="Zhou et al. 2022 experiment (range)")
ax[0].plot(P, zhou_mean, "s--", color="#555555", lw=1.4, ms=5, label="Zhou et al. mean")
ax[0].plot(P, np.median(pos, 1), "o-", color="#1f6fb2", lw=2, ms=6,
           label="this work (observer, median)")
ax[0].plot(P, pos.max(1), "^:", color="#1f6fb2", alpha=0.7, ms=5, label="this work worst-case")
ax[0].axhline(5.0, color="r", ls="--", lw=1.2, label="5 % reference")
ax[0].set_xlabel("system pressure [MPa]")
ax[0].set_ylabel("position estimation error [% of stroke]")
ax[0].set_title("Position error vs. experiment")
ax[0].set_ylim(0, 22); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

ax[1].plot(P, np.median(force, 1), "o-", color="#c77a1f", lw=2, ms=6, label="median")
ax[1].plot(P, force.max(1), "^:", color="#c77a1f", alpha=0.7, ms=5, label="worst-case")
ax[1].axhline(12.0, color="r", ls="--", lw=1.2, label="±12 % FS")
ax[1].set_xlabel("system pressure [MPa]")
ax[1].set_ylabel("force error [% FS]")
ax[1].set_title("Force error (this work)")
ax[1].set_ylim(0, 14); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8.5)

fig.suptitle("Validation against Zhou et al. (2022): 180 mm bore, 900 mm stroke, "
             "500 kg, emulsion, 5–30 MPa", fontsize=11.5, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_zhou_validation.png"), dpi=140)
plt.close(fig)
print("wrote fig_zhou_validation.png")
