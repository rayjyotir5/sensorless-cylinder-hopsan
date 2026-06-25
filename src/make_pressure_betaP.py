"""Supply-pressure sweep WITH the entrained-air beta(P) model in the plant.
Low supply -> air expands -> oil softens -> the compressibility term in the
velocity estimate changes. Compares a naive observer (assumes constant stiff
beta) against a pressure-aware observer (uses beta_eff from measured pressure).
Matched pressure transducer throughout, to isolate the beta(P) effect on POSITION
from the force-SNR effect shown in the constant-beta sweep."""
import os, sys, json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, POS_ERR_LIMIT
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
base = Config()
supplies_bar = [60, 100, 150, 210, 280]
seeds = list(range(6))
T = 30.0
FREQ = 0.03


def pos_err(truth, meas, cfg, kw):
    obs = CylinderObserver(cfg, **kw)
    n = truth["t"].size; xh = np.empty(n)
    for i in range(n):
        xh[i], _ = obs.step(meas["Q1"][i], meas["Q2"][i],
                            meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, cfg.cyl.stroke)
    m = truth["t"] >= 1.0
    return float(np.max(np.abs((xh - xt)[m]))) * 100


res = {"supplies": supplies_bar, "naive": [], "aware": []}
for Ps in supplies_bar:
    Ps_pa = Ps * 1e5
    circ = replace(base.circ, supply_pressure=Ps_pa)
    sens = replace(base.sens, p_fullscale=1.2 * Ps_pa)        # matched transducer
    cfg = replace(base, circ=circ, sens=sens)
    p_rep = 0.5 * Ps_pa                                        # representative chamber pressure
    nv, aw = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mm = draw_mismatch(cfg, rng)
        truth = simulate_truth(cfg, T, freq=FREQ, mm=mm, p_rep=p_rep)   # plant beta_eff(p_rep)
        meas = make_measurements(truth, cfg, np.random.default_rng(20_000 + sd))
        nv.append(pos_err(truth, meas, cfg, {"use_beta_p": False}))     # naive const beta
        aw.append(pos_err(truth, meas, cfg, {"use_beta_p": True}))      # beta_eff(measured P)
    res["naive"].append(nv); res["aware"].append(aw)
    print(f"  Ps={Ps:3d} bar  beta_eff={base.fluid.beta_eff(T, p_rep)/1e9:.2f} GPa  "
          f"pos naive~{np.median(nv):.2f}cm  aware~{np.median(aw):.2f}cm")

json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "codex-reviews", "pressure_betaP.json"), "w"), indent=2)

# ---- figure: (a) beta_eff(P) curve, (b) position vs supply naive/aware ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
Pc = np.linspace(10e5, 300e5, 200)
ax[0].plot(Pc/1e5, [base.fluid.beta_eff(T, p)/1e9 for p in Pc], lw=2, color="C0")
ax[0].axhline(base.fluid.beta(T)/1e9, color="0.5", ls=":", lw=1.5, label="air-free beta(T)")
ax[0].set_xlabel("chamber pressure [bar]"); ax[0].set_ylabel(r"effective $\beta_{eff}$ [GPa]")
ax[0].set_title(f"Entrained-air bulk modulus ({base.fluid.air_fraction*100:.0f}% air)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

sup = np.array(supplies_bar)
def med(a): return np.median(np.array(a), axis=1)
def band(a): return np.percentile(np.array(a),25,axis=1), np.percentile(np.array(a),75,axis=1)
for key, c, lab in [("naive", "#c0392b", "naive observer (constant $\\beta$)"),
                    ("aware", "#2e7d32", "pressure-aware observer ($\\beta_{eff}(P)$)")]:
    lo, hi = band(res[key])
    ax[1].fill_between(sup, lo, hi, color=c, alpha=0.13)
    ax[1].plot(sup, med(res[key]), "o-", color=c, label=lab)
ax[1].axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.5, label="±5 cm limit")
ax[1].set_xlabel("supply pressure [bar]"); ax[1].set_ylabel("position error [cm]")
ax[1].set_title("Position error vs supply pressure (with air $\\beta(P)$)")
ax[1].legend(fontsize=8.5); ax[1].set_ylim(0, None); ax[1].grid(alpha=0.3)
fig.suptitle("Entrained-air bulk modulus: low-pressure position behaviour",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_betaP.png"), dpi=140)
plt.close(fig)
print("wrote fig_betaP.png")
