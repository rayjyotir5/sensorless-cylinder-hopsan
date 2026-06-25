"""Head-to-head: well-scaled EKF vs the complementary observer, on identical
truth + measurements. Reports full-run and post-convergence (after the first
homing, t>=8 s) errors, force error, and the EKF covariance calibration. Also
saves a representative trajectory with the EKF +/-2 sigma band."""
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
from cylinder_sim.ekf import CylinderEKF

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
cfg = Config(); Ffs = cfg.force_fullscale
temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
seeds = list(range(4))
T_CONV = 8.0    # post-convergence window start (after first homing)


def drive(est, meas, n, want_sigma=False):
    xh = np.empty(n); Fh = np.empty(n); sg = np.empty(n)
    for i in range(n):
        xh[i], Fh[i] = est.step(meas["Q1"][i], meas["Q2"][i],
                                meas["P1"][i], meas["P2"][i], meas["T"][i])
        sg[i] = est.pos_sigma() if want_sigma else 0.0
    return xh, Fh, sg


agg = {"obs": {"max": [], "rms": [], "ss_max": [], "force": []},
       "ekf": {"max": [], "rms": [], "ss_max": [], "force": [], "cov2s": []}}
rep = None
for Ti in temps:
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mm = draw_mismatch(cfg, rng)
        truth = simulate_truth(cfg, Ti, mm=mm)
        meas = make_measurements(truth, cfg, rng)
        t = truth["t"]; xt = np.clip(truth["x"], 0, cfg.cyl.stroke); n = t.size
        full = t >= 1.0; ss = t >= T_CONV
        xo, Fo, _ = drive(CylinderObserver(cfg), meas, n)
        xe, Fe, se = drive(CylinderEKF(cfg), meas, n, want_sigma=True)
        for tag, xh, Fh in [("obs", xo, Fo), ("ekf", xe, Fe)]:
            pe = np.abs(xh - xt) * 100; fe = np.abs(Fh - truth["F_net"]) / Ffs * 100
            agg[tag]["max"].append(pe[full].max())
            agg[tag]["rms"].append(np.sqrt((pe[full]**2).mean()))
            agg[tag]["ss_max"].append(pe[ss].max())
            agg[tag]["force"].append(np.percentile(fe[full], 99.5))
        agg["ekf"]["cov2s"].append(((np.abs(xe - xt)[full]) <= 2*se[full]).mean()*100)
        if Ti == 50.0 and sd == 0:
            rep = (t, xt, xo, xe, se)

def stat(a): return float(np.median(a)), float(np.max(a))
summary = {}
for tag in ("obs", "ekf"):
    summary[tag] = {k: stat(agg[tag][k]) for k in ("max", "rms", "ss_max", "force")}
summary["ekf"]["cov2s_median"] = float(np.median(agg["ekf"]["cov2s"]))
json.dump(summary, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "codex-reviews", "ekf_benchmark.json"), "w"), indent=2)
print("metric            observer(med/max)     EKF(med/max)")
for k in ("max", "rms", "ss_max", "force"):
    o = summary["obs"][k]; e = summary["ekf"][k]
    print(f"  {k:8s}  {o[0]:6.2f}/{o[1]:6.2f}      {e[0]:6.2f}/{e[1]:6.2f}")
print(f"  EKF within-2sigma (median): {summary['ekf']['cov2s_median']:.0f}%")

# ---- figure: representative trajectory + EKF band, and error-vs-time ----
t, xt, xo, xe, se = rep
fig, ax = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True, constrained_layout=True)
ax[0].fill_between(t, (xe-2*se)*100, (xe+2*se)*100, color="#c77a1f", alpha=0.25,
                   label="EKF $\\pm2\\sigma$")
ax[0].plot(t, xt*100, "k", lw=1.6, label="true")
ax[0].plot(t, xo*100, "--", color="#1f6fb2", lw=1.1, label="complementary observer")
ax[0].plot(t, xe*100, ":", color="#c77a1f", lw=1.3, label="EKF")
ax[0].set_ylabel("position [cm]"); ax[0].legend(loc="upper right", fontsize=8.5, ncol=2)
ax[0].set_title("EKF vs complementary observer (T = 50 °C)"); ax[0].grid(alpha=0.3)
ax[1].plot(t, (xo-xt)*100, color="#1f6fb2", lw=1.0, label="observer error")
ax[1].plot(t, (xe-xt)*100, color="#c77a1f", lw=1.0, label="EKF error")
ax[1].axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1); ax[1].axhline(-POS_ERR_LIMIT*100, color="r", ls="--", lw=1)
ax[1].axvline(T_CONV, color="0.6", ls=":", lw=1); ax[1].text(T_CONV+0.3, -4.5, "post-convergence", fontsize=8, color="0.4")
ax[1].set_ylabel("position error [cm]"); ax[1].set_xlabel("time [s]")
ax[1].set_ylim(-6, 6); ax[1].legend(loc="upper right", fontsize=8.5); ax[1].grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "fig_ekf_compare.png"), dpi=140)
plt.close(fig)
print("wrote fig_ekf_compare.png")
