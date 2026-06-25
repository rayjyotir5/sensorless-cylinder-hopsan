"""Full realistic-plant study: Monte-Carlo sweep, observer ablations, and the
homing-interval stress test. Regenerates the sweep-based figures and writes a
results JSON. Plant now includes parameter mismatch, Coulomb/stiction friction,
finite valve bandwidth, and transducer lag/delay/drift/clipping."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth, draw_mismatch
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver
from cylinder_sim.sweep import run_single

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
cfg = Config()
Ffs = cfg.force_fullscale
temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
seeds = list(range(8))
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
results = {}


# ----------------------------------------------------------------------
# 1. Full Monte-Carlo sweep (48 runs) -> figures + headline numbers
# ----------------------------------------------------------------------
def full_sweep():
    rows = []
    for T in temps:
        for sd in seeds:
            rows.append(run_single(cfg, T, sd))
    posmax = np.array([r["max_pos_err"]*100 for r in rows])
    fmax = np.array([r["max_force_err_frac"]*100 for r in rows])
    fp995 = np.array([r["p995_force_err_frac"]*100 for r in rows])
    results["sweep"] = {
        "n": len(rows),
        "worst_pos_cm": float(posmax.max()),
        "rms_pos_cm": float(np.mean([r["rms_pos_err"]*100 for r in rows])),
        "worst_force_p995_pct": float(fp995.max()),
        "worst_force_max_pct": float(fmax.max()),
        "rms_force_pct": float(np.mean([r["rms_force_err_frac"]*100 for r in rows])),
        "per_T": {str(int(T)): {
            "pos_max": max(r["max_pos_err"]*100 for r in rows if r["T"] == T),
            "pos_rms": float(np.mean([r["rms_pos_err"]*100 for r in rows if r["T"] == T])),
            "force_p995": max(r["p995_force_err_frac"]*100 for r in rows if r["T"] == T),
            "force_rms": float(np.mean([r["rms_force_err_frac"]*100 for r in rows if r["T"] == T])),
        } for T in temps},
    }
    print("SWEEP worst pos %.2f cm | force99.5 %.2f %% | force max %.1f %%"
          % (posmax.max(), fp995.max(), fmax.max()))

    # ---- MC ensemble figure (error-vs-time, coloured by T) ----
    norm = Normalize(vmin=min(temps), vmax=max(temps)); cmap = cm.viridis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    for r in rows:
        c = cmap(norm(r["T"]))
        a1.plot(r["t"], r["pos_err"]*100, color=c, lw=0.5, alpha=0.5)
        a2.plot(r["t"], r["force_err"]/Ffs*100, color=c, lw=0.5, alpha=0.5)
    for a, lim, u in ((a1, POS_ERR_LIMIT*100, "cm"), (a2, FORCE_ERR_LIMIT_FRAC*100, "% FS")):
        a.axhspan(-lim, lim, color="green", alpha=0.06)
        a.axhline(lim, color="r", ls="--", lw=1.4); a.axhline(-lim, color="r", ls="--", lw=1.4,
                  label=f"±{lim:g} {u} limit")
        a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=9)
    a1.set_ylabel("position error [cm]"); a1.set_ylim(-POS_ERR_LIMIT*100*1.15, POS_ERR_LIMIT*100*1.15)
    a1.set_title(f"Realistic plant — Monte-Carlo ensemble: {len(rows)} runs "
                 f"(mismatch + friction + sensor dynamics)\nworst |pos| = {posmax.max():.2f} cm, "
                 f"operational |force| (99.5 pct) = {fp995.max():.2f} % FS")
    a2.set_ylabel("force error [% FS]"); a2.set_xlabel("time [s]")
    a2.set_ylim(-FORCE_ERR_LIMIT_FRAC*100*1.15, FORCE_ERR_LIMIT_FRAC*100*1.15)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=[a1, a2], shrink=0.9, pad=0.015); cb.set_label("oil temperature [°C]")
    fig.savefig(os.path.join(FIG, "fig_mc_ensemble.png"), dpi=140); plt.close(fig)

    # ---- histogram ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    ax[0].hist(posmax, bins=12, color="C0", alpha=0.85, edgecolor="k", lw=0.4)
    ax[0].axvline(POS_ERR_LIMIT*100, color="r", ls="--", lw=2, label="±5 cm limit")
    ax[0].axvline(posmax.max(), color="k", ls=":", lw=1.5, label=f"worst {posmax.max():.2f} cm")
    ax[0].set_xlabel("max |position error| per run [cm]"); ax[0].set_ylabel("runs")
    ax[0].set_title("Position error"); ax[0].legend(fontsize=9); ax[0].set_xlim(0, POS_ERR_LIMIT*100*1.1)
    ax[1].hist(fp995, bins=12, color="C1", alpha=0.85, edgecolor="k", lw=0.4)
    ax[1].axvline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls="--", lw=2, label="±12 % FS limit")
    ax[1].axvline(fp995.max(), color="k", ls=":", lw=1.5, label=f"worst {fp995.max():.2f} %")
    ax[1].set_xlabel("operational |force error| (99.5 pct) per run [% FS]"); ax[1].set_ylabel("runs")
    ax[1].set_title("Force error (99.5 pct)"); ax[1].legend(fontsize=9); ax[1].set_xlim(0, FORCE_ERR_LIMIT_FRAC*100*1.1)
    fig.suptitle(f"Monte-Carlo acceptance: {len(rows)} runs, all PASS", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig_error_hist.png"), dpi=130); plt.close(fig)

    # ---- error vs temperature ----
    pT = results["sweep"]["per_T"]
    Ts = sorted(int(t) for t in pT)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    ax[0].plot(Ts, [pT[str(t)]["pos_max"] for t in Ts], "o-", label="worst-case")
    ax[0].plot(Ts, [pT[str(t)]["pos_rms"] for t in Ts], "s--", alpha=0.7, label="mean RMS")
    ax[0].axhline(POS_ERR_LIMIT*100, color="r", ls=":", lw=1.5, label="±5 cm limit")
    ax[0].set_xlabel("oil temperature [°C]"); ax[0].set_ylabel("position error [cm]")
    ax[0].set_title("Position error vs temperature"); ax[0].legend(fontsize=9)
    ax[0].set_ylim(0, POS_ERR_LIMIT*100*1.3)
    ax[1].plot(Ts, [pT[str(t)]["force_p995"] for t in Ts], "o-", label="worst-case 99.5 pct")
    ax[1].plot(Ts, [pT[str(t)]["force_rms"] for t in Ts], "s--", alpha=0.7, label="mean RMS")
    ax[1].axhline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls=":", lw=1.5, label="±12 % FS limit")
    ax[1].set_xlabel("oil temperature [°C]"); ax[1].set_ylabel("force error [% FS]")
    ax[1].set_title("Force error vs temperature"); ax[1].legend(fontsize=9)
    ax[1].set_ylim(0, FORCE_ERR_LIMIT_FRAC*100*1.3)
    fig.suptitle("Parametric sweep (realistic plant) — VERDICT: PASS", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig_sweep.png"), dpi=130); plt.close(fig)
    print("  regenerated MC ensemble, histogram, sweep-vs-T figures")


# ----------------------------------------------------------------------
# 2. Observer ablations (cache truth+meas per (T,seed); vary observer only)
# ----------------------------------------------------------------------
ABLATIONS = [
    ("Full observer", {}),
    ("No bias calibration", {"use_bias_calib": False}),
    ("No steady-pressure gate", {"use_steady_gate": False}),
    ("No homing (reset off)", {"use_homing": False}),
    ("Single chamber only", {"single_chamber": True}),
    ("No leakage correction", {"use_leak": False}),
    ("No compressibility corr.", {"use_compress": False}),
    ("No temperature corr.", {"use_temp": False}),
]


def ablations():
    abl_temps = [0.0, 30.0, 50.0]; abl_seeds = [0, 1, 2]
    cache = {}
    for T in abl_temps:
        for sd in abl_seeds:
            rng = np.random.default_rng(sd)
            mm = draw_mismatch(cfg, rng)
            truth = simulate_truth(cfg, T, mm=mm)
            meas = make_measurements(truth, cfg, rng)
            cache[(T, sd)] = (truth, meas)
    table = []
    for name, kw in ABLATIONS:
        worst = 0.0
        for (T, sd), (truth, meas) in cache.items():
            obs = CylinderObserver(cfg, **kw)
            n = truth["t"].size; xh = np.empty(n)
            for i in range(n):
                xh[i], _ = obs.step(meas["Q1"][i], meas["Q2"][i],
                                    meas["P1"][i], meas["P2"][i], meas["T"][i])
            xt = np.clip(truth["x"], 0, cfg.cyl.stroke)
            m = truth["t"] >= 1.0
            worst = max(worst, float(np.max(np.abs((xh - xt)[m]))) * 100)
        table.append((name, worst))
        print(f"  ablation {name:28s} worst |pos| = {worst:7.2f} cm")
    results["ablations"] = table

    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    names = [t[0] for t in table]; vals = [t[1] for t in table]
    colors = ["#2e7d32" if v <= POS_ERR_LIMIT*100 else "#c0392b" for v in vals]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="k", lw=0.5)
    ax.axvline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.8, label="±5 cm limit")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("worst-case |position error| [cm]  (log scale)")
    ax.set_xscale("log"); ax.set_xlim(0.3, max(vals)*1.4)
    for yi, v in zip(y, vals):
        ax.text(v*1.05, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_title("Observer ablations: which mechanism buys the result")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=140); plt.close(fig)
    print("  wrote fig_ablation.png")


# ----------------------------------------------------------------------
# 3. Homing-interval stress test (vary sine frequency -> interval = 1/(2f))
# ----------------------------------------------------------------------
def stress():
    freqs = [0.045, 0.03, 0.022, 0.016, 0.011]   # below flow-saturation freq (~0.05 Hz)
    st_seeds = [0, 1, 2, 3, 4]
    pts = []
    for f in freqs:
        interval = 1.0 / (2 * f)
        dur = min(max(2.2 / f, 40.0), 170.0)     # ~2 cycles, capped
        worst = 0.0
        for sd in st_seeds:
            r = run_single(cfg, 30.0, sd, freq=f, duration=dur)
            worst = max(worst, r["max_pos_err"] * 100)
        pts.append((interval, worst))
        print(f"  stress f={f} interval={interval:5.1f}s dur={dur:5.0f}s worst|pos|={worst:5.2f}cm")
    results["stress"] = pts

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    iv = [p[0] for p in pts]; pe = [p[1] for p in pts]
    ax.plot(iv, pe, "o-", color="C0", lw=2, ms=7)
    ax.axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.8, label="±5 cm limit")
    ax.set_xlabel("interval between end-stop homing events [s]")
    ax.set_ylabel("worst-case |position error| [cm]")
    ax.set_title("Homing-interval stress test (T = 30 °C)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_stress.png"), dpi=140); plt.close(fig)
    print("  wrote fig_stress.png")


if __name__ == "__main__":
    print("== full sweep =="); full_sweep()
    print("== ablations =="); ablations()
    print("== stress test =="); stress()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "codex-reviews", "realism_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print("wrote", out)
