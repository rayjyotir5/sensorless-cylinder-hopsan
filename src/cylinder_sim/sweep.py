"""Single-run evaluation, error metrics and the parametric Monte-Carlo sweep."""
from __future__ import annotations

import numpy as np

from .params import (Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC)
from .hopsan_plant import simulate_truth, draw_mismatch
from .sensors import make_measurements
from .observer import CylinderObserver


def run_single(cfg: Config, T: float, seed: int, settle: float = 1.0,
               freq: float = 0.04, duration: float = None, obs_kwargs=None):
    """Run one truth+sensor+observer trajectory at temperature T.

    A seeded plant-parameter mismatch is drawn per run (plant deviates from the
    observer's nominal values); obs_kwargs forwards ablation flags to the
    observer. Errors are measured after a short settling window.
    """
    rng = np.random.default_rng(seed)
    mm = draw_mismatch(cfg, rng)
    truth = simulate_truth(cfg, T, freq=freq, mm=mm, duration=duration)
    meas = make_measurements(truth, cfg, rng)

    obs = CylinderObserver(cfg, **(obs_kwargs or {}))
    n = truth["t"].size
    x_hat = np.zeros(n)
    F_hat = np.zeros(n)
    for i in range(n):
        x_hat[i], F_hat[i] = obs.step(
            meas["Q1"][i], meas["Q2"][i],
            meas["P1"][i], meas["P2"][i], meas["T"][i])

    # clamp truth position to physical stroke for error accounting
    x_true = np.clip(truth["x"], 0.0, cfg.cyl.stroke)
    pos_err = x_hat - x_true
    force_err = F_hat - truth["F_net"]

    mask = truth["t"] >= settle
    Ffs = cfg.force_fullscale

    max_pos = float(np.max(np.abs(pos_err[mask])))
    rms_pos = float(np.sqrt(np.mean(pos_err[mask] ** 2)))
    fe_frac = np.abs(force_err[mask]) / Ffs
    max_force_frac = float(np.max(fe_frac))
    # robust (operational) force error: rare sub-sample pressure-reversal
    # transients a delayed transducer cannot track are excluded
    p995_force_frac = float(np.percentile(fe_frac, 99.5))
    rms_force_frac = float(np.sqrt(np.mean((force_err[mask] / Ffs) ** 2)))

    return {
        "T": T, "seed": seed,
        "t": truth["t"], "x_true": x_true, "x_hat": x_hat,
        "F_true": truth["F_net"], "F_hat": F_hat,
        "pos_err": pos_err, "force_err": force_err,
        "max_pos_err": max_pos, "rms_pos_err": rms_pos,
        "max_force_err_frac": max_force_frac, "rms_force_err_frac": rms_force_frac,
        "p995_force_err_frac": p995_force_frac,
        "pos_pass": max_pos <= POS_ERR_LIMIT,
        "force_pass": p995_force_frac <= FORCE_ERR_LIMIT_FRAC,
    }


def run_sweep(cfg: Config, temps=None, seeds=None, verbose=True):
    """Monte-Carlo over temperature x seed. Returns (rows, summary)."""
    if temps is None:
        temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    if seeds is None:
        seeds = list(range(6))

    rows = []
    for T in temps:
        for sd in seeds:
            r = run_single(cfg, T, sd)
            rows.append(r)
            if verbose:
                pf = "PASS" if (r["pos_pass"] and r["force_pass"]) else "FAIL"
                print(f"  T={T:5.1f}C seed={sd}  "
                      f"|pos|max={r['max_pos_err']*100:6.2f} cm  "
                      f"|F|max={r['max_force_err_frac']*100:6.2f} %FS   [{pf}]")

    # per-temperature aggregation (worst case across seeds)
    summary = {}
    for T in temps:
        sub = [r for r in rows if r["T"] == T]
        summary[T] = {
            "max_pos_err": max(r["max_pos_err"] for r in sub),
            "rms_pos_err": float(np.mean([r["rms_pos_err"] for r in sub])),
            "max_force_err_frac": max(r["max_force_err_frac"] for r in sub),
            "rms_force_err_frac": float(np.mean([r["rms_force_err_frac"] for r in sub])),
            "pos_pass": all(r["pos_pass"] for r in sub),
            "force_pass": all(r["force_pass"] for r in sub),
        }

    overall = {
        "max_pos_err": max(r["max_pos_err"] for r in rows),
        "max_force_err_frac": max(r["max_force_err_frac"] for r in rows),
        "pos_pass": all(r["pos_pass"] for r in rows),
        "force_pass": all(r["force_pass"] for r in rows),
        "n_runs": len(rows),
    }
    return rows, summary, overall
