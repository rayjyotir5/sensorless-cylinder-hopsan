"""Plotting helpers for the sensorless-cylinder demo."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .params import Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC


def plot_trajectory(run, cfg: Config, path):
    """Time-series of true vs estimated position & force + error traces."""
    t = run["t"]
    Ffs = cfg.force_fullscale
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)

    ax[0, 0].plot(t, run["x_true"] * 100, label="true", lw=1.6)
    ax[0, 0].plot(t, run["x_hat"] * 100, "--", label="estimate", lw=1.2)
    ax[0, 0].set_ylabel("position [cm]")
    ax[0, 0].set_title(f"Position  (T={run['T']:.0f} °C, seed={run['seed']})")
    ax[0, 0].legend(loc="best"); ax[0, 0].grid(alpha=0.3)

    ax[1, 0].plot(t, run["pos_err"] * 100, color="C3", lw=1.0)
    ax[1, 0].axhline(POS_ERR_LIMIT * 100, color="k", ls=":", lw=1)
    ax[1, 0].axhline(-POS_ERR_LIMIT * 100, color="k", ls=":", lw=1)
    ax[1, 0].set_ylabel("position error [cm]"); ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylim(-1.4 * POS_ERR_LIMIT * 100, 1.4 * POS_ERR_LIMIT * 100)
    ax[1, 0].grid(alpha=0.3)

    ax[0, 1].plot(t, run["F_true"] / 1e3, label="true", lw=1.6)
    ax[0, 1].plot(t, run["F_hat"] / 1e3, "--", label="estimate", lw=1.0)
    ax[0, 1].set_ylabel("net force [kN]")
    ax[0, 1].set_title("Force"); ax[0, 1].legend(loc="best"); ax[0, 1].grid(alpha=0.3)

    ax[1, 1].plot(t, run["force_err"] / Ffs * 100, color="C3", lw=1.0)
    ax[1, 1].axhline(FORCE_ERR_LIMIT_FRAC * 100, color="k", ls=":", lw=1)
    ax[1, 1].axhline(-FORCE_ERR_LIMIT_FRAC * 100, color="k", ls=":", lw=1)
    ax[1, 1].set_ylabel("force error [% FS]"); ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylim(-1.4 * FORCE_ERR_LIMIT_FRAC * 100, 1.4 * FORCE_ERR_LIMIT_FRAC * 100)
    ax[1, 1].grid(alpha=0.3)

    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(summary, overall, cfg: Config, path):
    """Worst-case error vs temperature against the acceptance limits."""
    temps = sorted(summary)
    max_pos = [summary[T]["max_pos_err"] * 100 for T in temps]
    rms_pos = [summary[T]["rms_pos_err"] * 100 for T in temps]
    max_f = [summary[T]["max_force_err_frac"] * 100 for T in temps]
    rms_f = [summary[T]["rms_force_err_frac"] * 100 for T in temps]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax[0].plot(temps, max_pos, "o-", label="worst-case |error|")
    ax[0].plot(temps, rms_pos, "s--", label="mean RMS", alpha=0.7)
    ax[0].axhline(POS_ERR_LIMIT * 100, color="r", ls=":", lw=1.5,
                  label=f"limit ±{POS_ERR_LIMIT*100:.0f} cm")
    ax[0].set_xlabel("oil temperature [°C]"); ax[0].set_ylabel("position error [cm]")
    ax[0].set_title("Position error vs temperature")
    ax[0].set_ylim(0, POS_ERR_LIMIT * 100 * 1.3)
    ax[0].legend(loc="best"); ax[0].grid(alpha=0.3)

    ax[1].plot(temps, max_f, "o-", label="worst-case |error|")
    ax[1].plot(temps, rms_f, "s--", label="mean RMS", alpha=0.7)
    ax[1].axhline(FORCE_ERR_LIMIT_FRAC * 100, color="r", ls=":", lw=1.5,
                  label=f"limit ±{FORCE_ERR_LIMIT_FRAC*100:.0f} % FS")
    ax[1].set_xlabel("oil temperature [°C]"); ax[1].set_ylabel("force error [% FS]")
    ax[1].set_title("Force error vs temperature")
    ax[1].set_ylim(0, FORCE_ERR_LIMIT_FRAC * 100 * 1.3)
    ax[1].legend(loc="best"); ax[1].grid(alpha=0.3)

    verdict = ("PASS" if (overall["pos_pass"] and overall["force_pass"]) else "FAIL")
    fig.suptitle(f"Parametric sweep — {overall['n_runs']} runs — VERDICT: {verdict}",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=120)
    plt.close(fig)
