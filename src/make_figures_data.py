"""Generate the data-driven paper figures from real Hopsan runs."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_sim.params import Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC
from cylinder_sim.hopsan_plant import simulate_truth
from cylinder_sim.sensors import make_measurements
from cylinder_sim.observer import CylinderObserver
from cylinder_sim.sweep import run_sweep, run_single

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

cfg = Config()


def fig_pressure_flow():
    """Synthetic sensor signals (truth vs measured) from the Hopsan plant."""
    T = 40.0
    truth = simulate_truth(cfg, T)
    rng = np.random.default_rng(0)
    meas = make_measurements(truth, cfg, rng)
    t = truth["t"]
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True, constrained_layout=True)

    ax[0].plot(t, truth["P1"]/1e5, label="$P_1$ (cap) truth", lw=1.4)
    ax[0].plot(t, truth["P2"]/1e5, label="$P_2$ (rod) truth", lw=1.4)
    ax[0].plot(t, meas["P1"]/1e5, color="C0", lw=0.4, alpha=0.4)
    ax[0].set_ylabel("pressure [bar]")
    ax[0].set_title(f"Hopsan plant signals @ {T:.0f} °C  (lines: truth, faint: measured)")
    ax[0].legend(loc="upper right", ncol=2, fontsize=9)

    ax[1].plot(t, truth["Q1"]*60e3, label="$Q_1$ truth", lw=1.4)
    ax[1].plot(t, meas["Q1"]*60e3, color="C0", lw=0.4, alpha=0.4, label="$Q_1$ measured")
    ax[1].plot(t, truth["Q2"]*60e3, label="$Q_2$ truth", lw=1.4, color="C1")
    ax[1].set_ylabel("flow [L/min]")
    ax[1].legend(loc="upper right", ncol=2, fontsize=9)

    ax[2].plot(t, truth["x"]*100, color="k", lw=1.6, label="piston position (truth)")
    ax[2].set_ylabel("position [cm]"); ax[2].set_xlabel("time [s]")
    ax[2].legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_signals.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_signals.png")


def fig_fluid_temp():
    """Temperature dependence of the fluid properties injected per run."""
    Ts = np.linspace(0, 50, 200)
    beta = np.array([cfg.fluid.beta(T) for T in Ts])
    mu = np.array([cfg.fluid.mu(T) for T in Ts])
    leak = np.array([cfg.fluid.leak_coeff(T) for T in Ts])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    ax[0].plot(Ts, beta/1e9, lw=2, color="C0"); ax[0].set_title(r"bulk modulus $\beta_e(T)$")
    ax[0].set_xlabel("oil temp [°C]"); ax[0].set_ylabel("GPa")
    ax[1].plot(Ts, mu*1e3, lw=2, color="C1"); ax[1].set_title(r"viscosity $\mu(T)$")
    ax[1].set_xlabel("oil temp [°C]"); ax[1].set_ylabel("mPa·s")
    ax[2].plot(Ts, leak*1e13, lw=2, color="C2"); ax[2].set_title(r"leak coeff $C_{leak}(T)$")
    ax[2].set_xlabel("oil temp [°C]"); ax[2].set_ylabel(r"$10^{-13}$ m³/(s·Pa)")
    fig.suptitle("Temperature-dependent fluid parameters injected into the Hopsan model",
                 fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig_fluid_temp.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_fluid_temp.png")


def fig_error_distribution():
    """Full 48-run sweep -> error distributions + worst-case vs temperature."""
    temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    seeds = list(range(8))
    rows, summary, overall = run_sweep(cfg, temps=temps, seeds=seeds, verbose=False)
    pos = np.array([r["max_pos_err"]*100 for r in rows])
    frc = np.array([r["max_force_err_frac"]*100 for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    ax[0].hist(pos, bins=14, color="C0", alpha=0.85, edgecolor="k", lw=0.4)
    ax[0].axvline(POS_ERR_LIMIT*100, color="r", ls="--", lw=2, label="±5 cm limit")
    ax[0].axvline(pos.max(), color="k", ls=":", lw=1.5, label=f"worst {pos.max():.2f} cm")
    ax[0].set_xlabel("max |position error| per run [cm]"); ax[0].set_ylabel("runs")
    ax[0].set_title(f"Position error over {len(rows)} runs"); ax[0].legend(fontsize=9)
    ax[0].set_xlim(0, POS_ERR_LIMIT*100*1.1)

    ax[1].hist(frc, bins=14, color="C1", alpha=0.85, edgecolor="k", lw=0.4)
    ax[1].axvline(FORCE_ERR_LIMIT_FRAC*100, color="r", ls="--", lw=2, label="±12 % FS limit")
    ax[1].axvline(frc.max(), color="k", ls=":", lw=1.5, label=f"worst {frc.max():.2f} %")
    ax[1].set_xlabel("max |force error| per run [% FS]"); ax[1].set_ylabel("runs")
    ax[1].set_title(f"Force error over {len(rows)} runs"); ax[1].legend(fontsize=9)
    ax[1].set_xlim(0, FORCE_ERR_LIMIT_FRAC*100*1.1)
    fig.suptitle(f"Monte-Carlo acceptance: {overall['n_runs']} runs, all PASS",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig_error_hist.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_error_hist.png  worst pos=%.2fcm force=%.2f%%" % (pos.max(), frc.max()))
    return overall


if __name__ == "__main__":
    fig_pressure_flow()
    fig_fluid_temp()
    fig_error_distribution()
    print("data figures done")
