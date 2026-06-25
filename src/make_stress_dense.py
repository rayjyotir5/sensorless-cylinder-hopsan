"""Dense homing-interval stress sweep with confidence bands, comparing three
observer modes to settle which mechanism (homing reset vs bias calibration)
bounds the drift as the interval between stop-seatings grows.

For each (frequency, seed) the plant (with per-seed mismatch) is simulated once;
all three observer modes are then run on the same truth+measurements so the only
difference is the observer. interval = 1/(2*frequency)."""
import os, sys, json
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
cfg = Config()
T = 30.0
freqs = [0.045, 0.035, 0.027, 0.020, 0.015, 0.012]   # below flow saturation
seeds = list(range(8))
MODES = [
    ("full (reset + calibration)", {}, "#2e7d32", "o-"),
    ("reset only (no calibration)", {"use_bias_calib": False}, "#1f6fb2", "s--"),
    ("calibration only (no reset)", {"use_homing": False}, "#c77a1f", "^:"),
]


def run_observer(truth, meas, kw):
    obs = CylinderObserver(cfg, **kw)
    n = truth["t"].size
    xh = np.empty(n)
    for i in range(n):
        xh[i], _ = obs.step(meas["Q1"][i], meas["Q2"][i],
                            meas["P1"][i], meas["P2"][i], meas["T"][i])
    xt = np.clip(truth["x"], 0.0, cfg.cyl.stroke)
    m = truth["t"] >= 1.0
    return float(np.max(np.abs((xh - xt)[m]))) * 100.0


def main():
    # data[mode_name][interval] = list of per-seed max errors
    data = {m[0]: {} for m in MODES}
    intervals = []
    for f in freqs:
        interval = 1.0 / (2 * f)
        intervals.append(interval)
        dur = min(max(2.4 / f, 45.0), 185.0)
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)   # distinct from main sweep seeds
            mm = draw_mismatch(cfg, rng)
            truth = simulate_truth(cfg, T, freq=f, mm=mm, duration=dur)
            meas = make_measurements(truth, cfg, rng)
            for name, kw, _, _ in MODES:
                data[name].setdefault(interval, []).append(run_observer(truth, meas, kw))
        print(f"  interval={interval:5.1f}s dur={dur:5.0f}s  " +
              "  ".join(f"{m[0].split()[0]}={np.median(data[m[0]][interval]):.1f}" for m in MODES))

    # crossing intervals (median crosses 5 cm)
    summary = {}
    for name, kw, color, style in MODES:
        med = np.array([np.median(data[name][iv]) for iv in intervals])
        cross = None
        for j in range(len(intervals)-1):
            if med[j] <= POS_ERR_LIMIT*100 < med[j+1]:
                # linear interp
                x0, x1 = intervals[j], intervals[j+1]; y0, y1 = med[j], med[j+1]
                cross = x0 + (POS_ERR_LIMIT*100 - y0)*(x1-x0)/(y1-y0)
                break
        summary[name] = {"median": med.tolist(), "cross_s": cross}
        print(f"  {name:32s} crosses 5 cm at "
              + (f"{cross:.1f}s" if cross else "(not within range)"))
    raw = {name: {str(round(iv, 1)): data[name][iv] for iv in intervals} for name, *_ in MODES}
    json.dump({"intervals": intervals, "summary": summary, "raw": raw},
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "codex-reviews", "stress_dense.json"), "w"), indent=2)

    # ---- figure: median + 10-90 band per mode ----
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    iv = np.array(intervals)
    YMAX = 16.0
    for name, kw, color, style in MODES:
        arr = np.array([data[name][i] for i in intervals])   # (n_interval, n_seed)
        med = np.median(arr, axis=1)
        lo = np.percentile(arr, 25, axis=1); hi = np.percentile(arr, 75, axis=1)
        ax.fill_between(iv, lo, np.minimum(hi, YMAX), color=color, alpha=0.13)
        ax.plot(iv, med, style, color=color, lw=2.2, ms=6, label=name)
    ax.axhline(POS_ERR_LIMIT*100, color="r", ls="--", lw=1.6, label="±5 cm limit")
    ax.annotate("calibration-only diverges\n(no position reset)", xy=(33, 9),
                xytext=(22, 12.5), fontsize=8.5, color="#c77a1f",
                arrowprops=dict(arrowstyle="-|>", color="#c77a1f", lw=1.2))
    ax.set_xlabel("interval between end-stop homing events [s]")
    ax.set_ylabel("|position error| [cm]  (median, IQR band)")
    ax.set_title(f"Dense homing-interval stress test (T = {T:.0f} °C, "
                 f"{len(seeds)} seeds/point)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, YMAX)
    fig.savefig(os.path.join(FIG, "fig_stress_dense.png"), dpi=150)
    plt.close(fig)
    print("wrote fig_stress_dense.png")


if __name__ == "__main__":
    main()
