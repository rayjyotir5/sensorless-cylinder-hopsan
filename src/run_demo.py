#!/usr/bin/env python3
"""Sensorless double-acting hydraulic cylinder -- parametric offline demo.

Demonstrates that position can be estimated to <= +/- 5 cm and net force to
<= +/- 12 % of full scale across the full stroke and a 0-50 degC oil-
temperature envelope, from synthetic pressure / flow data with realistic
sensor noise, bias and quantization -- with NO position sensor.

Usage:
    python run_demo.py                 # full sweep + plots + summary
    python run_demo.py --quick         # fewer seeds (fast)
    python run_demo.py --no-plots      # skip figure generation
"""
from __future__ import annotations

import argparse
import os

from cylinder_sim.params import (Config, POS_ERR_LIMIT, FORCE_ERR_LIMIT_FRAC,
                                  TEMP_RANGE)
from cylinder_sim.sweep import run_single, run_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds")
    ap.add_argument("--no-plots", action="store_true", help="skip figures")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = Config()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 74)
    print(" Sensorless double-acting hydraulic cylinder -- offline estimation demo")
    print("=" * 74)
    print(f" Cylinder : bore {cfg.cyl.bore*1e3:.0f} mm  rod {cfg.cyl.rod*1e3:.0f} mm"
          f"  stroke {cfg.cyl.stroke*1e3:.0f} mm  mass {cfg.cyl.mass:.0f} kg")
    print(f" Supply   : {cfg.circ.supply_pressure/1e5:.0f} bar"
          f"   Force full-scale : {cfg.force_fullscale/1e3:.1f} kN")
    print(f" Targets  : position <= +/- {POS_ERR_LIMIT*100:.0f} cm,"
          f"  force <= +/- {FORCE_ERR_LIMIT_FRAC*100:.0f} % FS,"
          f"  T in [{TEMP_RANGE[0]:.0f}, {TEMP_RANGE[1]:.0f}] degC")
    print(f" Sensors  : P noise {cfg.sens.p_noise_frac*100:.1f}% FS + bias,"
          f"  Q noise {cfg.sens.q_noise_frac*100:.1f}% + {cfg.sens.q_bias_frac*100:.1f}% FS bias,"
          f"  T +/-{cfg.sens.T_noise:.0f} degC")
    print("=" * 74)

    # fail fast with an actionable message if Hopsan is not installed
    from cylinder_sim.hopsan_plant import find_hopsancli, HopsanNotFound
    try:
        cli = find_hopsancli()
        print(f" Hopsan   : {cli}")
    except HopsanNotFound as e:
        print("\n[ERROR] " + str(e))
        print("\n The plant is now modelled in Hopsan (hopsan/sensorless_cylinder_"
              "plant.hmf).\n Offline integration checks (no Hopsan needed):"
              "  python tests/test_hopsan_integration.py")
        raise SystemExit(2)

    seeds = list(range(3)) if args.quick else list(range(8))
    temps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    print(f"\nRunning sweep: {len(temps)} temperatures x {len(seeds)} seeds "
          f"= {len(temps)*len(seeds)} Monte-Carlo runs (HopsanCLI) ...\n")
    rows, summary, overall = run_sweep(cfg, temps=temps, seeds=seeds, verbose=True)

    # ---- summary table ----
    print("\n" + "-" * 74)
    print(f"{'T [degC]':>9} | {'pos max':>9} {'pos rms':>9} | "
          f"{'F max':>8} {'F rms':>8} | verdict")
    print(f"{'':>9} | {'[cm]':>9} {'[cm]':>9} | {'[%FS]':>8} {'[%FS]':>8} |")
    print("-" * 74)
    for T in temps:
        s = summary[T]
        ok = "PASS" if (s["pos_pass"] and s["force_pass"]) else "FAIL"
        print(f"{T:9.0f} | {s['max_pos_err']*100:9.2f} {s['rms_pos_err']*100:9.2f} | "
              f"{s['max_force_err_frac']*100:8.2f} {s['rms_force_err_frac']*100:8.2f} | {ok}")
    print("-" * 74)

    v = "PASS" if (overall["pos_pass"] and overall["force_pass"]) else "FAIL"
    print(f"\n OVERALL ({overall['n_runs']} runs): "
          f"worst position {overall['max_pos_err']*100:.2f} cm "
          f"(limit {POS_ERR_LIMIT*100:.0f}), "
          f"worst force {overall['max_force_err_frac']*100:.2f} %FS "
          f"(limit {FORCE_ERR_LIMIT_FRAC*100:.0f})  ->  {v}")

    # ---- plots ----
    if not args.no_plots:
        from cylinder_sim.plots import plot_trajectory, plot_sweep
        # representative single run at the hottest (worst) temperature
        rep = run_single(cfg, 50.0, seed=0)
        traj_path = os.path.join(args.outdir, "trajectory_50C.png")
        sweep_path = os.path.join(args.outdir, "sweep_summary.png")
        plot_trajectory(rep, cfg, traj_path)
        plot_sweep(summary, overall, cfg, sweep_path)
        print(f"\n Figures written to: {traj_path}\n"
              f"                     {sweep_path}")


if __name__ == "__main__":
    main()
