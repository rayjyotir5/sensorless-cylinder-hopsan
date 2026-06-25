# Sensorless position & force estimation for a double-acting hydraulic cylinder

Recover piston **position** and net **force** for a double-acting hydraulic
cylinder from synthetic **pressure** and **flow** signals with realistic noise,
bias and parameter mismatch — **no position sensor**. The **plant (truth model) is
[Hopsan](https://github.com/Hopsan/hopsan)**, a transmission-line (TLM)
fluid-power simulator; the estimator, sensor model, Monte-Carlo sweep and
acceptance check are Python.

This repository accompanies the paper *Sensorless Position and Force Estimation
for a Double-Acting Hydraulic Cylinder Using Pressure and Flow Signals*
([`paper/Sensorless_Hydraulic_Cylinder_Hopsan.pdf`](paper/Sensorless_Hydraulic_Cylinder_Hopsan.pdf)).

**Headline result** — 48-run Monte-Carlo sweep (6 temperatures × 8 seeds, real
HopsanCLI simulation, per-run plant/estimator parameter mismatch):

| Quantity | Target | Worst case | Verdict |
|----------|--------|-----------|---------|
| Position error | ≤ **±5 cm** | **2.58 cm** | ✅ PASS |
| Net force error (99.5 pct) | ≤ **±12 % FS** | **5.08 % FS** | ✅ PASS |
| Oil temperature | 0–50 °C | full envelope | ✅ |

Ablations identify the **end-stop homing reset** as the load-bearing mechanism.
The harness also includes a well-scaled EKF benchmark, supply-pressure / bore /
hose-length sweeps, an open-loop bench-scale variant, and a cross-check against a
published physical experiment (Zhou et al., *Machines* 2022).

## Layout

```
src/
  cylinder_sim/        estimator + plant driver (Python package)
    hopsan_plant.py    HopsanCLI driver: per-run param map + results parser
    params.py          cylinder / fluid / sensor params + temperature curves
    sensors.py         synthetic noisy P / Q / T sensor model
    observer.py        sensorless complementary observer (position + force)
    ekf.py             well-scaled (non-dimensionalised) Extended Kalman Filter
    sweep.py           Monte-Carlo sweep + metrics + pass/fail
    plots.py           figures
  hopsan/
    sensorless_cylinder_plant.hmf           closed-loop plant (with manifold hoses)
    sensorless_cylinder_plant_openloop.hmf  open-loop (bench) variant
    build_hopsancli_macos.sh                reproducible head-less HopsanCLI build
  run_demo.py          entry point (sweep + plots + verdict)
  make_*.py            figure-generation scripts (one per paper figure)
  tests/               offline integration checks (no Hopsan needed)
figures/               generated figures used by the paper
paper/                 IEEE LaTeX source + compiled PDF
```

## Requirements

- **HopsanCLI** on `PATH` (or via `$HOPSANCLI`). No official macOS package, so it
  is built from source (HopsanCore + HopsanCLI are plain C++, no Qt):
  ```bash
  ./src/hopsan/build_hopsancli_macos.sh
  ```
  On Windows/Linux, install a release from
  <https://github.com/Hopsan/hopsan/releases> and put `HopsanCLI` on `PATH`. The
  driver auto-discovers the default component library next to the binary, or via
  `$HOPSAN_COMPONENTLIB`.
- Python: `numpy`, `scipy`, `matplotlib`.

## Run

```bash
python src/run_demo.py                 # full Monte-Carlo sweep + plots + verdict
python src/run_demo.py --quick         # fewer seeds
python src/tests/test_hopsan_integration.py   # offline checks (no Hopsan needed)

# regenerate individual paper figures, e.g.
python src/make_realism_results.py     # headline sweep + ablations + stress
python src/make_hose_sweep.py          # manifold-sensing hose-length sweep
python src/make_zhou_validation.py     # validation vs. Zhou et al. (2022)
```

## How the estimator works

- **Position** from flow-continuity velocity (both chambers, SNR-weighted),
  corrected for flow-meter bias, cross-piston leakage and fluid compressibility,
  then integrated. The integration-killing flow-meter **bias is calibrated at
  end-stop seating** (true velocity = 0, gated on steady pressure), and absolute
  position is pinned by **homing** at the stops — the mechanism that bounds drift.
- **Force** from the (bias-calibrated) pressure transducers, `F = P₁A₁ − P₂A₂`.

The pressure/flow sensors are modelled at the **valve manifold**, with hoses
carrying oil to the cylinder, so the estimator sees the line loss `Δp` upstream of
the cylinder — quantified by the hose-length sweep.

## The Hopsan model

`src/hopsan/sensorless_cylinder_plant.hmf` is derived from Hopsan's Apache-2.0
"Position Servo" example: a `HydraulicCylinderC` double-acting cylinder, a
`Hydraulic43Valve` 4/3 proportional valve, pump + relief supply, a translational
mass with hard end-stops, manifold-to-cylinder hoses (volume + laminar orifice),
and a PI position controller driven by a full-stroke sine so both stops seat each
half-cycle. Hopsan holds bulk modulus / leakage / friction constant, so
`hopsan_plant.parameter_rows(cfg, T)` re-parameterises them per run from the
`β(T)`, `μ(T)`, `leak(T)` curves in `params.py` (one steady temperature per run).

## License

Code is provided for research use. The Hopsan model is derived from an Apache-2.0
example. The referenced experimental paper (Zhou et al.) and any third-party lab
models are **not** redistributed here.
