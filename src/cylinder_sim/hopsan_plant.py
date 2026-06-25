"""Hopsan-backed truth plant.

This replaces the former hand-written Python RK4 plant. The double-acting
cylinder, 4/3 servo valve, pump, relief valve, mass/load and PI position
controller are all modelled in Hopsan (``hopsan/sensorless_cylinder_plant.hmf``,
based on the Apache-2.0 "Position Servo" example). We drive it head-less with
HopsanCLI, overriding the temperature-dependent fluid parameters per run, and
parse the exported results CSV back into the log-dict that ``sensors.py`` /
``observer.py`` / ``sweep.py`` already consume.

Hopsan models bulk modulus / leakage / friction as *constant* parameters (it
has no built-in temperature dependence), so the 0-50 degC envelope is covered
by re-parameterising the model per run from the same beta(T)/mu(T)/leak(T)
curves in ``params.py`` -- each sweep run is at one steady oil temperature.

Excitation: the reference is a SignalSineWave with amplitude 1.2 x stroke, so
the servo drives the piston across the full stroke and seats *both* end-stops
every half cycle -- giving the observer its periodic homing / bias-calibration
opportunities, in a single simulation run.

Requires HopsanCLI on PATH or in $HOPSANCLI. See README for install.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .params import Config

# Component + parameter names as they appear in the .hmf (verified against the
# vendored model). Addressed to HopsanCLI as "Component#Parameter".
MODEL_REL = os.path.join("hopsan", "sensorless_cylinder_plant.hmf")

# Result-CSV column full-names (Component#Port#DataName).
# In Hopsan's TLM the two ports sharing a node log on one side: the cap/rod
# chamber pressures & flows are reported on the valve side (Valve PA<->Cyl P1
# cap A_1, Valve PB<->Cyl P2 rod A_2). Pressure is identical either side;
# port flow is positive *into the valve*, i.e. the negative of flow into the
# chamber, so Q1/Q2 are sign-flipped below.
COL_T = "Time"
COL_P1 = "4_3_Servo_Valve#PA#Pressure"
COL_P2 = "4_3_Servo_Valve#PB#Pressure"
COL_Q1 = "4_3_Servo_Valve#PA#Flow"
COL_Q2 = "4_3_Servo_Valve#PB#Flow"
COL_Q_SIGN = 1.0    # valve PA/PB flow already = flow into the chamber
# The manifold-to-cylinder hoses put the sensors (valve PA/PB) upstream of a
# real line loss, so the *true* cylinder chamber pressures differ from what is
# measured. The cylinder-side pressure is the orifice downstream node
# (Line_Orif_*#P2 shares the cylinder hydraulic node). Used for ground-truth
# force; absent in any model without the line elements, in which case we fall
# back to the manifold pressures (zero line loss).
COL_P1_CYL = "Line_Orif_A#P2#Pressure"
COL_P2_CYL = "Line_Orif_B#P2#Pressure"
COL_X_CANDIDATES = (
    "Position_Sensor#out#Value",
    "Mass#P1#Position",
    "Cylinder_C#P3#Position",
)


class HopsanNotFound(RuntimeError):
    pass


def find_hopsancli() -> str:
    """Locate the HopsanCLI executable."""
    env = os.environ.get("HOPSANCLI")
    if env and os.path.exists(env):
        return env
    for name in ("HopsanCLI", "hopsancli", "HopsanCLI.exe"):
        p = shutil.which(name)
        if p:
            return p
    raise HopsanNotFound(
        "HopsanCLI not found. Install Hopsan (https://github.com/Hopsan/hopsan/"
        "releases) and either put HopsanCLI on PATH or set $HOPSANCLI to its "
        "full path.")


MODEL_REL_OL = os.path.join("hopsan", "sensorless_cylinder_plant_openloop.hmf")


def model_path(open_loop: bool = False) -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, MODEL_REL_OL if open_loop else MODEL_REL)


def find_component_lib():
    """Locate the default component library shared object, if needed.

    A full Hopsan install auto-loads its components, but a CLI-only source
    build keeps them in libdefaultcomponentlibrary.{dylib,so,dll} that must be
    loaded with -e. Checked: $HOPSAN_COMPONENTLIB, then near the CLI binary.
    Returns a path or None (None => assume components are built in / auto-loaded).
    """
    env = os.environ.get("HOPSAN_COMPONENTLIB")
    if env and os.path.exists(env):
        return env
    names = ("libdefaultcomponentlibrary.dylib",
             "libdefaultcomponentlibrary.so",
             "defaultcomponentlibrary.dll")
    try:
        cli_dir = os.path.dirname(os.path.realpath(find_hopsancli()))
    except HopsanNotFound:
        return None
    roots = [
        os.path.join(cli_dir, "..", "componentLibraries", "defaultLibrary"),
        os.path.join(cli_dir, "..", "lib"),
        cli_dir,
    ]
    for root in roots:
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                return os.path.realpath(p)
    return None


def draw_mismatch(cfg: Config, rng):
    """Draw per-run multiplicative plant-parameter perturbations (lognormal-ish),
    so the Hopsan plant deviates from the observer's nominal values. Returns a
    dict of multipliers; identity dict if mismatch disabled or rng is None."""
    m = cfg.mismatch
    if rng is None or not m.enabled:
        return {k: 1.0 for k in ("beta", "leak", "visc", "area", "deadvol")}

    def mult(sig):
        return float(np.exp(rng.normal(0.0, sig)))   # positive multiplier
    return {"beta": mult(m.beta), "leak": mult(m.leak), "visc": mult(m.visc),
            "area": mult(m.area), "deadvol": mult(m.deadvol)}


def parameter_rows(cfg: Config, T: float, freq: float = 0.04, mm=None, p_rep=None,
                   gain_scale: float = 1.0, xvmax: float = None,
                   open_loop: bool = False, spool_amp: float = 0.7,
                   hose_len: float = 2.0, hose_dia: float = 0.019):
    """Build (name, value) parameter overrides for one temperature.

    Temperature-dependent fluid props follow the curves in params.py; mm is a
    multiplier dict from draw_mismatch() applied to the *plant* only. If p_rep
    (a representative operating pressure) is given, the bulk modulus uses the
    entrained-air model beta_eff(T, p_rep) instead of beta(T).
    """
    cyl, fl, fr, circ = cfg.cyl, cfg.fluid, cfg.fric, cfg.circ
    mu_ratio = fl.mu(T) / fl.mu_ref
    if mm is None:
        mm = {k: 1.0 for k in ("beta", "leak", "visc", "area", "deadvol")}
    beta_plant = fl.beta(T) if p_rep is None else fl.beta_eff(T, p_rep)

    rows = [
        # ---- cylinder geometry (+/- area & dead-volume mismatch) ----
        ("Cylinder_C#A_1#Value", cyl.area_cap * mm["area"]),
        ("Cylinder_C#A_2#Value", cyl.area_rod * mm["area"]),
        ("Cylinder_C#s_l#Value", cyl.stroke),
        ("Cylinder_C#V_1#Value", cyl.dead_vol_cap * mm["deadvol"]),
        ("Cylinder_C#V_2#Value", cyl.dead_vol_rod * mm["deadvol"]),
        # ---- cylinder fluid props (temperature dependent + mismatch) ----
        ("Cylinder_C#Beta_e#Value", beta_plant * mm["beta"]),
        ("Cylinder_C#c_leak#Value", fl.leak_coeff(T) * mm["leak"]),
        ("Cylinder_C#B_p#Value", fr.viscous_ref * mu_ratio * mm["visc"]),
        # ---- moving mass with Coulomb (stiction>kinetic) + end-stops ----
        ("Mass#m", cyl.mass),
        ("Mass#b#Value", 0.0),
        ("Mass#f_s#Value", fr.coulomb * 1.5),    # static (stiction)
        ("Mass#f_k#Value", fr.coulomb),          # kinetic
        ("Mass#x_min#Value", 0.0),
        ("Mass#x_max#Value", cyl.stroke),
        # ---- supply: relief sets system pressure ----
        ("Pressure_Relief_Valve#p_max#Value", circ.supply_pressure),
        # ---- valve: density + finite spool bandwidth (2nd-order spool) ----
        ("4_3_Servo_Valve#rho#Value", 870.0),
        ("4_3_Servo_Valve#omega_h", 120.0),        # spool natural freq [rad/s]
        # ---- reference sine. Closed-loop: amplitude > stroke (position ref) so
        #      the servo sweeps the full stroke and seats both stops. Open-loop:
        #      the sine drives the valve spool directly, amplitude = spool_amp
        #      fraction of x_vmax (in metres). ----
        ("Step#f#Value", freq),
        ("Step#y_A#Value", (spool_amp * (xvmax if xvmax is not None else 0.01))
                           if open_loop else 1.1 * cyl.stroke),
        ("Step#y_offset#Value", 0.0),
        ("Step#t_start#Value", 0.0),
        # ---- PI position controller (spool command in metres) ----
        ("GainP#k#Value", 0.03 * gain_scale),
        ("GainI#k#Value", 0.004 * gain_scale),
    ]
    if xvmax is not None:
        rows.append(("4_3_Servo_Valve#x_vmax#Value", xvmax))

    # ---- manifold-to-cylinder hoses (sensors sit at the valve = manifold) ----
    # Only the industrial/closed-loop model carries the hose elements; the
    # benchtop (open-loop) rig has its sensors at the cylinder (no hose run), so
    # its model is line-free and these rows are skipped. Each line = a volume
    # (compliance + the manifold node) then a laminar orifice (friction loss).
    # Hagen-Poiseuille gives Q = Kc * dP with Kc = pi d^4 / (128 mu L), so the
    # orifice is temperature dependent through mu(T); the volume is A_hose * L.
    if not open_loop:
        L = max(hose_len, 1e-4)
        d = hose_dia
        mu = fl.mu(T)
        Kc_line = math.pi * d ** 4 / (128.0 * mu * L)
        V_line = math.pi / 4.0 * d ** 2 * L
        rows += [
            ("Line_Orif_A#Kc#Value", Kc_line),
            ("Line_Orif_B#Kc#Value", Kc_line),
            ("Line_Vol_A#V#Value", V_line),
            ("Line_Vol_B#V#Value", V_line),
            ("Line_Vol_A#Beta_e#Value", beta_plant),
            ("Line_Vol_B#Beta_e#Value", beta_plant),
        ]
    return rows


def _write_param_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for name, val in rows:
            w.writerow([name, repr(float(val))])


def _read_results_csv(path):
    """Read a --resultsCSVSort cols CSV into {colname: np.array}.

    Hopsan writes two header rows in cols layout: variable names, then a units
    row (e.g. 's', 'm', 'Pa'). The units row is detected and skipped.
    """
    def _isfloat(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    with open(path, newline="") as f:
        raw = list(csv.reader(f))
    # drop fully-empty rows (Hopsan writes a blank row + a units row after the
    # header in cols layout)
    raw = [r for r in raw if any(c.strip() for c in r)]
    header = raw[0]
    # data rows = those whose non-empty cells are all numeric (skips units row)
    data = [r for r in raw[1:]
            if all(_isfloat(c) for c in r if c.strip())]

    cols = {h: [] for h in header}
    for row in data:
        for h, v in zip(header, row):
            cols[h].append(v if v.strip() else "nan")
    return {h: np.array(v, dtype=float) for h, v in cols.items()}


def _pick(cols, name):
    if name in cols:
        return cols[name]
    # tolerate a root-system prefix (e.g. "Position_Servo$Cylinder_C#...")
    for k in cols:
        if k.endswith(name) or k.split("$")[-1] == name:
            return cols[k]
    return None


def simulate_truth(cfg: Config, T: float, freq: float = 0.04,
                   mm=None, duration: float = None, p_rep=None,
                   gain_scale: float = 1.0, xvmax: float = None,
                   open_loop: bool = False, spool_amp: float = 0.7,
                   hose_len: float = 2.0, hose_dia: float = 0.019) -> dict:
    """Run the Hopsan plant at temperature T; return the standard log dict.

    Keys: t, x, v, P1, P2, Q1, Q2, F_net, F_load, T (1-D arrays).
    freq sets the sine-reference frequency (homing interval = 1/(2 freq));
    mm is a plant-parameter mismatch dict from draw_mismatch().
    """
    cli = find_hopsancli()
    model = model_path(open_loop=open_loop)
    if not os.path.exists(model):
        raise FileNotFoundError(f"Hopsan model not found: {model}")

    A1, A2 = cfg.cyl.area_cap, cfg.cyl.area_rod
    dur = cfg.duration if duration is None else duration
    n_log = int(round(dur * cfg.sens.sample_rate))

    with tempfile.TemporaryDirectory(prefix="hopsan_run_") as td:
        par_csv = os.path.join(td, "params.csv")
        out_csv = os.path.join(td, "results.csv")
        _write_param_csv(par_csv, parameter_rows(cfg, T, freq=freq, mm=mm, p_rep=p_rep,
                                                 gain_scale=gain_scale, xvmax=xvmax,
                                                 open_loop=open_loop, spool_amp=spool_amp,
                                                 hose_len=hose_len, hose_dia=hose_dia))

        cmd = [cli]
        comp_lib = find_component_lib()
        if comp_lib:
            cmd += ["-e", comp_lib]
        cmd += [
            "-m", model,
            "--parameterImport", par_csv,
            "-s", f"0,{cfg.sim_dt},{dur}",
            "-l", str(n_log),
            "--resultsFullCSV", out_csv,
            "--resultsCSVSort", "cols",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_csv):
            raise RuntimeError(
                "HopsanCLI failed (rc=%d).\nCMD: %s\nSTDOUT:\n%s\nSTDERR:\n%s"
                % (proc.returncode, " ".join(cmd), proc.stdout, proc.stderr))

        cols = _read_results_csv(out_csv)

    t = _pick(cols, COL_T)
    P1 = _pick(cols, COL_P1)
    P2 = _pick(cols, COL_P2)
    Q1 = _pick(cols, COL_Q1)
    Q2 = _pick(cols, COL_Q2)
    x = None
    for cand in COL_X_CANDIDATES:
        x = _pick(cols, cand)
        if x is not None:
            break

    missing = [n for n, v in
               [(COL_T, t), (COL_P1, P1), (COL_P2, P2),
                (COL_Q1, Q1), (COL_Q2, Q2), ("position", x)] if v is None]
    if missing:
        raise KeyError(
            "Result CSV is missing required variables %s.\nAvailable columns:\n%s"
            "\nMake sure these variables are logged (connect them to a Scope in "
            "the Hopsan GUI, or check component names)."
            % (missing, sorted(cols.keys())))

    Q1 = COL_Q_SIGN * Q1
    Q2 = COL_Q_SIGN * Q2
    x = np.clip(x, 0.0, cfg.cyl.stroke)
    v = np.gradient(x, t)
    # ground-truth net force from the *cylinder* chamber pressures (downstream of
    # the hose loss); fall back to the manifold pressures if the line elements
    # are absent (then there is no loss and the two coincide).
    P1c = _pick(cols, COL_P1_CYL)
    P2c = _pick(cols, COL_P2_CYL)
    if P1c is None or P2c is None:
        P1c, P2c = P1, P2
    F_net = P1c * A1 - P2c * A2

    return {
        "t": t, "x": x, "v": v,
        "P1": P1, "P2": P2, "Q1": Q1, "Q2": Q2,
        "P1_cyl": P1c, "P2_cyl": P2c,
        "F_net": F_net, "F_load": np.zeros_like(t),
        "T": np.full(t.size, T),
    }
