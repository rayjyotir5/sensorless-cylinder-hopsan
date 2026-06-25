"""Offline self-tests for the Hopsan plant integration.

These validate everything checkable WITHOUT Hopsan installed:
  - the vendored .hmf is valid XML and has the expected components,
  - every parameter name the driver overrides actually exists in the model
    (catches typos in the Component#Parameter mapping),
  - the reference source is the SignalSineWave with its 'out' wired to the
    controller,
  - the results-CSV parser produces the expected log-dict,
  - HopsanCLI discovery raises a clear error when absent.

Run:  python -m tests.test_hopsan_integration       (or: python tests/...py)
"""
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cylinder_sim.params import Config
from cylinder_sim import hopsan_plant as hp


def _model_root():
    return ET.parse(hp.model_path()).getroot()


def test_model_valid_and_components():
    r = _model_root()
    names = {c.get("name"): c.get("typename") for c in r.iter("component")}
    assert names.get("Cylinder_C") == "HydraulicCylinderC"
    assert names.get("4_3_Servo_Valve") == "Hydraulic43Valve"
    assert names.get("Step") == "SignalSineWave", "reference must be sine-wave"
    print("OK  model valid; cylinder/valve/sine-reference present")


def test_reference_wiring_intact():
    r = _model_root()
    conns = {(c.get("startcomponent"), c.get("startport"),
              c.get("endcomponent"), c.get("endport")) for c in r.iter("connect")}
    assert ("Step", "out", "Subtract", "in1") in conns
    print("OK  reference Step.out -> Subtract.in1 wired")


def test_all_override_params_exist():
    """Every Component#Param the driver sets must exist in the model."""
    r = _model_root()
    available = set()
    for c in r.iter("component"):
        cname = c.get("name")
        for p in c.iter("parameter"):
            available.add(f"{cname}#{p.get('name')}")

    cfg = Config()
    bad = []
    for name, _ in hp.parameter_rows(cfg, 25.0):
        if name not in available:
            bad.append(name)
    assert not bad, (
        "these override names are NOT in the model: %s\n(available sample: %s)"
        % (bad, sorted(p for p in available if p.split('#')[0] in
           {'Cylinder_C', 'Step', 'Mass'})))
    print(f"OK  all {len(hp.parameter_rows(cfg, 25.0))} parameter overrides "
          "exist in the model")


def test_param_values_temperature_varies():
    cfg = Config()
    def beta(T):
        return dict(hp.parameter_rows(cfg, T))["Cylinder_C#Beta_e#Value"]
    assert beta(0.0) > beta(50.0), "bulk modulus should be stiffer cold"
    cold_leak = dict(hp.parameter_rows(cfg, 0.0))["Cylinder_C#c_leak#Value"]
    hot_leak = dict(hp.parameter_rows(cfg, 50.0))["Cylinder_C#c_leak#Value"]
    assert hot_leak > cold_leak, "leakage should rise when hot/thin"
    print("OK  temperature-dependent params vary correctly (beta, leak)")


def test_results_parser():
    """Parse a CSV in Hopsan's real cols layout: names, blank row, units row,
    then data. Uses the valve-side column names the driver maps to."""
    import tempfile
    cfg = Config()
    A1, A2 = cfg.cyl.area_cap, cfg.cyl.area_rod
    rows = [
        f"{hp.COL_P1},{hp.COL_P2},{hp.COL_Q1},{hp.COL_Q2},"
        "Position_Sensor#out#Value,Time",
        ",,,,,",                                     # blank row (Hopsan writes this)
        "Pa,Pa,m^3/s,m^3/s,m,s",                     # units row
        "1.0e7,2.0e6,1.0e-4,-5.0e-5,0.0,0.0",
        "1.1e7,2.1e6,1.1e-4,-5.1e-5,0.10,0.001",
        "1.2e7,2.2e6,1.2e-4,-5.2e-5,0.20,0.002",
    ]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "r.csv")
        open(p, "w").write("\n".join(rows) + "\n")
        cols = hp._read_results_csv(p)
    assert hp._pick(cols, hp.COL_P1)[2] == 1.2e7, "blank+units rows not skipped"
    assert hp._pick(cols, hp.COL_T)[-1] == 0.002
    x = hp._pick(cols, "Position_Sensor#out#Value")
    assert x[-1] == 0.20
    assert 1.2e7 * A1 - 2.2e6 * A2 > 0      # F_net sanity
    print("OK  results-CSV parser skips blank+units rows; column mapping works")


def test_hopsancli_discovery_error():
    saved = os.environ.pop("HOPSANCLI", None)
    try:
        import shutil
        if any(shutil.which(n) for n in ("HopsanCLI", "hopsancli")):
            print("..  HopsanCLI present on PATH; skipping not-found check")
            return
        try:
            hp.find_hopsancli()
            assert False, "should have raised HopsanNotFound"
        except hp.HopsanNotFound as e:
            assert "Install Hopsan" in str(e)
            print("OK  HopsanCLI-absent raises a clear, actionable error")
    finally:
        if saved:
            os.environ["HOPSANCLI"] = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} checks passed")
    sys.exit(1 if fails else 0)
