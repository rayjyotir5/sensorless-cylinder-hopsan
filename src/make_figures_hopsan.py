"""Render (a) the Hopsan plant schematic straight from the .hmf canvas layout,
and (b) a HopsanCLI console 'screenshot' from a real run."""
import os, sys, subprocess, shutil, textwrap
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
HMF = os.path.join(HERE, "hopsan", "sensorless_cylinder_plant.hmf")

# domain colouring by component typename prefix/keyword
def domain_color(typename):
    t = typename.lower()
    if "hydraulic" in t or "tank" in t or "pump" in t or "valve" in t or "volume" in t:
        return "#cfe8ff", "#1f6fb2"      # hydraulic = blue
    if "mechanic" in t or "mass" in t or "spring" in t or "position" in t:
        return "#d7f0d7", "#2e7d32"      # mechanical = green
    return "#ffe6c7", "#c77a1f"          # signal = orange


def fig_hopsan_model():
    root = ET.parse(HMF).getroot()
    comps = {}
    for c in root.iter("component"):
        name = c.get("name"); typ = c.get("typename")
        pose = c.find(".//pose")
        if pose is None:
            continue
        x = float(pose.get("x")); y = float(pose.get("y"))
        comps[name] = {"typ": typ, "x": x, "y": y}
    conns = [(c.get("startcomponent"), c.get("endcomponent")) for c in root.iter("connect")]

    xs = [v["x"] for v in comps.values()]; ys = [v["y"] for v in comps.values()]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    def nx(x): return (x - x0) / (x1 - x0 + 1e-9)
    def ny(y): return 1.0 - (y - y0) / (y1 - y0 + 1e-9)   # flip: Hopsan y is downward

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.1, 1.12); ax.axis("off")

    # connections first (under boxes)
    for a, b in conns:
        if a in comps and b in comps:
            ax.add_patch(FancyArrowPatch(
                (nx(comps[a]["x"]), ny(comps[a]["y"])),
                (nx(comps[b]["x"]), ny(comps[b]["y"])),
                arrowstyle="-", color="#888", lw=1.2, alpha=0.7,
                connectionstyle="arc3,rad=0.05"))

    nice = {"Cylinder_C": "Cylinder\n(HydraulicCylinderC)",
            "4_3_Servo_Valve": "4/3 Servo Valve\n(Hydraulic43Valve)",
            "Q_type_Fixed_Displacement_Pump_1": "Pump",
            "Pressure_Relief_Valve": "Relief Valve",
            "Mass": "Mass + end-stops",
            "Position_Sensor": "Position Sensor",
            "Step": "Reference\n(SignalSineWave)",
            "Subtract": "Σ (error)", "GainP": "Kp", "GainI": "Ki",
            "Integrator": "∫", "Add": "Σ",
            "Hydraulic_Volume_Multi_Port": "Supply Volume",
            "Translational_Spring": "Load Spring",
            "Fixed_Position_Attachment": "Ground",
            "Tank_C": "Tank", "Tank_C_1": "Tank", "Tank_C_2": "Tank",
            "Scope": "Scope", "Sink": "Sink"}

    for name, v in comps.items():
        face, edge = domain_color(v["typ"])
        label = nice.get(name, name)
        bw, bh = 0.115, 0.072
        cx, cy = nx(v["x"]), ny(v["y"])
        ax.add_patch(FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
                     boxstyle="round,pad=0.006,rounding_size=0.012",
                     facecolor=face, edgecolor=edge, lw=1.6, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.0,
                zorder=4, color="#222")

    # legend
    import matplotlib.patches as mp
    handles = [mp.Patch(facecolor="#cfe8ff", edgecolor="#1f6fb2", label="Hydraulic"),
               mp.Patch(facecolor="#d7f0d7", edgecolor="#2e7d32", label="Mechanical"),
               mp.Patch(facecolor="#ffe6c7", edgecolor="#c77a1f", label="Signal / control")]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
              frameon=True, bbox_to_anchor=(0.5, 1.07))
    ax.set_title("Hopsan plant model (rendered from the .hmf canvas layout)",
                 fontsize=12, fontweight="bold", pad=22)
    fig.savefig(os.path.join(FIG, "fig_hopsan_model.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_hopsan_model.png  (%d components, %d connections)"
          % (len(comps), len(conns)))


def fig_hopsan_cli():
    cli = shutil.which("hopsancli") or shutil.which("HopsanCLI")
    deflib = None
    for root in [os.path.dirname(os.path.realpath(cli))+"/../componentLibraries/defaultLibrary"]:
        for n in ("libdefaultcomponentlibrary.dylib", "libdefaultcomponentlibrary.so"):
            p = os.path.join(root, n)
            if os.path.exists(p): deflib = os.path.realpath(p)
    cmd = [cli, "-e", deflib, "-m", HMF, "-s", "0,1e-4,2.0", "-l", "100",
           "--resultsFullCSV", "/tmp/_cli_demo.csv", "--resultsCSVSort", "cols"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    # strip ANSI + drop benign external-lib path probes; truncate long lines
    import re
    raw = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout).splitlines()
    clean = []
    for ln in raw:
        if "no such file" in ln.lower() or "opening external lib" in ln.lower():
            continue
        if len(ln) > 78:
            ln = ln[:75] + "..."
        clean.append(ln)
    header = ("$ hopsancli -e libdefaultcomponentlibrary.dylib \\\n"
              "    -m sensorless_cylinder_plant.hmf \\\n"
              "    -s 0,1e-4,2.0 --resultsFullCSV results.csv --resultsCSVSort cols\n")
    body = header + "\n" + "\n".join(clean[:40]) + "\n$ _"

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.axis("off")
    ax.set_facecolor("#1e1e2e"); fig.patch.set_facecolor("#1e1e2e")
    ax.add_patch(plt.Rectangle((0, 0.972), 1, 0.028, transform=ax.transAxes,
                 color="#33334a", zorder=1))
    for i, col in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((0.018+0.02*i, 0.986), 0.006, transform=ax.transAxes,
                     color=col, zorder=2))
    ax.text(0.5, 0.986, "HopsanCLI 2.24.0  —  headless plant simulation",
            transform=ax.transAxes, ha="center", va="center", color="#cfcfe0", fontsize=8.5)
    ax.text(0.015, 0.955, body, transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=8.6, color="#d4d4e8", linespacing=1.32)
    fig.savefig(os.path.join(FIG, "fig_hopsan_cli.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_hopsan_cli.png  (rc=%d)" % out.returncode)


if __name__ == "__main__":
    fig_hopsan_model()
    fig_hopsan_cli()
    print("hopsan figures done")
