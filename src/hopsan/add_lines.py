"""Insert hose/line elements between the valve manifold and the cylinder, so the
pressure/flow sensors (logged at the valve = manifold) sit upstream of a real
hose run. Per line: a Hydraulic Volume (C, the line compliance + the manifold
node) then a Hydraulic Laminar Orifice (Q, the line friction loss), giving the
TLM-legal chain  valve(Q) -> Volume(C) -> Orifice(Q) -> Cylinder(C).

Idempotent: running twice is a no-op. Numeric values are placeholders; the real
length/diameter/temperature-dependent values are written per run by
hopsan_plant.parameter_rows()."""
import re, sys, os

VOL = ('<component typename="HydraulicVolume" name="{name}">'
       '<parameters>'
       '<parameter unit="m^3" value="1e-05" type="double" name="V#Value"/>'
       '<parameter unit="Pa" value="1e+09" type="double" name="Beta_e#Value"/>'
       '<parameter unit="-" value="0.1" type="double" name="alpha#Value"/>'
       '</parameters>'
       '<ports><port nodetype="NodeHydraulic" name="P1"/>'
       '<port nodetype="NodeHydraulic" name="P2"/></ports>'
       '<hopsangui><pose x="{x}" y="2470" flipped="false" a="0"/></hopsangui>'
       '</component>')
ORF = ('<component typename="HydraulicLaminarOrifice" name="{name}">'
       '<parameters>'
       '<parameter unit="m^5/Ns" value="1e-09" type="double" name="Kc#Value"/>'
       '</parameters>'
       '<ports><port nodetype="NodeHydraulic" name="P1"/>'
       '<port nodetype="NodeHydraulic" name="P2"/></ports>'
       '<hopsangui><pose x="{x}" y="2475" flipped="false" a="0"/></hopsangui>'
       '</component>')


def conn(sc, sp, ec, ep):
    return (f'<connect startcomponent="{sc}" startport="{sp}" '
            f'endcomponent="{ec}" endport="{ep}">'
            '<hopsangui><coordinates>'
            '<coordinate x="0" y="0"/><coordinate x="1" y="0"/>'
            '</coordinates><geometries><geometry>diagonal</geometry></geometries>'
            '<style>solid</style></hopsangui></connect>')


def patch(path):
    s = open(path).read()
    if 'name="Line_Vol_A"' in s:
        print(f"  {os.path.basename(path)}: already patched, skipping")
        return
    # add components before </objects>
    comps = (VOL.format(name="Line_Vol_A", x="2555") + ORF.format(name="Line_Orif_A", x="2563")
             + VOL.format(name="Line_Vol_B", x="2595") + ORF.format(name="Line_Orif_B", x="2603"))
    s = s.replace("</objects>", comps + "\n    </objects>", 1)

    # remove the two direct valve->cylinder connections
    for ep, sp in (("P1", "PA"), ("P2", "PB")):
        pat = (rf'<connect endport="{ep}" endcomponent="Cylinder_C" '
               rf'startport="{sp}" startcomponent="4_3_Servo_Valve">.*?</connect>')
        s, n = re.subn(pat, "", s, flags=re.S)
        assert n == 1, f"expected 1 valve->cyl {sp} connection, found {n}"

    # add the chained connections before </connections>
    chains = (
        conn("4_3_Servo_Valve", "PA", "Line_Vol_A", "P1")
        + conn("Line_Orif_A", "P1", "Line_Vol_A", "P2")
        + conn("Line_Orif_A", "P2", "Cylinder_C", "P1")
        + conn("4_3_Servo_Valve", "PB", "Line_Vol_B", "P1")
        + conn("Line_Orif_B", "P1", "Line_Vol_B", "P2")
        + conn("Line_Orif_B", "P2", "Cylinder_C", "P2"))
    s = s.replace("</connections>", chains + "\n    </connections>", 1)

    open(path, "w").write(s)
    print(f"  {os.path.basename(path)}: patched (added 4 components, rewired 2 lines)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("sensorless_cylinder_plant.hmf", "sensorless_cylinder_plant_openloop.hmf"):
        patch(os.path.join(here, f))
