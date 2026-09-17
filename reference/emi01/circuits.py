"""Frozen external-reference decks and explicit hypothetical filter design model."""

import math

OPTIONS = ".options reltol=1e-5 abstol=1e-9 vntol=1e-7 chgtol=1e-16 method=gear maxord=2 itl1=300 itl4=100"
DPT_NAMES = ["time", "v(d)", "v(g)", "i(lload)", "i(vdrain)"]
STUDY_NAMES = (
    ["time"]
    + [
        f"v({n})"
        for n in [
            "p",
            "a",
            "b",
            "ga",
            "gb",
            "fa",
            "fb",
            "ch",
            "la",
            "lb",
            "lr",
            "xcap",
            "yca",
            "ycb",
        ]
    ]
    + [
        f"i({n})"
        for n in [
            "va",
            "vb",
            "vdah",
            "vdal",
            "vdbh",
            "vdbl",
            "lda",
            "ldb",
            "lcma",
            "lcmb",
            "lch",
            "lload",
        ]
    ]
)
STUDY_NAMES = ["time"] + sorted(STUDY_NAMES[1:], key=lambda name: name[2:-1])


def design(c):
    """SI calculations; geometry input mm², mm, integer turns, mm."""
    result = {}
    mass = 0.0
    for kind, cores, windings in [("dm", 2, 1), ("cm", 1, 2)]:
        ae_mm2, le_mm, turns, mlt_mm = c[f"{kind}_geometry"]
        ae, le, mlt = ae_mm2 * 1e-6, le_mm * 1e-3, mlt_mm * 1e-3
        copper_volume = turns * mlt * 4e-6
        mass += 1.2 * cores * (4800 * ae * le + windings * 8960 * copper_volume)
        result[f"r{kind}_ohm"] = 1.724e-8 * turns * mlt / 4e-6
        inductance = c[f"l{kind}_h"]
        result[f"{kind}_mu_effective"] = (
            inductance * le / (4e-7 * math.pi * turns**2 * ae)
        )
        result[f"{kind}_gap_equivalent_m"] = 4e-7 * math.pi * turns**2 * ae / inductance
    cap_mass = sum(
        0.001 + 2 * (cap * 630**2 / 2) / 0.2 * 0.0012
        for cap in [c["cx_f"], c["cy_f"], c["cy_f"]]
    )
    result["mass_kg"] = mass + cap_mass + 0.010
    result["capacitor_mass_kg"] = cap_mass
    return result


def dpt(max_step):
    return f"""EMI-01 v1 double pulse
.include model.lib
Vbus bus 0 400
Rbus bus p1 0.02
Lbus p1 p 20n
Cbus p1 0 10u
Lload p sw 200u
Rload sw drain 0.005
Vdrain drain d 0
Xlo d g 0 MSC040SMA120B
Xhi p gh drain MSC040SMA120B
Vhi ghdrive drain -3
Rgh ghdrive gh 4.7
Vgate drive 0 PWL(0 -3 1u -3 1.01u 20 10u 20 10.01u -3 12u -3 12.01u 20 14u 20 14.01u -3)
Rg drive g 4.7
{OPTIONS}
.tran {max_step / 2:.17g} 16u 0 {max_step:.17g}
.save {" ".join(DPT_NAMES[1:])}
.end
"""


def ensemble(c, k, max_step):
    d = design(c)
    lc, cc, sc = k["l_scale"], k["c_scale"], k["stray_scale"]
    lines = [
        f"EMI-01 v1 {c['id']} {k['id']}",
        ".include model.lib",
        f"Vbus bus 0 {k['bus_v']}",
        "Rbus bus p1 0.02",
        "Lbus p1 p 20n",
        "Cbus p1 0 10u",
    ]
    for leg, width in [("a", 14.8e-6), ("b", 4.8e-6)]:
        # The model's source/package inductance stays inside each coupled leg.
        lines += [
            f"Vd{leg}h p {leg}hd 0",
            f"Xd{leg}h {leg}hd g{leg}h {leg} MSC040SMA120B PARAMS: TJ_C={k['tj_c']}",
            f"Vd{leg}l {leg} {leg}ld 0",
            f"Xd{leg}l {leg}ld g{leg} 0 MSC040SMA120B PARAMS: TJ_C={k['tj_c']}",
            f"Vg{leg}h drive{leg}h {leg} PULSE(-3 20 1u 10n 10n {width:.17g} 20u)",
            f"Vg{leg}l drive{leg}l 0 PULSE(20 -3 0.8u 10n 10n {width + 400e-9:.17g} 20u)",
            f"Rg{leg}h drive{leg}h g{leg}h {4.7 * k['rg_scale']:.17g}",
            f"Rg{leg}l drive{leg}l g{leg} {4.7 * k['rg_scale']:.17g}",
            f"Csw{leg} {leg} ch {100e-12 * sc:.17g}",
            f"Rd{leg} {leg} {leg}r {d['rdm_ohm']:.17g}",
            f"Ld{leg} {leg}r {leg}m {c['ldm_h'] * lc:.17g}",
            f"Rcm{leg} {leg}m {leg}mr {d['rcm_ohm']:.17g}",
            f"Lcm{leg} {leg}mr f{leg} {c['lcm_h'] * lc:.17g}",
            f"Cwind{leg} {leg}mr f{leg} {10e-12 * sc:.17g}",
            f"Ry{leg} f{leg} yc{leg} 22",
            f"Cy{leg} yc{leg} ch {c['cy_f'] * cc:.17g}",
            f"V{leg} f{leg} h{leg} 0",
            f"Rh{leg} h{leg} h{leg}r 0.1",
            f"Lh{leg} h{leg}r l{leg} 1u",
            f"Ch{leg} l{leg} ch {200e-12 * sc:.17g}",
            f"Cl{leg} l{leg} ch {1e-9 * sc:.17g}",
        ]
    lines += [
        "Kcm Lcma Lcmb 0.995",
        "Kh Lha Lhb 0.2",
        f"Ccouple fa fb {5e-12 * sc:.17g}",
        "Rx fa xcap 0.2",
        f"Cx xcap fb {c['cx_f'] * cc:.17g}",
        "Rload la lr 20",
        "Lload lr lb 100u",
        "Rloadloss lr lb 100",
        "Rch ch chr 1",
        "Lch chr 0 50n",
        f"Cbp p ch {100e-12 * sc:.17g}",
        f"Cbn 0 ch {100e-12 * sc:.17g}",
        OPTIONS,
        f".tran {max_step / 2:.17g} 200u 0 {max_step:.17g}",
        ".save " + " ".join(STUDY_NAMES[1:]),
        ".end",
    ]
    return "\n".join(lines) + "\n"


def driver():
    return """EMI-01 v1 isolated PSpice compatibility driver
.control
set ngbehavior=ps
set numdgt=17
set filetype=binary
source circuit.cir
run
write waveform.raw all
rusage all
quit
.endc
.end
"""
