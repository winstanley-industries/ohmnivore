#!/usr/bin/env python3
"""Generate CMOS inverter chain SPICE netlists of arbitrary size."""

import argparse
import os


def generate_netlist(n_stages, analysis):
    """Generate a CMOS inverter chain netlist.

    Args:
        n_stages: Number of inverter stages.
        analysis: "dc" or "tran".

    Returns:
        Netlist string.
    """
    lines = []
    lines.append(f"* CMOS Inverter Chain - {n_stages} stages - {analysis.upper()}")
    lines.append(".MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)")
    lines.append(".MODEL PMOD PMOS(VTO=-0.7 KP=5.5e-5 LAMBDA=0.04)")
    lines.append("")
    lines.append("VDD vdd 0 DC 5")

    if analysis == "dc":
        lines.append("VIN in 0 DC 5")
    else:
        lines.append("VIN in 0 PULSE(0 5 10n 1n 1n 50n 100n)")

    lines.append("")

    for i in range(1, n_stages + 1):
        gate_in = "in" if i == 1 else f"out_{i - 1}"
        out = f"out_{i}"
        lines.append(f"* Stage {i}")
        lines.append(f"MP{i} {out} {gate_in} vdd vdd PMOD")
        lines.append(f"MN{i} {out} {gate_in} 0 0 NMOD")
        lines.append("")

    if analysis == "dc":
        lines.append(".DC")
    else:
        lines.append(".TRAN 1n 100n")

    lines.append(".END")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMOS inverter chain SPICE netlists."
    )
    parser.add_argument("stages", type=int, help="Number of inverter stages")
    args = parser.parse_args()

    if args.stages < 1:
        parser.error("Stage count must be at least 1")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuits")
    os.makedirs(out_dir, exist_ok=True)

    for analysis in ("dc", "tran"):
        netlist = generate_netlist(args.stages, analysis)
        filename = f"inverter_chain_{args.stages}_{analysis}.spice"
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            f.write(netlist)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
