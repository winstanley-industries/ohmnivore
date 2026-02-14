# Ohmnivore

A GPU-accelerated SPICE circuit simulator. Write a netlist, run an analysis, get results as CSV.

Ohmnivore parses a subset of SPICE netlists, builds Modified Nodal Analysis (MNA) matrices, and solves them on the GPU (wgpu/BiCGSTAB) or CPU (direct LU). It runs DC operating point, AC frequency sweep, and transient analyses. The solver architecture supports multi-GPU and multi-node execution via domain decomposition with RAS preconditioning and MPI communication.

## Quick Start

Build the project (requires [Rust](https://www.rust-lang.org/tools/install)):

```sh
cargo build --release
```

Create a netlist file `divider.spice`:

```spice
* Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
```

Run it:

```sh
./target/release/ohmnivore divider.spice
```

Output:

```csv
Variable,Value
V(1),10
V(2),5
I(V1),-0.005
```

Node 2 sits at 5V -- half the supply, as expected from two equal resistors.

## Usage

```
ohmnivore <netlist.spice> [--cpu]
```

Ohmnivore writes results to stdout as CSV. Redirect to a file if needed:

```sh
ohmnivore circuit.spice > results.csv
ohmnivore circuit.spice --cpu > results.csv   # use CPU solver
```

## Supported Elements

| Element | Syntax | Example |
|---|---|---|
| Resistor | `Rname n+ n- value` | `R1 1 2 10k` |
| Capacitor | `Cname n+ n- value` | `C1 2 0 100n` |
| Inductor | `Lname n+ n- value` | `L1 3 4 4.7u` |
| Voltage source | `Vname n+ n- [DC val] [AC mag [phase]]` | `V1 1 0 DC 5` |
| Current source | `Iname n+ n- [DC val] [AC mag [phase]]` | `I1 2 0 DC 1m` |
| Diode | `Dname anode cathode model` | `D1 2 0 DMOD` |
| BJT | `Qname C B E model` | `Q1 vc base 0 Q2N2222` |
| MOSFET | `Mname D G S model` | `M1 drain gate 0 NMOD` |

Values accept engineering suffixes (case-insensitive): `T` `G` `MEG` `K` `M` `U` `N` `P` `F`.

## Examples

The `examples/` directory contains circuits you can run immediately. These netlists also work with [ngspice](https://ngspice.sourceforge.io/).

| File | Analysis | What it demonstrates |
|---|---|---|
| `voltage_divider.spice` | DC | Two resistors splitting a voltage |
| `diode_clamp.spice` | DC | Diode clamping with a .MODEL |
| `common_source.spice` | DC | NMOS amplifier biasing |
| `rc_lowpass.spice` | AC | Frequency response of an RC filter |
| `rc_charging.spice` | Transient | Capacitor charging curve |
| `pulse_filter.spice` | Transient | Square wave through an RC filter |

```sh
ohmnivore examples/voltage_divider.spice
ohmnivore examples/rc_lowpass.spice > lowpass.csv
```

## Distributed / Multi-GPU

Ohmnivore can partition circuits across multiple GPUs via METIS graph decomposition. Each GPU solves its subdomain with a local ISAI(1) preconditioner, coordinated by a global BiCGSTAB solver. Enable MPI support:

```sh
cargo build --release --features distributed
```

This requires an MPI installation (e.g., OpenMPI). See the [Installation Guide](docs/installation.md) for details.

## Documentation

- **[Installation Guide](docs/installation.md)** -- building from source, GPU setup, MPI, platform notes
- **[Netlist Format](docs/netlist-format.md)** -- component syntax, models, node naming, transient sources
- **[Analysis Types](docs/analyses.md)** -- DC, AC, and transient analysis with worked examples
