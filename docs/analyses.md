# Analysis Types

Ohmnivore runs three analysis types: DC operating point, AC frequency sweep, and transient. A dot command in the netlist requests each analysis.

> **Migration note:** The active C++ Phase 3A path adds deterministic FP64 diode `.DC`/`.OP` to
> the linear RLCVI `.DC`/`.OP`, `.AC`, and `.TRAN` subset. Every production solve uses the
> checksum-pinned KLU real/complex FP64 sparse-direct solver. The former dense partial-pivoting
> implementation is an exact-small test oracle only. BJT, MOSFET, nonlinear AC/transient, GPU
> solver-dispatch, and CLI-option details below remain legacy Rust reference material.

All results print to stdout as CSV.

## DC Operating Point

Computes the steady-state DC voltages and currents.

**Syntax:**

```spice
.DC
```

(`.OP` is an alias for `.DC`.)

**Example -- voltage divider:**

```spice
* Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
```

```sh
$ ohmnivore divider.spice
Variable,Value
V(1),10
V(2),5
I(V1),-0.005
```

The output lists node voltages as `V(node)` and branch currents as `I(source)`. The negative sign means 5 mA flows into V1's positive terminal (power supply convention).

**Example -- diode circuit:**

```spice
* Diode Forward Bias
V1 1 0 DC 5
R1 1 2 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 2 0 DMOD
.DC
.END
```

```sh
$ ohmnivore diode.spice
Variable,Value
V(1),5
V(2),0.6924903752200562
I(V1),-0.004307509629779944
```

The fixed-temperature Phase 3A model gives a diode drop of about `0.69249 V`.

For a diode voltage `v=va-vc`, Phase 3A evaluates
`i=IS*expm1(v/(N*0.02585 V))` and
`g=IS*exp(v/(N*0.02585 V))/(N*0.02585 V)`. The exponent is clamped to `[-80,80]`; every
intermediate and result must be finite and no larger than `1e100` in magnitude. In the MNA
model calculation, a positive conductance that underflows to zero is rejected. In the MNA residual
`F(x)=G*x-b+sum(S*i)`, `S` is `+1` at the anode and `-1` at the cathode. The Jacobian is
`J=G+sum(S*g*S^T)`, giving the stamp `+g,-g,-g,+g` at `aa,ac,ca,cc`. Ground has no row or column.

Newton starts from the all-zero state and evaluates devices in circuit insertion order. Each KLU
step solves `J*delta=-F`, after which SPICE3 logarithmic PN-junction limiting is applied in that
same order. Scaled voltage and current update limits are respectively
`1e-9 V + 1e-6*scale` and `1e-12 A + 1e-6*scale`; acceptance also requires a freshly recomputed
residual using `1e-12 A + 1e-6*row_scale` for node KCL rows and
`1e-9 V + 1e-6*row_scale` for branch KVL rows. A direct solve has at most 50 iterations. If it
does not converge, fixed source scales `0.0,0.1,...,1.0` receive 30 iterations each, followed by
fixed extra-GMIN values `1e-3,1e-4,...,1e-12,0 S` with 30 iterations at nonzero values and 50 at
zero. There is no adaptive retry schedule. The final state is accepted only after validation with
full sources and no extra GMIN; the normal production `1e-12 S` node conductance remains present.

Compilation creates one canonical CSR union pattern containing the existing linear stamps and all
possible diode Jacobian coordinates. Required numerical zeros remain explicit, rows have strictly
increasing columns, and one KLU symbolic analysis is reused across all iterations and continuation
attempts. Numeric values are refactorized with the Phase 2D pivot-safe retry and backward-error
validation. Invalid structure or size, missing/invalid models, non-finite arithmetic, singular or
rank-deficient Jacobians, factorization or validation failure, and exhausted bounded continuation
return typed errors. None may fall back to the dense test oracle.

Even an iteration-zero point whose residual already passes must numerically factor and zero-solve
its current Jacobian through KLU before acceptance. Repeated direct, source-stepping, and
GMIN-stepping runs are bitwise deterministic on one supported toolchain/platform. Phase 3A makes
no cross-libm or cross-platform numerical-equivalence claim; scalar-oracle and ngspice checks use
their individual recorded tolerances.

### CSV Format

```csv
Variable,Value
V(node1),voltage
V(node2),voltage
I(Vsource),current
```

## AC Frequency Sweep

Sweeps a frequency range and reports magnitude and phase at each node. Sources with an `AC` specification drive the stimulus; the `DC` value sets the linearization operating point.

**Syntax:**

```spice
.AC DEC npoints fstart fstop    * logarithmic, npoints per decade
.AC OCT npoints fstart fstop    * logarithmic, npoints per octave
.AC LIN npoints fstart fstop    * linear, npoints total
```

On the C++ Phase 3A path, all frequencies must be finite and positive with `fstop > fstart`.
DEC/OCT require a positive points-per-decade/octave value and emit the start, geometric interior
grid, and exact stop once (`ceil(npoints * log_base(fstop/fstart)) + 1` total rows). LIN requires at
least two points, emits exactly `npoints` rows, and includes both endpoints. Sweeps are strictly
increasing and limited to one million generated points.
If the requested density cannot be represented without duplicate FP64 frequencies, the C++ path
returns a typed error instead of silently reducing the point count. CSV text fields containing
commas, quotes, or line breaks use standard doubled-quote escaping; ordinary identifiers retain
the legacy unquoted schema.

**Example -- RC low-pass filter:**

```spice
* RC Low-Pass Filter
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC DEC 10 1 1000000
.END
```

```sh
$ ohmnivore lowpass.spice
Frequency,V(1)_mag,V(1)_phase_deg,V(2)_mag,V(2)_phase_deg,I(V1)_mag,I(V1)_phase_deg
1,1,0,0.99998,-0.36,0.0000063,-90.36
...
100,1,0,0.8467,-32.14,0.000532,-122.14
...
100000,1,0,0.001591,-89.09,0.000999,-179.09
```

At 100 Hz the output magnitude is about 0.85 (within the passband). At 100 kHz it has rolled off to 0.0016 -- deep in the stopband. The -3dB cutoff of a 1k/1uF filter is about 159 Hz.

### CSV Format

```csv
Frequency,V(n1)_mag,V(n1)_phase_deg,V(n2)_mag,V(n2)_phase_deg,I(V1)_mag,I(V1)_phase_deg
```

Each frequency point is a row. Magnitudes are linear (not dB). Phases are in degrees.

## Transient Analysis

The C++ Phase 3A path simulates only linear RLCVI transient circuits. It solves
`G*x + C*dx/dt = b(t)` with backward Euler for the initial, recovery, and waveform-breakpoint
landing steps, then trapezoidal integration with deterministic adaptive timestep control. A
discontinuous source is integrated to its edge with the left-limit forcing; the right-limit
algebraic state is then projected without changing capacitor voltages or inductor currents. It
does not perform Newton iteration or admit nonlinear devices.

**Syntax:**

```spice
.TRAN tstep tstop [tstart] [UIC]
```

| Parameter | Description |
|---|---|
| `tstep` | Hard maximum accepted timestep |
| `tstop` | End time |
| `tstart` | Exact first output time; integration still starts at zero (default: 0) |
| `UIC` | Skip the DC solve and enforce zero capacitor voltage and zero inductor current |

All time values must be finite. `tstep` and `tstop` must be positive, and
`0 <= tstart <= tstop`. Parsing consumes the full directive, so misspelled `UIC` or extra tokens
are errors.

The accepted time grid is strictly increasing. It contains exact `tstart` and `tstop` values plus
every PULSE, SIN-delay, PWL, and EXP breakpoint within the interval; adaptive stepping cannot jump
over them. Dynamic BE steps compare one full step with two half steps; trapezoidal steps compare
TRAP and BE with `(2/3)*|x_trap-x_be|`. Both estimates are scaled by
`1e-9 + 1e-3*max(|x_high|,|x_low|)`. Its safety factor is `0.9`,
growth/shrink factor is clamped to `[0.5,2]`, and its adaptive minimum is `tstep/10000`.
Mandatory exact-boundary clips may be smaller than that minimum. Non-finite arithmetic,
unrepresentable progress, singular systems, minimum-step exhaustion, and bounded step/attempt
limits return typed failures instead of partial output. The bounds are 1,000,000 accepted steps
and 2,000,000 attempts. An algebraic-only BE step has zero dynamic LTE; a mandatory one-ULP
boundary clip is accepted when no representable midpoint exists.

Without UIC, the complete state comes from the existing GMIN-inclusive DC operating-point solve.
With UIC, zero reactive state is a constraint, not an assumption that every voltage/current is
zero; the remaining algebraic system is solved and inconsistent constraints fail. A transient
constraint already imposed consistently by an ideal voltage source is accepted as redundant. A
transient waveform replaces its source's DC value while transient time advances. Transient-only sources use
zero during DC or UIC initialization.

**Example -- RC charging:**

```spice
* RC Charging
V1 1 0 DC 5
R1 1 2 1k
C1 2 0 1u
.TRAN 10u 5m 0 UIC
.END
```

```sh
$ ohmnivore charging.spice
time,V(1),V(2),I(V1)
0,5,0,-0.005
0.00001,5,0.0495,-0.00495
0.00002,5,0.0988,-0.00490
...
0.005,5,4.9327,-0.0000673
```

The capacitor charges from 0V toward 5V with a time constant of RC = 1ms.

**Example -- pulse through RC filter:**

```spice
* Pulse through RC filter
V1 1 0 PULSE(0 5 0 1n 1n 0.5m 1m)
R1 1 2 1k
C1 2 0 1u
.TRAN 10u 1m 0 UIC
.END
```

This applies a 1 kHz square wave (0-5V) to an RC filter and records the filtered output at node 2.

### CSV Format

```csv
time,V(n1),V(n2),I(V1)
```

Each time point is a row. All node voltages and branch currents are included.
Nodes and the interleaved voltage-source/inductor branches retain IR insertion order. Identifiers
requiring CSV quoting use standard doubled-quote escaping.

## GPU Acceleration

The GPU solver is used by default. Pass `--cpu` to force the CPU solver:

```sh
ohmnivore circuit.spice --cpu
```

The GPU solver uses wgpu (Vulkan/Metal/DX12) with BiCGSTAB iteration. It benefits large circuits where the matrix solve dominates runtime. For small circuits, direct LU decomposition on the CPU may be faster.

If ohmnivore finds no compatible GPU, it exits with an error. Use `--cpu` to fall back to the CPU solver.
