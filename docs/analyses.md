# Analysis Types

Ohmnivore runs three analysis types: DC operating point, AC frequency sweep, and transient. A dot command in the netlist requests each analysis.

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
V(2),0.6425...
I(V1),-0.004357...
```

The diode drops about 0.64V, matching a typical silicon junction.

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

Simulates the circuit over time with time-varying sources. Uses backward Euler integration with adaptive timestep and Newton-Raphson for nonlinear elements.

**Syntax:**

```spice
.TRAN tstep tstop [tstart] [UIC]
```

| Parameter | Description |
|---|---|
| `tstep` | Suggested output timestep |
| `tstop` | End time |
| `tstart` | Start time for recording output (default: 0) |
| `UIC` | Use Initial Conditions -- skip DC operating point, start from zero |

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

## GPU Acceleration

Add the `--gpu` flag to use the GPU solver:

```sh
ohmnivore circuit.spice --gpu
```

The GPU solver uses wgpu (Vulkan/Metal/DX12) with BiCGSTAB iteration. It benefits large circuits where the matrix solve dominates runtime. For small circuits, direct LU decomposition on the CPU is faster.

If ohmnivore finds no compatible GPU, it exits with an error. Omit `--gpu` to fall back to the CPU solver.
