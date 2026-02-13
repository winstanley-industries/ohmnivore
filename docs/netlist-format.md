# Netlist Format

Ohmnivore reads SPICE-subset netlists. This document covers the supported syntax.

## Structure

A netlist is a plain text file with one statement per line:

```spice
* Title or comment
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
```

- Lines starting with `*` are comments.
- `.END` marks the end of the netlist. The parser ignores everything after it.
- Blank lines are ignored.
- Syntax is case-insensitive (`R1`, `r1`, `.dc`, `.DC` all work).

## Nodes

Nodes are named with alphanumeric strings: `0`, `1`, `out`, `vcc`, `n1`. The reference (ground) node is `0` or `GND`.

Every circuit must have at least one connection to ground.

## Values and Suffixes

Numeric values accept engineering suffixes and scientific notation:

| Suffix | Multiplier | Example |
|---|---|---|
| `T` | 10^12 | `1T` = 1e12 |
| `G` | 10^9 | `2.5G` = 2.5e9 |
| `MEG` | 10^6 | `1MEG` = 1e6 |
| `K` | 10^3 | `10K` = 10000 |
| `M` | 10^-3 | `100M` = 0.1 |
| `U` | 10^-6 | `4.7U` = 4.7e-6 |
| `N` | 10^-9 | `100N` = 1e-7 |
| `P` | 10^-12 | `22P` = 22e-12 |
| `F` | 10^-15 | `10F` = 1e-14 |

Scientific notation also works: `1e3`, `2.52e-9`, `1e-14`.

Suffixes are case-insensitive. `MEG` must be spelled out to distinguish it from `M` (milli).

## Passive Components

```
Rname  n+  n-  value        Resistor (ohms)
Cname  n+  n-  value        Capacitor (farads)
Lname  n+  n-  value        Inductor (henries)
```

Current flows from `n+` to `n-`. Examples:

```spice
R1 1 2 10k        * 10 kohm resistor between nodes 1 and 2
C1 out 0 100n     * 100 nF capacitor from out to ground
L1 3 4 4.7u       * 4.7 uH inductor between nodes 3 and 4
```

## Independent Sources

Voltage and current sources support DC, AC, and transient specifications:

```
Vname  n+  n-  [DC val]  [AC mag [phase]]  [transient_func]
Iname  n+  n-  [DC val]  [AC mag [phase]]  [transient_func]
```

All three specs are optional and combinable. A bare number means DC. Examples:

```spice
V1 1 0 DC 5               * 5V DC source
V1 1 0 5                  * same thing (bare number = DC)
V1 1 0 AC 1 0             * 1V AC source at 0 degrees
V1 1 0 DC 5 AC 1 0        * both DC and AC
I1 2 0 DC 1m              * 1 mA DC current source
```

For voltage sources, conventional current flows from `n+` through the external circuit to `n-` (positive current enters at `n+`).

### Transient Source Functions

For transient analysis, sources specify a time-varying waveform:

**PULSE** -- rectangular pulse:

```
PULSE(v1 v2 td tr tf pw per)
```

| Parameter | Description | Default |
|---|---|---|
| `v1` | Initial value | required |
| `v2` | Pulse value | required |
| `td` | Delay time | 0 |
| `tr` | Rise time | 0 |
| `tf` | Fall time | 0 |
| `pw` | Pulse width | infinity |
| `per` | Period | infinity |

Example: `V1 1 0 PULSE(0 5 0 1n 1n 0.5m 1m)`

**SIN** -- sinusoidal:

```
SIN(vo va freq td theta)
```

| Parameter | Description | Default |
|---|---|---|
| `vo` | DC offset | required |
| `va` | Amplitude | required |
| `freq` | Frequency (Hz) | required |
| `td` | Delay time | 0 |
| `theta` | Damping factor | 0 |

Example: `V1 1 0 SIN(0 1 1MEG)`

**PWL** -- piecewise linear:

```
PWL(t1 v1 t2 v2 ... tn vn)
```

Specifies time-value pairs. The source linearly interpolates between them.

Example: `V1 1 0 PWL(0 0 1u 5 2u 5 3u 0)`

**EXP** -- exponential:

```
EXP(v1 v2 td1 tau1 td2 tau2)
```

| Parameter | Description | Default |
|---|---|---|
| `v1` | Initial value | required |
| `v2` | Target value | required |
| `td1` | Rise delay | 0 |
| `tau1` | Rise time constant | infinity |
| `td2` | Fall delay | infinity |
| `tau2` | Fall time constant | infinity |

Example: `V1 1 0 EXP(0 5 0 1u 5u 1u)`

## Semiconductor Devices

Semiconductor elements reference a named `.MODEL` statement.

### Diodes

```
Dname  anode  cathode  modelname
```

Model definition (Shockley diode):

```
.MODEL modelname D(IS=val N=val)
```

| Parameter | Description | Default |
|---|---|---|
| `IS` | Saturation current | 1e-14 |
| `N` | Emission coefficient | 1.0 |

Example:

```spice
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 2 0 DMOD
```

### BJTs (Bipolar Junction Transistors)

```
Qname  collector  base  emitter  modelname
```

Model definition (Ebers-Moll):

```
.MODEL modelname NPN(IS=val BF=val BR=val NF=val NR=val)
.MODEL modelname PNP(IS=val BF=val BR=val NF=val NR=val)
```

| Parameter | Description | Default |
|---|---|---|
| `IS` | Saturation current | 1e-16 |
| `BF` | Forward current gain | 100 |
| `BR` | Reverse current gain | 1 |
| `NF` | Forward emission coefficient | 1.0 |
| `NR` | Reverse emission coefficient | 1.0 |

Example:

```spice
.MODEL Q2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
Q1 vc base 0 Q2N2222
```

### MOSFETs

```
Mname  drain  gate  source  [bulk]  modelname
```

The bulk terminal is optional; Ohmnivore currently ignores it.

Model definition (Level 1, Shichman-Hodges):

```
.MODEL modelname NMOS(VTO=val KP=val LAMBDA=val)
.MODEL modelname PMOS(VTO=val KP=val LAMBDA=val)
```

| Parameter | Description | Default |
|---|---|---|
| `VTO` | Threshold voltage | 1.0 |
| `KP` | Transconductance parameter | 2e-5 |
| `LAMBDA` | Channel-length modulation | 0.0 |

Example:

```spice
.MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)
M1 drain gate 0 NMOD
```

## Analysis Commands

See [Analysis Types](analyses.md) for details and worked examples.

```spice
.DC                                   * DC operating point
.OP                                   * same as .DC
.AC DEC|OCT|LIN npoints fstart fstop  * AC frequency sweep
.TRAN tstep tstop [tstart] [UIC]      * Transient analysis
```
