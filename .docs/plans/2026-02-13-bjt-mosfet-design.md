# Non-Linear Elements: BJT and MOSFET with GPU-Resident Newton-Raphson

## Problem

Ohmnivore supports diodes as its only non-linear element. Real circuits use transistors — BJTs for analog amplifiers, MOSFETs for switching and digital logic. The existing Newton-Raphson infrastructure (companion models, GPU-resident iteration, descriptor-based stamping) extends naturally to these devices.

## Scope

**BJT:** Ebers-Moll Level 1 model, 3 terminals (collector, base, emitter). NPN and PNP variants. Parameters: IS, BF, BR, NF, NR.

**MOSFET:** Shichman-Hodges Level 1 model, 3 terminals (drain, gate, source). NMOS and PMOS variants. Parameters: VTO, KP, LAMBDA. Parser accepts a 4th bulk terminal for SPICE compatibility but ignores it at Level 1.

## Architecture

### Separate methods per device type

The `NonlinearBackend` trait gains new methods alongside the existing diode ones. Each device type gets its own evaluation shader, assembly shader, and descriptor struct. The Newton loop calls each evaluator in sequence.

BJTs have two coupled junctions; MOSFETs have region-dependent drain current. These physics differ enough that a unified shader would add complexity without benefit. Separate shaders keep each device self-contained and independently testable.

This choice does not affect multi-GPU/multi-node scalability. Device evaluation is embarrassingly parallel — descriptor buffers shard across GPUs regardless of whether methods are separate or unified. Separate methods simplify per-device-type load balancing.

### Polarity handling

NPN/PNP and NMOS/PMOS differ only in current direction and voltage polarity. Each descriptor carries a `polarity` field (+1.0 or -1.0). The shader multiplies terminal voltages by polarity, evaluates as NPN/NMOS, then multiplies output currents by polarity. Zero branching — just arithmetic.

### Region selection

MOSFETs have three operating regions (cutoff, linear, saturation). BJTs have four (forward active, reverse active, saturation, cutoff). The shader must branch per device to select the correct equations. This branching is unavoidable for any transistor model, but the penalty is small: branches are short (a few multiplies each), and within a converged circuit most devices of the same type occupy the same region.

### Shader language

All shaders target WGSL for the existing `WgpuBackend`. The shader logic is backend-agnostic — porting to CUDA/ROCm kernels requires reimplementing the same math in a different language. The `NonlinearBackend` trait insulates the Newton loop from the backend.

## BJT Model: Ebers-Moll

### Equations

Two coupled diodes with current gain:

```
I_f = IS * (exp(V_BE / (NF * V_T)) - 1)     // forward diode
I_r = IS * (exp(V_BC / (NR * V_T)) - 1)     // reverse diode
I_C = BF/(BF+1) * I_f - I_r/(BR+1)          // collector current
I_B = I_f/(BF+1) + I_r/(BR+1)               // base current
I_E = -(I_C + I_B)                           // emitter current (KCL)
```

### Companion model

Each Newton iteration linearizes the BJT into conductances and current sources. The Jacobian has four independent partial derivatives:

- `dI_C/dV_BE`, `dI_C/dV_BC`
- `dI_B/dV_BE`, `dI_B/dV_BC`

The emitter row derives from KCL: `dI_E/dV_x = -(dI_C/dV_x + dI_B/dV_x)`. This produces a 3x3 stamp (9 G-matrix positions) plus 3 RHS entries.

### Model parameters

```spice
.MODEL 2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
.MODEL MPNP1 PNP(IS=1e-14 BF=100)
```

| Param | Description | Default |
|-------|-------------|---------|
| IS | Saturation current | 1e-16 A |
| BF | Forward current gain | 100 |
| BR | Reverse current gain | 1 |
| NF | Forward emission coefficient | 1.0 |
| NR | Reverse emission coefficient | 1.0 |

### Voltage limiting

Clamp V_BE changes to `+/-2*NF*V_T` and V_BC changes to `+/-2*NR*V_T` per iteration, matching the existing diode limiting strategy.

## MOSFET Model: Shichman-Hodges

### Equations

Region-dependent drain current:

**Cutoff** (`V_GS < VTO`):
```
I_D = 0
```

**Linear** (`V_GS >= VTO` and `V_DS < V_GS - VTO`):
```
I_D = KP * ((V_GS - VTO) * V_DS - V_DS^2/2) * (1 + LAMBDA * V_DS)
```

**Saturation** (`V_GS >= VTO` and `V_DS >= V_GS - VTO`):
```
I_D = KP/2 * (V_GS - VTO)^2 * (1 + LAMBDA * V_DS)
```

### Companion model

The gate draws no DC current. The Jacobian has two entries:

- `g_m = dI_D/dV_GS` (transconductance)
- `g_ds = dI_D/dV_DS` (output conductance)

Since only I_D flows (drain to source), the stamp covers 2 rows x 3 columns (drain and source rows; drain, gate, and source columns) — 6 G-matrix positions plus 2 RHS entries. No gate row.

### Model parameters

```spice
.MODEL NMOS1 NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)
.MODEL PMOS1 PMOS(VTO=0.7 KP=4.5e-5 LAMBDA=0.05)
```

| Param | Description | Default |
|-------|-------------|---------|
| VTO | Threshold voltage | 1.0 V |
| KP | Transconductance parameter | 2e-5 A/V^2 |
| LAMBDA | Channel-length modulation | 0.0 V^-1 |

### Voltage limiting

Clamp V_GS and V_DS changes to `+/-0.5V` per iteration, the standard SPICE MOSFET limit.

## Data Model

### IR additions

```rust
// ir.rs
pub enum Component {
    // ... existing variants unchanged ...
    Bjt { name: String, nodes: [String; 3], model: String },
    Mosfet { name: String, nodes: [String; 3], model: String },
}

pub struct BjtModel {
    pub name: String,
    pub is: f64,      // default 1e-16
    pub bf: f64,      // default 100
    pub br: f64,      // default 1
    pub nf: f64,      // default 1.0
    pub nr: f64,      // default 1.0
    pub is_npn: bool, // true = NPN, false = PNP
}

pub struct MosfetModel {
    pub name: String,
    pub vto: f64,      // default 1.0
    pub kp: f64,       // default 2e-5
    pub lambda: f64,   // default 0.0
    pub is_nmos: bool, // true = NMOS, false = PMOS
}
```

The `Circuit` struct gains `bjt_models: Vec<BjtModel>` and `mosfet_models: Vec<MosfetModel>` alongside the existing `models: Vec<DiodeModel>`.

### GPU descriptors

**BJT (96 bytes):**

```rust
pub struct GpuBjtDescriptor {
    pub collector_idx: u32,    // node index (u32::MAX = ground)
    pub base_idx: u32,
    pub emitter_idx: u32,
    pub polarity: f32,         // +1.0 NPN, -1.0 PNP
    pub is_val: f32,
    pub bf: f32,
    pub br: f32,
    pub nf_vt: f32,            // NF * V_T
    pub nr_vt: f32,            // NR * V_T
    pub _padding: [u32; 3],
    pub g_row_col: [u32; 9],   // CSR indices: [CC,CB,CE, BC,BB,BE, EC,EB,EE]
    pub b_idx: [u32; 3],       // RHS indices: [C, B, E]
}
```

**MOSFET (64 bytes):**

```rust
pub struct GpuMosfetDescriptor {
    pub drain_idx: u32,
    pub gate_idx: u32,
    pub source_idx: u32,
    pub polarity: f32,         // +1.0 NMOS, -1.0 PMOS
    pub vto: f32,
    pub kp: f32,
    pub lambda: f32,
    pub _padding: u32,
    pub g_row_col: [u32; 6],   // CSR indices: [DD,DG,DS, SD,SG,SS]
    pub b_idx: [u32; 2],       // RHS indices: [D, S]
}
```

Ground-connected terminals use `u32::MAX` to skip stamping, same as diodes.

### SPICE syntax

```spice
* BJT
Q1 collector base emitter 2N2222

* MOSFET (3 or 4 terminals — bulk ignored at Level 1)
M1 drain gate source NMOS1
M2 drain gate source bulk PMOS1

* Models
.MODEL 2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
.MODEL NMOS1 NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)
```

## Compute Shaders

### BJT evaluation (`bjt_eval`, one invocation per BJT)

```wgsl
let v_be = polarity * (x[desc.base_idx] - x[desc.emitter_idx]);
let v_bc = polarity * (x[desc.base_idx] - x[desc.collector_idx]);
let e_f = exp(clamp(v_be / desc.nf_vt, -80.0, 80.0));
let e_r = exp(clamp(v_bc / desc.nr_vt, -80.0, 80.0));
let i_f = desc.is * (e_f - 1.0);
let i_r = desc.is * (e_r - 1.0);
let i_c = desc.bf / (desc.bf + 1.0) * i_f - i_r / (desc.br + 1.0);
let i_b = i_f / (desc.bf + 1.0) + i_r / (desc.br + 1.0);
// Jacobian: 4 independent values, expand to 3x3 via KCL in assembly
let di_c_dvbe = desc.bf / (desc.bf + 1.0) * desc.is * e_f / desc.nf_vt;
let di_c_dvbc = -desc.is * e_r / desc.nr_vt / (desc.br + 1.0);
let di_b_dvbe = desc.is * e_f / desc.nf_vt / (desc.bf + 1.0);
let di_b_dvbc = desc.is * e_r / desc.nr_vt / (desc.br + 1.0);
```

Output buffer: `[I_C, I_B, dI_C/dV_BE, dI_C/dV_BC, dI_B/dV_BE, dI_B/dV_BC]` per BJT (6 values).

### MOSFET evaluation (`mosfet_eval`, one invocation per MOSFET)

```wgsl
let v_gs = polarity * (x[desc.gate_idx] - x[desc.source_idx]);
let v_ds = polarity * (x[desc.drain_idx] - x[desc.source_idx]);
var i_d: f32 = 0.0;
var g_m: f32 = 0.0;
var g_ds: f32 = 0.0;

let v_ov = v_gs - desc.vto;
if v_ov > 0.0 {
    if v_ds < v_ov {
        // Linear region
        let lam = 1.0 + desc.lambda * v_ds;
        i_d = desc.kp * (v_ov * v_ds - v_ds * v_ds / 2.0) * lam;
        g_m = desc.kp * v_ds * lam;
        g_ds = desc.kp * (v_ov - v_ds) * lam
             + desc.kp * (v_ov * v_ds - v_ds * v_ds / 2.0) * desc.lambda;
    } else {
        // Saturation region
        let lam = 1.0 + desc.lambda * v_ds;
        i_d = desc.kp / 2.0 * v_ov * v_ov * lam;
        g_m = desc.kp * v_ov * lam;
        g_ds = desc.kp / 2.0 * v_ov * v_ov * desc.lambda;
    }
}
// Multiply by polarity for PMOS
i_d = polarity * i_d;
```

Output buffer: `[I_D, g_m, g_ds, polarity]` per MOSFET (4 values).

### Assembly shaders

Same structure as existing `assemble_matrix_stamp` / `assemble_rhs_stamp`, one pair per device type.

**BJT assembly:** Reads the 6-value eval output, expands the 4 Jacobian derivatives to 9 G-matrix entries using KCL (`dI_E/dV_x = -(dI_C/dV_x + dI_B/dV_x)`). Stamps via precomputed `g_row_col`. RHS stamps Norton companion currents: `I_node - sum(g_node_x * V_x)`.

**MOSFET assembly:** Reads the 4-value eval output, stamps 6 G-matrix entries and 2 RHS entries via precomputed indices. Drain row gets `+g_m, +g_ds`; source row gets `-g_m, -g_ds`.

### Convergence shaders

Two new voltage-limiting entry points:

- `bjt_voltage_limit`: clamps V_BE change to `+/-2*NF*V_T`, V_BC change to `+/-2*NR*V_T`
- `mosfet_voltage_limit`: clamps V_GS and V_DS changes to `+/-0.5V`

The existing `convergence_check` shader is unchanged — it computes `max|x_new - x_old|` over the entire solution vector, which covers all device contributions.

## Newton Loop Changes

```
for iter in 0..max_iterations {
    1. Copy x → x_old
    2. Evaluate all device types (independent dispatches):
       - evaluate_diodes(x, diode_descriptors)     → diode_output
       - evaluate_bjts(x, bjt_descriptors)          → bjt_output
       - evaluate_mosfets(x, mosfet_descriptors)    → mosfet_output
    3. Assemble system:
       - Copy base G and b
       - Stamp diode contributions
       - Stamp BJT contributions
       - Stamp MOSFET contributions
    4. Solve linear system (BiCGSTAB)
    5. Voltage limiting per device type
    6. Convergence check (unified across all nodes)
}
```

Step 2 launches independent GPU kernels. On a single GPU they run sequentially; the pattern extends to multi-GPU by sharding descriptors across devices.

Step 3 must complete for all device types before step 4 — all stamps go into the same G matrix and RHS vector.

## NonlinearBackend Trait Changes

```rust
pub trait NonlinearBackend: SolverBackend {
    // Existing (unchanged)
    fn evaluate_diodes(...);
    fn assemble_nonlinear_system(...);
    fn limit_and_check_convergence(...);

    // New
    fn evaluate_bjts(
        &self, x_buf: &Self::Buffer, descriptors: &Self::Buffer, output: &Self::Buffer,
    );
    fn evaluate_mosfets(
        &self, x_buf: &Self::Buffer, descriptors: &Self::Buffer, output: &Self::Buffer,
    );
    fn assemble_bjt_stamps(
        &self, bjt_output: &Self::Buffer, descriptors: &Self::Buffer,
        g_out: &GpuCsrMatrix, b_out: &Self::Buffer,
    );
    fn assemble_mosfet_stamps(
        &self, mosfet_output: &Self::Buffer, descriptors: &Self::Buffer,
        g_out: &GpuCsrMatrix, b_out: &Self::Buffer,
    );
}
```

## Compiler Changes

`MnaSystem` gains two new fields:

```rust
pub struct MnaSystem {
    // ... existing fields ...
    pub bjt_descriptors: Vec<GpuBjtDescriptor>,
    pub mosfet_descriptors: Vec<GpuMosfetDescriptor>,
}
```

BJTs stamp 9 placeholder zeros into G (the full 3x3 block). MOSFETs stamp 6 placeholder zeros (2 rows x 3 columns, no gate row). After CSR finalization, the compiler resolves each position to a CSR value index, same as existing diode descriptor construction.

Nonlinear detection in `analysis/dc.rs` broadens to check all three descriptor vectors.

## Integration

### Unchanged
- `solver/bicgstab.rs` — reused inside Newton loop
- `solver/backend.rs` — `SolverBackend` trait
- `solver/preconditioner.rs` — ISAI preconditioner
- `sparse.rs`, `analysis/ac.rs`

### Extended
- `ir.rs` — `Bjt` and `Mosfet` variants, `BjtModel`, `MosfetModel`
- `parser.rs` — `Q` and `M` elements, `.MODEL NPN/PNP/NMOS/PMOS(...)` directives
- `compiler.rs` — placeholder stamping and descriptor construction for BJTs and MOSFETs
- `solver/newton.rs` — dispatch evaluate/assemble/limit for all three device types
- `solver/nonlinear.rs` — new trait methods and `WgpuBackend` implementations
- `solver/gpu_shaders.rs` — new WGSL shader source strings
- `analysis/dc.rs` — nonlinear detection includes BJT and MOSFET descriptors

## Testing

### Per-device circuits (validated against ngspice)
- NPN common-emitter with base resistor — verify V_CE, I_C
- PNP current mirror — confirm polarity handling
- BJT in saturation (both junctions forward-biased)
- BJT in cutoff (both junctions reverse-biased)
- NMOS common-source with gate bias — verify I_D in saturation
- PMOS equivalent — confirm polarity handling
- MOSFET in linear region (low V_DS)
- MOSFET in cutoff (V_GS < VTO)

### Mixed circuits
- Circuit combining diodes, BJTs, and MOSFETs — all three stamp and solve together

### Convergence edge cases
- BJT with high BF (sensitive to V_BE — tests voltage limiting)
- MOSFET at region boundaries (linear/saturation transition)

### Parser tests
- Q and M element lines, model parameter parsing with defaults
- 4-terminal MOSFET syntax (bulk node accepted, ignored)
- Error cases: undefined model, wrong model type for element

### Validation approach
Generate reference netlists and compare node voltages and branch currents against ngspice output in integration tests.

## Future Work

1. Upgrade BJT to Gummel-Poon model (4th substrate terminal, base resistance, charge storage)
2. Upgrade MOSFET to Level 2/3 (4th bulk terminal, body effect GAMMA/PHI, subthreshold)
3. Add JFET non-linear element (NJF/PJF)
4. Advanced convergence aids (source stepping, Gmin stepping, continuation methods)
5. CUDA and ROCm implementations of the new trait methods
