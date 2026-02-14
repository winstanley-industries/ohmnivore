//! WGSL compute shader sources for GPU-accelerated linear algebra.
//!
//! All shaders use f32 for real values and vec2<f32> for complex values
//! (where .x = real, .y = imaginary). The CSR matrix format matches
//! `crate::sparse::CsrMatrix` but with u32 indices and f32/vec2<f32> values.

/// WGSL source containing all GPU compute kernels.
///
/// Entry points:
/// - `spmv_real`: y = A * x (real f32, CSR format)
/// - `spmv_complex`: y = A * x (complex vec2<f32>, CSR format)
/// - `axpy_real`: y = alpha * x + y (real)
/// - `axpy_complex`: y = alpha * x + y (complex)
/// - `dot_real`: partial dot product reduction (real)
/// - `dot_complex`: partial dot product reduction (complex, conjugate dot)
/// - `scale_real`: x = alpha * x (real)
/// - `scale_complex`: x = alpha * x (complex)
/// - `copy_real`: y = x (real)
/// - `copy_complex`: y = x (complex)
/// - `subtract_real`: z = x - y (real)
/// - `subtract_complex`: z = x - y (complex)
/// - `jacobi_real`: z = inv_diag * r (real, element-wise)
/// - `jacobi_complex`: z = inv_diag * r (complex, element-wise multiply)
pub const SHADER_SOURCE: &str = r#"
// ============================================================
// Ohmnivore GPU Compute Shaders
// ============================================================

// --- SpMV (real) ---
// y = A * x, where A is in CSR format.
// One thread per row.

struct SpmvParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> spmv_values: array<f32>;
@group(0) @binding(1) var<storage, read> spmv_col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> spmv_row_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> spmv_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> spmv_y: array<f32>;
@group(0) @binding(5) var<uniform> spmv_params: SpmvParams;

@compute @workgroup_size(64)
fn spmv_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= spmv_params.n {
        return;
    }
    let row_start = spmv_row_pointers[row];
    let row_end = spmv_row_pointers[row + 1u];
    var sum: f32 = 0.0;
    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        sum = sum + spmv_values[idx] * spmv_x[spmv_col_indices[idx]];
    }
    spmv_y[row] = sum;
}

// --- SpMV (complex) ---
// y = A * x, complex multiply: (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)

struct SpmvComplexParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> spmv_c_values: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> spmv_c_col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> spmv_c_row_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> spmv_c_x: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> spmv_c_y: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> spmv_c_params: SpmvComplexParams;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn spmv_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= spmv_c_params.n {
        return;
    }
    let row_start = spmv_c_row_pointers[row];
    let row_end = spmv_c_row_pointers[row + 1u];
    var sum = vec2<f32>(0.0, 0.0);
    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        sum = sum + complex_mul(spmv_c_values[idx], spmv_c_x[spmv_c_col_indices[idx]]);
    }
    spmv_c_y[row] = sum;
}

// --- Vector operations: shared parameter struct ---

struct VecParams {
    alpha_re: f32,
    alpha_im: f32,
    n: u32,
}

// --- axpy (real): y = alpha * x + y ---

@group(0) @binding(0) var<storage, read> axpy_r_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> axpy_r_y: array<f32>;
@group(0) @binding(2) var<uniform> axpy_r_params: VecParams;

@compute @workgroup_size(64)
fn axpy_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_r_params.n {
        return;
    }
    axpy_r_y[i] = axpy_r_params.alpha_re * axpy_r_x[i] + axpy_r_y[i];
}

// --- axpy (complex): y = alpha * x + y ---

@group(0) @binding(0) var<storage, read> axpy_c_x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> axpy_c_y: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> axpy_c_params: VecParams;

@compute @workgroup_size(64)
fn axpy_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_c_params.n {
        return;
    }
    let alpha = vec2<f32>(axpy_c_params.alpha_re, axpy_c_params.alpha_im);
    axpy_c_y[i] = complex_mul(alpha, axpy_c_x[i]) + axpy_c_y[i];
}

// --- dot product (real): parallel reduction ---
// Each workgroup reduces a chunk, writes partial sum to output.
// Final reduction across workgroups done on CPU.

const DOT_WG_SIZE: u32 = 64u;

var<workgroup> dot_scratch: array<f32, 64>;

@group(0) @binding(0) var<storage, read> dot_r_x: array<f32>;
@group(0) @binding(1) var<storage, read> dot_r_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> dot_r_out: array<f32>;
@group(0) @binding(3) var<uniform> dot_r_params: VecParams;

@compute @workgroup_size(64)
fn dot_real(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    // Each thread loads its element (or 0 if out of bounds)
    if i < dot_r_params.n {
        dot_scratch[local_id] = dot_r_x[i] * dot_r_y[i];
    } else {
        dot_scratch[local_id] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction within the workgroup
    var stride = DOT_WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            dot_scratch[local_id] = dot_scratch[local_id] + dot_scratch[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes workgroup result
    if local_id == 0u {
        dot_r_out[wid.x] = dot_scratch[0];
    }
}

// --- dot product (complex): conjugate dot product ---
// result = sum( conj(x[i]) * y[i] )
// conj(a) * b = (a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x)

var<workgroup> dot_c_scratch_re: array<f32, 64>;
var<workgroup> dot_c_scratch_im: array<f32, 64>;

@group(0) @binding(0) var<storage, read> dot_c_x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> dot_c_y: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dot_c_out: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dot_c_params: VecParams;

@compute @workgroup_size(64)
fn dot_complex(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    if i < dot_c_params.n {
        let a = dot_c_x[i]; // will be conjugated
        let b = dot_c_y[i];
        // conj(a) * b
        dot_c_scratch_re[local_id] = a.x * b.x + a.y * b.y;
        dot_c_scratch_im[local_id] = a.x * b.y - a.y * b.x;
    } else {
        dot_c_scratch_re[local_id] = 0.0;
        dot_c_scratch_im[local_id] = 0.0;
    }
    workgroupBarrier();

    var stride = DOT_WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            dot_c_scratch_re[local_id] = dot_c_scratch_re[local_id] + dot_c_scratch_re[local_id + stride];
            dot_c_scratch_im[local_id] = dot_c_scratch_im[local_id] + dot_c_scratch_im[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if local_id == 0u {
        dot_c_out[wid.x] = vec2<f32>(dot_c_scratch_re[0], dot_c_scratch_im[0]);
    }
}

// --- scale (real): x = alpha * x ---

@group(0) @binding(0) var<storage, read_write> scale_r_x: array<f32>;
@group(0) @binding(1) var<uniform> scale_r_params: VecParams;

@compute @workgroup_size(64)
fn scale_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= scale_r_params.n {
        return;
    }
    scale_r_x[i] = scale_r_params.alpha_re * scale_r_x[i];
}

// --- scale (complex): x = alpha * x ---

@group(0) @binding(0) var<storage, read_write> scale_c_x: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> scale_c_params: VecParams;

@compute @workgroup_size(64)
fn scale_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= scale_c_params.n {
        return;
    }
    let alpha = vec2<f32>(scale_c_params.alpha_re, scale_c_params.alpha_im);
    scale_c_x[i] = complex_mul(alpha, scale_c_x[i]);
}

// --- copy (real): y = x ---

@group(0) @binding(0) var<storage, read> copy_r_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_r_y: array<f32>;
@group(0) @binding(2) var<uniform> copy_r_params: VecParams;

@compute @workgroup_size(64)
fn copy_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= copy_r_params.n {
        return;
    }
    copy_r_y[i] = copy_r_x[i];
}

// --- copy (complex): y = x ---

@group(0) @binding(0) var<storage, read> copy_c_x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> copy_c_y: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> copy_c_params: VecParams;

@compute @workgroup_size(64)
fn copy_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= copy_c_params.n {
        return;
    }
    copy_c_y[i] = copy_c_x[i];
}

// --- subtract (real): z = x - y ---

@group(0) @binding(0) var<storage, read> sub_r_x: array<f32>;
@group(0) @binding(1) var<storage, read> sub_r_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> sub_r_z: array<f32>;
@group(0) @binding(3) var<uniform> sub_r_params: VecParams;

@compute @workgroup_size(64)
fn subtract_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sub_r_params.n {
        return;
    }
    sub_r_z[i] = sub_r_x[i] - sub_r_y[i];
}

// --- subtract (complex): z = x - y ---

@group(0) @binding(0) var<storage, read> sub_c_x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> sub_c_y: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> sub_c_z: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> sub_c_params: VecParams;

@compute @workgroup_size(64)
fn subtract_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sub_c_params.n {
        return;
    }
    sub_c_z[i] = sub_c_x[i] - sub_c_y[i];
}

// --- Jacobi preconditioner apply (real): z[i] = inv_diag[i] * r[i] ---

@group(0) @binding(0) var<storage, read> jac_r_inv_diag: array<f32>;
@group(0) @binding(1) var<storage, read> jac_r_r: array<f32>;
@group(0) @binding(2) var<storage, read_write> jac_r_z: array<f32>;
@group(0) @binding(3) var<uniform> jac_r_params: VecParams;

@compute @workgroup_size(64)
fn jacobi_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= jac_r_params.n {
        return;
    }
    jac_r_z[i] = jac_r_inv_diag[i] * jac_r_r[i];
}

// --- Jacobi preconditioner apply (complex): z[i] = inv_diag[i] * r[i] ---

@group(0) @binding(0) var<storage, read> jac_c_inv_diag: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> jac_c_r: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> jac_c_z: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> jac_c_params: VecParams;

@compute @workgroup_size(64)
fn jacobi_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= jac_c_params.n {
        return;
    }
    jac_c_z[i] = complex_mul(jac_c_inv_diag[i], jac_c_r[i]);
}
"#;

/// WGSL shader for evaluating the Shockley diode equation on GPU.
///
/// One invocation per diode. Reads the solution vector and diode descriptors,
/// computes diode current (i_d) and conductance (g_d) using the exponential
/// model with clamping for numerical stability.
///
/// Descriptor layout per diode (flat u32 buffer, 12 words per diode):
///   [0] anode_idx, [1] cathode_idx, [2] is_val (bitcast f32),
///   [3] n_vt (bitcast f32), [4..7] g_row_col (4 CSR positions),
///   [8..9] b_idx (2 RHS indices), [10..11] padding
///
/// Output: 2 floats per diode in eval_output: [2*i] = i_d, [2*i+1] = g_d
pub const DIODE_EVAL_SHADER: &str = r#"
// ============================================================
// Diode Evaluation Shader
// ============================================================
// Computes Shockley diode model: i_d = Is*(exp(Vd/nVt) - 1)

struct DiodeEvalParams {
    num_diodes: u32,
}

// Diode descriptors: flat array of u32, 12 words per diode
@group(0) @binding(0) var<storage, read> diode_desc: array<u32>;
// Solution vector x
@group(0) @binding(1) var<storage, read> eval_x: array<f32>;
// Output: 2 floats per diode [i_d, g_d]
@group(0) @binding(2) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(3) var<uniform> eval_params: DiodeEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

// Read voltage at a node index, returning 0.0 for ground (sentinel).
fn read_voltage(idx: u32) -> f32 {
    if idx == SENTINEL {
        return 0.0;
    }
    return eval_x[idx];
}

@compute @workgroup_size(64)
fn diode_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= eval_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = diode_desc[base + 0u];
    let cathode_idx = diode_desc[base + 1u];
    let is_val = bitcast<f32>(diode_desc[base + 2u]);
    let n_vt = bitcast<f32>(diode_desc[base + 3u]);

    let v_d = read_voltage(anode_idx) - read_voltage(cathode_idx);

    // Clamp exponent to [-80, 80] to avoid inf/nan from exp()
    let exponent = clamp(v_d / n_vt, -80.0, 80.0);
    let e = exp(exponent);

    let i_d = is_val * (e - 1.0);
    let g_d = is_val * e / n_vt;

    eval_output[2u * i] = i_d;
    eval_output[2u * i + 1u] = g_d;
}
"#;

/// WGSL shader for assembling the nonlinear contributions into MNA matrix and RHS.
///
/// Two entry points:
/// - `nonlinear_assemble_matrix`: Copies base G values then stamps diode conductance.
///   First pass (one invocation per NNZ): copy base values.
///   Second pass (one invocation per diode): add g_d stamps to 4 CSR positions.
///
/// - `nonlinear_assemble_rhs`: Copies base b values then stamps Norton companion current.
///   First pass (one invocation per RHS element): copy base values.
///   Second pass (one invocation per diode): apply Norton equivalent stamps.
pub const NONLINEAR_ASSEMBLE_SHADER: &str = r#"
// ============================================================
// Nonlinear Assembly Shader
// ============================================================

struct AssembleMatParams {
    nnz: u32,
}

// --- Matrix base copy: out_values = base_values ---

@group(0) @binding(0) var<storage, read> base_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_values: array<f32>;
@group(0) @binding(2) var<uniform> assemble_mat_params: AssembleMatParams;

@compute @workgroup_size(64)
fn assemble_matrix_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= assemble_mat_params.nnz {
        return;
    }
    out_values[i] = base_values[i];
}

// --- Matrix stamp: add g_d into 4 precomputed CSR positions per diode ---

struct StampMatParams {
    num_diodes: u32,
}

// Diode descriptors (same layout as eval shader)
@group(0) @binding(0) var<storage, read> stamp_desc: array<u32>;
// Diode eval results: [i_d, g_d] per diode
@group(0) @binding(1) var<storage, read> stamp_eval: array<f32>;
// CSR values buffer (read_write for stamping)
@group(0) @binding(2) var<storage, read_write> stamp_values: array<f32>;
@group(0) @binding(3) var<uniform> stamp_mat_params: StampMatParams;

const STAMP_SENTINEL: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn assemble_matrix_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= stamp_mat_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let g_d = stamp_eval[2u * i + 1u];

    // 4 CSR positions for the conductance stamp:
    // G[anode,anode] += g_d, G[anode,cathode] -= g_d
    // G[cathode,anode] -= g_d, G[cathode,cathode] += g_d
    // Positions are SENTINEL when the corresponding node is ground.
    let pos0 = stamp_desc[base + 4u];
    let pos1 = stamp_desc[base + 5u];
    let pos2 = stamp_desc[base + 6u];
    let pos3 = stamp_desc[base + 7u];

    if pos0 != STAMP_SENTINEL {
        stamp_values[pos0] = stamp_values[pos0] + g_d;    // [anode, anode]
    }
    if pos1 != STAMP_SENTINEL {
        stamp_values[pos1] = stamp_values[pos1] - g_d;    // [anode, cathode]
    }
    if pos2 != STAMP_SENTINEL {
        stamp_values[pos2] = stamp_values[pos2] - g_d;    // [cathode, anode]
    }
    if pos3 != STAMP_SENTINEL {
        stamp_values[pos3] = stamp_values[pos3] + g_d;    // [cathode, cathode]
    }
}

// --- RHS base copy: out_b = base_b ---

struct AssembleRhsParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> base_b: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(2) var<uniform> assemble_rhs_params: AssembleRhsParams;

@compute @workgroup_size(64)
fn assemble_rhs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= assemble_rhs_params.n {
        return;
    }
    out_b[i] = base_b[i];
}

// --- RHS stamp: apply Norton companion current per diode ---

struct StampRhsParams {
    num_diodes: u32,
}

// Diode descriptors
@group(0) @binding(0) var<storage, read> rhs_desc: array<u32>;
// Solution vector (for v_d calculation)
@group(0) @binding(1) var<storage, read> rhs_x: array<f32>;
// Diode eval results: [i_d, g_d] per diode
@group(0) @binding(2) var<storage, read> rhs_eval: array<f32>;
// RHS vector (read_write for stamping)
@group(0) @binding(3) var<storage, read_write> rhs_b: array<f32>;
@group(0) @binding(4) var<uniform> stamp_rhs_params: StampRhsParams;

const RHS_SENTINEL: u32 = 0xFFFFFFFFu;

fn rhs_read_voltage(idx: u32) -> f32 {
    if idx == RHS_SENTINEL {
        return 0.0;
    }
    return rhs_x[idx];
}

@compute @workgroup_size(64)
fn assemble_rhs_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= stamp_rhs_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = rhs_desc[base + 0u];
    let cathode_idx = rhs_desc[base + 1u];

    let i_d = rhs_eval[2u * i];
    let g_d = rhs_eval[2u * i + 1u];

    let v_d = rhs_read_voltage(anode_idx) - rhs_read_voltage(cathode_idx);

    // Norton companion: i_eq = i_d - g_d * v_d
    let i_eq = i_d - g_d * v_d;

    // RHS indices from descriptor (SENTINEL for ground nodes)
    let b_idx_anode = rhs_desc[base + 8u];
    let b_idx_cathode = rhs_desc[base + 9u];

    // b[anode] -= i_eq, b[cathode] += i_eq
    if b_idx_anode != RHS_SENTINEL {
        rhs_b[b_idx_anode] = rhs_b[b_idx_anode] - i_eq;
    }
    if b_idx_cathode != RHS_SENTINEL {
        rhs_b[b_idx_cathode] = rhs_b[b_idx_cathode] + i_eq;
    }
}
"#;

/// WGSL shader for evaluating the Ebers-Moll BJT model on GPU.
///
/// One invocation per BJT. Reads the solution vector and BJT descriptors,
/// computes collector/base currents and four Jacobian derivatives.
///
/// Descriptor layout per BJT (flat u32 buffer, 24 words per BJT):
///   [0] collector_idx, [1] base_idx, [2] emitter_idx,
///   [3] polarity (bitcast f32), [4] is_val (bitcast f32), [5] bf (bitcast f32),
///   [6] br (bitcast f32), [7] nf_vt (bitcast f32), [8] nr_vt (bitcast f32),
///   [9..11] padding, [12..20] g_row_col (9 CSR indices), [21..23] b_idx (3 RHS indices)
///
/// Output: 6 floats per BJT in eval_output:
///   [I_C, I_B, dI_C/dV_BE, dI_C/dV_BC, dI_B/dV_BE, dI_B/dV_BC]
pub const BJT_EVAL_SHADER: &str = r#"
// ============================================================
// BJT Evaluation Shader (Ebers-Moll Level 1)
// ============================================================

struct BjtEvalParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_desc: array<u32>;
@group(0) @binding(1) var<storage, read> eval_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(3) var<uniform> eval_params: BjtEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn read_voltage(idx: u32) -> f32 {
    if idx == SENTINEL {
        return 0.0;
    }
    return eval_x[idx];
}

@compute @workgroup_size(64)
fn bjt_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= eval_params.num_bjts {
        return;
    }

    let base = i * 24u;
    let collector_idx = bjt_desc[base + 0u];
    let base_idx = bjt_desc[base + 1u];
    let emitter_idx = bjt_desc[base + 2u];
    let polarity = bitcast<f32>(bjt_desc[base + 3u]);
    let is_val = bitcast<f32>(bjt_desc[base + 4u]);
    let bf = bitcast<f32>(bjt_desc[base + 5u]);
    let br = bitcast<f32>(bjt_desc[base + 6u]);
    let nf_vt = bitcast<f32>(bjt_desc[base + 7u]);
    let nr_vt = bitcast<f32>(bjt_desc[base + 8u]);

    let v_be = polarity * (read_voltage(base_idx) - read_voltage(emitter_idx));
    let v_bc = polarity * (read_voltage(base_idx) - read_voltage(collector_idx));

    let e_f = exp(clamp(v_be / nf_vt, -80.0, 80.0));
    let e_r = exp(clamp(v_bc / nr_vt, -80.0, 80.0));

    let i_f = is_val * (e_f - 1.0);
    let i_r = is_val * (e_r - 1.0);

    let i_c = bf / (bf + 1.0) * i_f - i_r / (br + 1.0);
    let i_b = i_f / (bf + 1.0) + i_r / (br + 1.0);

    // Jacobian derivatives
    let di_c_dvbe = bf / (bf + 1.0) * is_val * e_f / nf_vt;
    let di_c_dvbc = -is_val * e_r / nr_vt / (br + 1.0);
    let di_b_dvbe = is_val * e_f / nf_vt / (bf + 1.0);
    let di_b_dvbc = is_val * e_r / nr_vt / (br + 1.0);

    let out = 6u * i;
    eval_output[out + 0u] = i_c * polarity;
    eval_output[out + 1u] = i_b * polarity;
    eval_output[out + 2u] = di_c_dvbe;
    eval_output[out + 3u] = di_c_dvbc;
    eval_output[out + 4u] = di_b_dvbe;
    eval_output[out + 5u] = di_b_dvbc;
}
"#;

/// WGSL shader for evaluating the Shichman-Hodges MOSFET model on GPU.
///
/// One invocation per MOSFET. Computes drain current and partial derivatives
/// (g_m, g_ds) with region selection (cutoff/linear/saturation).
///
/// Descriptor layout per MOSFET (flat u32 buffer, 16 words per MOSFET):
///   [0] drain_idx, [1] gate_idx, [2] source_idx,
///   [3] polarity (bitcast f32), [4] vto (bitcast f32), [5] kp (bitcast f32),
///   [6] lambda (bitcast f32), [7] padding,
///   [8..13] g_row_col (6 CSR indices), [14..15] b_idx (2 RHS indices)
///
/// Output: 4 floats per MOSFET in eval_output: [I_D, g_m, g_ds, polarity]
pub const MOSFET_EVAL_SHADER: &str = r#"
// ============================================================
// MOSFET Evaluation Shader (Shichman-Hodges Level 1)
// ============================================================

struct MosfetEvalParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_desc: array<u32>;
@group(0) @binding(1) var<storage, read> eval_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(3) var<uniform> eval_params: MosfetEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn read_voltage(idx: u32) -> f32 {
    if idx == SENTINEL {
        return 0.0;
    }
    return eval_x[idx];
}

@compute @workgroup_size(64)
fn mosfet_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= eval_params.num_mosfets {
        return;
    }

    let base = i * 16u;
    let drain_idx = mosfet_desc[base + 0u];
    let gate_idx = mosfet_desc[base + 1u];
    let source_idx = mosfet_desc[base + 2u];
    let polarity = bitcast<f32>(mosfet_desc[base + 3u]);
    let vto = bitcast<f32>(mosfet_desc[base + 4u]);
    let kp = bitcast<f32>(mosfet_desc[base + 5u]);
    let lambda = bitcast<f32>(mosfet_desc[base + 6u]);

    let v_gs = polarity * (read_voltage(gate_idx) - read_voltage(source_idx));
    let v_ds = polarity * (read_voltage(drain_idx) - read_voltage(source_idx));

    var i_d: f32 = 0.0;
    var g_m: f32 = 0.0;
    var g_ds: f32 = 0.0;

    // Smooth subthreshold transition using softplus function.
    // Replaces hard cutoff at v_ov=0 with a smooth curve that:
    // - equals v_ov when v_ov >> VT (active region unchanged)
    // - decays exponentially to 0 when v_ov << -VT (cutoff)
    // - transitions smoothly near v_ov=0 (prevents Newton oscillation)
    let VT: f32 = 0.02585;
    let v_ov = v_gs - vto;
    let arg = v_ov / VT;
    var v_ov_eff: f32;
    var sigmoid: f32;
    if arg > 20.0 {
        v_ov_eff = v_ov;
        sigmoid = 1.0;
    } else if arg < -20.0 {
        v_ov_eff = VT * exp(arg);
        sigmoid = exp(arg);
    } else {
        v_ov_eff = VT * log(1.0 + exp(arg));
        sigmoid = 1.0 / (1.0 + exp(-arg));
    }

    if v_ds < v_ov_eff {
        // Linear region
        let lam = 1.0 + lambda * v_ds;
        i_d = kp * (v_ov_eff * v_ds - v_ds * v_ds / 2.0) * lam;
        g_m = kp * v_ds * lam * sigmoid;
        g_ds = kp * (v_ov_eff - v_ds) * lam + kp * (v_ov_eff * v_ds - v_ds * v_ds / 2.0) * lambda;
    } else {
        // Saturation region
        let lam = 1.0 + lambda * v_ds;
        i_d = kp / 2.0 * v_ov_eff * v_ov_eff * lam;
        g_m = kp * v_ov_eff * lam * sigmoid;
        g_ds = kp / 2.0 * v_ov_eff * v_ov_eff * lambda;
    }

    let out = 4u * i;
    eval_output[out + 0u] = polarity * i_d;
    eval_output[out + 1u] = g_m;
    eval_output[out + 2u] = g_ds;
    eval_output[out + 3u] = polarity;
}
"#;

/// WGSL shader for assembling BJT contributions into MNA matrix and RHS.
///
/// Two entry points:
/// - `assemble_bjt_matrix_stamp`: Expands 4 Jacobian derivatives to 9 G-matrix
///   entries via KCL and stamps into CSR positions.
/// - `assemble_bjt_rhs_stamp`: Stamps Norton companion currents into RHS vector.
pub const BJT_ASSEMBLE_SHADER: &str = r#"
// ============================================================
// BJT Assembly Shader
// ============================================================

const SENTINEL: u32 = 0xFFFFFFFFu;

// --- Matrix stamp: expand 4 Jacobian values to 9 G-matrix entries per BJT ---

struct BjtStampMatParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_mat_desc: array<u32>;
@group(0) @binding(1) var<storage, read> bjt_mat_eval: array<f32>;
@group(0) @binding(2) var<storage, read_write> bjt_mat_values: array<f32>;
@group(0) @binding(3) var<uniform> bjt_mat_params: BjtStampMatParams;

@compute @workgroup_size(64)
fn assemble_bjt_matrix_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bjt_mat_params.num_bjts {
        return;
    }

    let desc_base = i * 24u;
    let eval_base = 6u * i;

    // Read 4 Jacobian derivatives from eval output
    let di_c_dvbe = bjt_mat_eval[eval_base + 2u];
    let di_c_dvbc = bjt_mat_eval[eval_base + 3u];
    let di_b_dvbe = bjt_mat_eval[eval_base + 4u];
    let di_b_dvbc = bjt_mat_eval[eval_base + 5u];

    // Compute 9 stamp values (3x3 matrix)
    // Row: Collector
    let g_cc = -(di_c_dvbc);              // dI_C/dV_C
    let g_cb = di_c_dvbe + di_c_dvbc;     // dI_C/dV_B
    let g_ce = -(di_c_dvbe);              // dI_C/dV_E

    // Row: Base
    let g_bc = -(di_b_dvbc);              // dI_B/dV_C
    let g_bb = di_b_dvbe + di_b_dvbc;     // dI_B/dV_B
    let g_be = -(di_b_dvbe);              // dI_B/dV_E

    // Row: Emitter (KCL: I_E = -(I_C + I_B))
    let g_ec = di_c_dvbc + di_b_dvbc;     // dI_E/dV_C = -(dI_C/dV_C + dI_B/dV_C)
    let g_eb = -(di_c_dvbe + di_c_dvbc + di_b_dvbe + di_b_dvbc); // dI_E/dV_B
    let g_ee = di_c_dvbe + di_b_dvbe;     // dI_E/dV_E

    // Stamp values correspond to g_row_col[0..8] in descriptor
    var stamps: array<f32, 9>;
    stamps[0] = g_cc;
    stamps[1] = g_cb;
    stamps[2] = g_ce;
    stamps[3] = g_bc;
    stamps[4] = g_bb;
    stamps[5] = g_be;
    stamps[6] = g_ec;
    stamps[7] = g_eb;
    stamps[8] = g_ee;

    for (var j = 0u; j < 9u; j = j + 1u) {
        let pos = bjt_mat_desc[desc_base + 12u + j];
        if pos != SENTINEL {
            bjt_mat_values[pos] = bjt_mat_values[pos] + stamps[j];
        }
    }
}

// --- RHS stamp: Norton companion current per BJT ---

struct BjtStampRhsParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_rhs_desc: array<u32>;
@group(0) @binding(1) var<storage, read> bjt_rhs_x: array<f32>;
@group(0) @binding(2) var<storage, read> bjt_rhs_eval: array<f32>;
@group(0) @binding(3) var<storage, read_write> bjt_rhs_b: array<f32>;
@group(0) @binding(4) var<uniform> bjt_rhs_params: BjtStampRhsParams;

fn bjt_read_voltage(idx: u32) -> f32 {
    if idx == SENTINEL {
        return 0.0;
    }
    return bjt_rhs_x[idx];
}

@compute @workgroup_size(64)
fn assemble_bjt_rhs_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bjt_rhs_params.num_bjts {
        return;
    }

    let desc_base = i * 24u;
    let eval_base = 6u * i;

    let collector_idx = bjt_rhs_desc[desc_base + 0u];
    let base_idx = bjt_rhs_desc[desc_base + 1u];
    let emitter_idx = bjt_rhs_desc[desc_base + 2u];

    // Read eval outputs
    let i_c = bjt_rhs_eval[eval_base + 0u];
    let i_b = bjt_rhs_eval[eval_base + 1u];
    let i_e = -(i_c + i_b);

    let di_c_dvbe = bjt_rhs_eval[eval_base + 2u];
    let di_c_dvbc = bjt_rhs_eval[eval_base + 3u];
    let di_b_dvbe = bjt_rhs_eval[eval_base + 4u];
    let di_b_dvbc = bjt_rhs_eval[eval_base + 5u];

    // Read node voltages
    let v_c = bjt_read_voltage(collector_idx);
    let v_b = bjt_read_voltage(base_idx);
    let v_e = bjt_read_voltage(emitter_idx);

    // Collector row stamp values (same as matrix stamp)
    let g_cc = -(di_c_dvbc);
    let g_cb = di_c_dvbe + di_c_dvbc;
    let g_ce = -(di_c_dvbe);

    // Base row stamp values
    let g_bc = -(di_b_dvbc);
    let g_bb = di_b_dvbe + di_b_dvbc;
    let g_be = -(di_b_dvbe);

    // Emitter row stamp values
    let g_ec = di_c_dvbc + di_b_dvbc;
    let g_eb = -(di_c_dvbe + di_c_dvbc + di_b_dvbe + di_b_dvbc);
    let g_ee = di_c_dvbe + di_b_dvbe;

    // Linearized currents at operating point
    let i_lin_c = g_cc * v_c + g_cb * v_b + g_ce * v_e;
    let i_lin_b = g_bc * v_c + g_bb * v_b + g_be * v_e;
    let i_lin_e = g_ec * v_c + g_eb * v_b + g_ee * v_e;

    // Norton equivalents
    let i_eq_c = i_c - i_lin_c;
    let i_eq_b = i_b - i_lin_b;
    let i_eq_e = i_e - i_lin_e;

    // RHS indices: b_idx[0]=C, b_idx[1]=B, b_idx[2]=E
    let b_c = bjt_rhs_desc[desc_base + 21u];
    let b_b = bjt_rhs_desc[desc_base + 22u];
    let b_e = bjt_rhs_desc[desc_base + 23u];

    if b_c != SENTINEL {
        bjt_rhs_b[b_c] = bjt_rhs_b[b_c] - i_eq_c;
    }
    if b_b != SENTINEL {
        bjt_rhs_b[b_b] = bjt_rhs_b[b_b] - i_eq_b;
    }
    if b_e != SENTINEL {
        bjt_rhs_b[b_e] = bjt_rhs_b[b_e] - i_eq_e;
    }
}
"#;

/// WGSL shader for assembling MOSFET contributions into MNA matrix and RHS.
///
/// Two entry points:
/// - `assemble_mosfet_matrix_stamp`: Stamps g_m and g_ds into 6 G-matrix entries.
/// - `assemble_mosfet_rhs_stamp`: Stamps Norton companion currents into RHS vector.
pub const MOSFET_ASSEMBLE_SHADER: &str = r#"
// ============================================================
// MOSFET Assembly Shader
// ============================================================

const SENTINEL: u32 = 0xFFFFFFFFu;

// --- Matrix stamp: 6 G-matrix entries per MOSFET ---

struct MosfetStampMatParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_mat_desc: array<u32>;
@group(0) @binding(1) var<storage, read> mosfet_mat_eval: array<f32>;
@group(0) @binding(2) var<storage, read_write> mosfet_mat_values: array<f32>;
@group(0) @binding(3) var<uniform> mosfet_mat_params: MosfetStampMatParams;

@compute @workgroup_size(64)
fn assemble_mosfet_matrix_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mosfet_mat_params.num_mosfets {
        return;
    }

    let desc_base = i * 16u;
    let eval_base = 4u * i;

    let g_m = mosfet_mat_eval[eval_base + 1u];
    let g_ds = mosfet_mat_eval[eval_base + 2u];

    // Stamp order: [DD, DG, DS, SD, SG, SS]
    // Drain row: G[D,D] += g_ds, G[D,G] += g_m, G[D,S] += -(g_m + g_ds)
    // Source row: G[S,D] += -g_ds, G[S,G] += -g_m, G[S,S] += g_m + g_ds
    var stamps: array<f32, 6>;
    stamps[0] = g_ds;               // DD
    stamps[1] = g_m;                // DG
    stamps[2] = -(g_m + g_ds);      // DS
    stamps[3] = -g_ds;              // SD
    stamps[4] = -g_m;               // SG
    stamps[5] = g_m + g_ds;         // SS

    for (var j = 0u; j < 6u; j = j + 1u) {
        let pos = mosfet_mat_desc[desc_base + 8u + j];
        if pos != SENTINEL {
            mosfet_mat_values[pos] = mosfet_mat_values[pos] + stamps[j];
        }
    }
}

// --- RHS stamp: Norton companion current per MOSFET ---

struct MosfetStampRhsParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_rhs_desc: array<u32>;
@group(0) @binding(1) var<storage, read> mosfet_rhs_x: array<f32>;
@group(0) @binding(2) var<storage, read> mosfet_rhs_eval: array<f32>;
@group(0) @binding(3) var<storage, read_write> mosfet_rhs_b: array<f32>;
@group(0) @binding(4) var<uniform> mosfet_rhs_params: MosfetStampRhsParams;

fn mosfet_read_voltage(idx: u32) -> f32 {
    if idx == SENTINEL {
        return 0.0;
    }
    return mosfet_rhs_x[idx];
}

@compute @workgroup_size(64)
fn assemble_mosfet_rhs_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mosfet_rhs_params.num_mosfets {
        return;
    }

    let desc_base = i * 16u;
    let eval_base = 4u * i;

    let drain_idx = mosfet_rhs_desc[desc_base + 0u];
    let source_idx = mosfet_rhs_desc[desc_base + 2u];

    let i_d = mosfet_rhs_eval[eval_base + 0u];  // polarity-adjusted from eval
    let g_m = mosfet_rhs_eval[eval_base + 1u];
    let g_ds = mosfet_rhs_eval[eval_base + 2u];

    let gate_idx = mosfet_rhs_desc[desc_base + 1u];

    let v_d = mosfet_read_voltage(drain_idx);
    let v_g = mosfet_read_voltage(gate_idx);
    let v_s = mosfet_read_voltage(source_idx);

    // Drain: I_lin = g_ds * V_D + g_m * V_G - (g_m + g_ds) * V_S
    let i_lin_d = g_ds * v_d + g_m * v_g - (g_m + g_ds) * v_s;
    let i_eq_d = i_d - i_lin_d;

    // Source: I_S = -I_D, I_lin = -g_ds * V_D - g_m * V_G + (g_m + g_ds) * V_S
    let i_lin_s = -g_ds * v_d - g_m * v_g + (g_m + g_ds) * v_s;
    let i_eq_s = -i_d - i_lin_s;

    // RHS indices: b_idx[0]=D, b_idx[1]=S
    let b_d = mosfet_rhs_desc[desc_base + 14u];
    let b_s = mosfet_rhs_desc[desc_base + 15u];

    if b_d != SENTINEL {
        mosfet_rhs_b[b_d] = mosfet_rhs_b[b_d] - i_eq_d;
    }
    if b_s != SENTINEL {
        mosfet_rhs_b[b_s] = mosfet_rhs_b[b_s] - i_eq_s;
    }
}
"#;

/// WGSL shader for BJT voltage limiting during Newton-Raphson iteration.
///
/// One invocation per BJT. Uses SPICE3f5 `pnjlim` function with logarithmic
/// damping for forward-biased junctions. Reverse-biased junctions are not limited.
pub const BJT_VOLTAGE_LIMIT_SHADER: &str = r#"
// ============================================================
// BJT Voltage Limiting Shader (SPICE pnjlim)
// ============================================================

struct BjtVoltLimitParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bvl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> bvl_x_old: array<f32>;
@group(0) @binding(2) var<storage, read_write> bvl_x_new: array<f32>;
@group(0) @binding(3) var<uniform> bvl_params: BjtVoltLimitParams;

const BVL_SENTINEL: u32 = 0xFFFFFFFFu;

fn bvl_read_old(idx: u32) -> f32 {
    if idx == BVL_SENTINEL {
        return 0.0;
    }
    return bvl_x_old[idx];
}

fn bvl_read_new(idx: u32) -> f32 {
    if idx == BVL_SENTINEL {
        return 0.0;
    }
    return bvl_x_new[idx];
}

// SPICE3f5 pnjlim: logarithmic damping for forward-biased PN junctions.
// vnew/vold are polarity-adjusted so forward bias is positive.
fn bvl_pnjlim(vnew: f32, vold: f32, vt: f32, vcrit: f32) -> f32 {
    if vnew > vcrit && abs(vnew - vold) > vt + vt {
        if vold > 0.0 {
            let arg = 1.0 + (vnew - vold) / vt;
            if arg > 0.0 {
                return vold + vt * (2.0 + log(arg - 2.0));
            } else {
                return vcrit;
            }
        } else {
            return vt * log(vnew / vt);
        }
    }
    return vnew;
}

@compute @workgroup_size(64)
fn bjt_voltage_limit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bvl_params.num_bjts {
        return;
    }

    let base = i * 24u;
    let collector_idx = bvl_desc[base + 0u];
    let base_idx = bvl_desc[base + 1u];
    let emitter_idx = bvl_desc[base + 2u];
    let polarity = bitcast<f32>(bvl_desc[base + 3u]);
    let is_val = bitcast<f32>(bvl_desc[base + 4u]);
    let nf_vt = bitcast<f32>(bvl_desc[base + 7u]);
    let nr_vt = bitcast<f32>(bvl_desc[base + 8u]);

    let collector_fixed = bvl_desc[base + 9u];
    let base_fixed = bvl_desc[base + 10u];
    let emitter_fixed = bvl_desc[base + 11u];

    let sqrt2 = 1.4142135;
    let vcrit_f = nf_vt * log(nf_vt / (sqrt2 * is_val));
    let vcrit_r = nr_vt * log(nr_vt / (sqrt2 * is_val));

    // V_BE limiting (SPICE pnjlim) — polarity-adjusted so forward bias is positive
    let vbe_old = polarity * (bvl_read_old(base_idx) - bvl_read_old(emitter_idx));
    let vbe_new = polarity * (bvl_read_new(base_idx) - bvl_read_new(emitter_idx));
    let vbe_limited = bvl_pnjlim(vbe_new, vbe_old, nf_vt, vcrit_f);

    if vbe_limited != vbe_new {
        let delta = vbe_new - vbe_old;
        if abs(delta) > 1e-30 {
            let scale_be = (vbe_limited - vbe_old) / delta;
            if base_idx != BVL_SENTINEL && base_fixed == 0u {
                let old_val = bvl_x_old[base_idx];
                let d = bvl_x_new[base_idx] - old_val;
                bvl_x_new[base_idx] = old_val + d * scale_be;
            }
            if emitter_idx != BVL_SENTINEL && emitter_fixed == 0u {
                let old_val = bvl_x_old[emitter_idx];
                let d = bvl_x_new[emitter_idx] - old_val;
                bvl_x_new[emitter_idx] = old_val + d * scale_be;
            }
        }
    }

    // V_BC limiting (SPICE pnjlim)
    let vbc_old = polarity * (bvl_read_old(base_idx) - bvl_read_old(collector_idx));
    let vbc_new = polarity * (bvl_read_new(base_idx) - bvl_read_new(collector_idx));
    let vbc_limited = bvl_pnjlim(vbc_new, vbc_old, nr_vt, vcrit_r);

    if vbc_limited != vbc_new {
        let delta = vbc_new - vbc_old;
        if abs(delta) > 1e-30 {
            let scale_bc = (vbc_limited - vbc_old) / delta;
            if base_idx != BVL_SENTINEL && base_fixed == 0u {
                let old_val = bvl_x_old[base_idx];
                let d = bvl_x_new[base_idx] - old_val;
                bvl_x_new[base_idx] = old_val + d * scale_bc;
            }
            if collector_idx != BVL_SENTINEL && collector_fixed == 0u {
                let old_val = bvl_x_old[collector_idx];
                let d = bvl_x_new[collector_idx] - old_val;
                bvl_x_new[collector_idx] = old_val + d * scale_bc;
            }
        }
    }
}
"#;

/// WGSL shader for MOSFET voltage limiting during Newton-Raphson iteration.
///
/// Uses a race-free two-pass scheme:
/// 1) per-MOSFET reduction computes per-node minimum scale via atomicMin
/// 2) per-node apply scales x_new using the reduced factor
pub const MOSFET_VOLTAGE_LIMIT_SHADER: &str = r#"
// ============================================================
// MOSFET Voltage Limiting Shader
// ============================================================

struct MosfetVoltLimitParams {
    num_mosfets: u32,
    n_nodes: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> mvl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> mvl_x_old: array<f32>;
@group(0) @binding(2) var<storage, read_write> mvl_x_new: array<f32>;
@group(0) @binding(3) var<storage, read_write> mvl_node_scale: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> mvl_params: MosfetVoltLimitParams;

const MVL_SENTINEL: u32 = 0xFFFFFFFFu;
const MVL_SCALE_ONE: u32 = 1000000u;
const MVL_STEP_LIMIT: f32 = 0.5;

fn mvl_read_old(idx: u32) -> f32 {
    if idx == MVL_SENTINEL {
        return 0.0;
    }
    return mvl_x_old[idx];
}

fn mvl_read_new(idx: u32) -> f32 {
    if idx == MVL_SENTINEL {
        return 0.0;
    }
    return mvl_x_new[idx];
}

fn mvl_scale_to_encoded(delta_v: f32, limit: f32) -> u32 {
    let scale = limit / abs(delta_v);
    let encoded = u32(scale * f32(MVL_SCALE_ONE));
    return max(encoded, 1u);
}

@compute @workgroup_size(64)
fn mosfet_voltage_limit_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mvl_params.num_mosfets {
        return;
    }

    let base = i * 16u;
    let drain_idx = mvl_desc[base + 0u];
    let gate_idx = mvl_desc[base + 1u];
    let source_idx = mvl_desc[base + 2u];

    // Per-node voltage-source-constrained flags (word 7, packed bits).
    // bit 0 = drain, bit 1 = gate, bit 2 = source.
    // Skip modifying nodes driven by voltage sources.
    let node_fixed_packed = mvl_desc[base + 7u];
    let drain_fixed = (node_fixed_packed & 1u) != 0u;
    let gate_fixed = (node_fixed_packed & 2u) != 0u;
    let source_fixed = (node_fixed_packed & 4u) != 0u;

    // V_GS limiting (clamp to +/-0.5V)
    let vgs_old = mvl_read_old(gate_idx) - mvl_read_old(source_idx);
    let vgs_new = mvl_read_new(gate_idx) - mvl_read_new(source_idx);
    let delta_vgs = vgs_new - vgs_old;

    if abs(delta_vgs) > MVL_STEP_LIMIT {
        let scale_gs = mvl_scale_to_encoded(delta_vgs, MVL_STEP_LIMIT);
        if gate_idx != MVL_SENTINEL && !gate_fixed {
            atomicMin(&mvl_node_scale[gate_idx], scale_gs);
        }
        if source_idx != MVL_SENTINEL && !source_fixed {
            atomicMin(&mvl_node_scale[source_idx], scale_gs);
        }
    }

    // V_DS limiting (clamp to +/-0.5V)
    let vds_old = mvl_read_old(drain_idx) - mvl_read_old(source_idx);
    let vds_new = mvl_read_new(drain_idx) - mvl_read_new(source_idx);
    let delta_vds = vds_new - vds_old;

    if abs(delta_vds) > MVL_STEP_LIMIT {
        let scale_ds = mvl_scale_to_encoded(delta_vds, MVL_STEP_LIMIT);
        if drain_idx != MVL_SENTINEL && !drain_fixed {
            atomicMin(&mvl_node_scale[drain_idx], scale_ds);
        }
        if source_idx != MVL_SENTINEL && !source_fixed {
            atomicMin(&mvl_node_scale[source_idx], scale_ds);
        }
    }
}

@compute @workgroup_size(64)
fn mosfet_voltage_limit_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = gid.x;
    if node_idx >= mvl_params.n_nodes {
        return;
    }

    // Keep descriptor binding active for a stable bind-group layout across
    // both limiter entry points.
    let keep_layout_word = mvl_desc[0u];
    if keep_layout_word == 0xFFFFFFFEu && node_idx == 0u {
        return;
    }

    let scale_encoded = atomicLoad(&mvl_node_scale[node_idx]);
    if scale_encoded >= MVL_SCALE_ONE {
        return;
    }

    let scale = f32(scale_encoded) / f32(MVL_SCALE_ONE);
    let old_val = mvl_x_old[node_idx];
    let delta = mvl_x_new[node_idx] - old_val;
    mvl_x_new[node_idx] = old_val + delta * scale;
}
"#;

/// WGSL shader for Newton-Raphson convergence checking and voltage limiting.
///
/// Two entry points:
/// - `voltage_limit`: One invocation per diode. Uses SPICE3f5 `pnjlim`
///   logarithmic damping for forward-biased junctions. No limiting for reverse bias.
///
/// - `convergence_check`: Parallel reduction computing max|x_new - x_old|.
///   Writes 1 to converged flag if max_diff < tolerance, 0 otherwise.
///   Also detects NaN/Inf in the solution.
pub const CONVERGENCE_SHADER: &str = r#"
// ============================================================
// Convergence & Voltage Limiting Shader
// ============================================================

// --- Voltage limiting: SPICE pnjlim for diode nodes ---

struct VoltLimitParams {
    num_diodes: u32,
}

@group(0) @binding(0) var<storage, read> vl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> vl_x_old: array<f32>;
@group(0) @binding(2) var<storage, read_write> vl_x_new: array<f32>;
@group(0) @binding(3) var<uniform> vl_params: VoltLimitParams;

const VL_SENTINEL: u32 = 0xFFFFFFFFu;

fn vl_read_old(idx: u32) -> f32 {
    if idx == VL_SENTINEL {
        return 0.0;
    }
    return vl_x_old[idx];
}

fn vl_read_new(idx: u32) -> f32 {
    if idx == VL_SENTINEL {
        return 0.0;
    }
    return vl_x_new[idx];
}

// SPICE3f5 pnjlim: logarithmic damping for forward-biased PN junctions.
fn vl_pnjlim(vnew: f32, vold: f32, vt: f32, vcrit: f32) -> f32 {
    if vnew > vcrit && abs(vnew - vold) > vt + vt {
        if vold > 0.0 {
            let arg = 1.0 + (vnew - vold) / vt;
            if arg > 0.0 {
                return vold + vt * (2.0 + log(arg - 2.0));
            } else {
                return vcrit;
            }
        } else {
            return vt * log(vnew / vt);
        }
    }
    return vnew;
}

@compute @workgroup_size(64)
fn voltage_limit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= vl_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = vl_desc[base + 0u];
    let cathode_idx = vl_desc[base + 1u];
    let is_val = bitcast<f32>(vl_desc[base + 2u]);
    let n_vt = bitcast<f32>(vl_desc[base + 3u]);

    let sqrt2 = 1.4142135;
    let vcrit = n_vt * log(n_vt / (sqrt2 * is_val));

    let v_old = vl_read_old(anode_idx) - vl_read_old(cathode_idx);
    let v_new = vl_read_new(anode_idx) - vl_read_new(cathode_idx);
    let v_limited = vl_pnjlim(v_new, v_old, n_vt, vcrit);

    if v_limited != v_new {
        let delta = v_new - v_old;
        if abs(delta) > 1e-30 {
            let scale = (v_limited - v_old) / delta;
            if anode_idx != VL_SENTINEL {
                let anode_old = vl_x_old[anode_idx];
                let anode_delta = vl_x_new[anode_idx] - anode_old;
                vl_x_new[anode_idx] = anode_old + anode_delta * scale;
            }
            if cathode_idx != VL_SENTINEL {
                let cathode_old = vl_x_old[cathode_idx];
                let cathode_delta = vl_x_new[cathode_idx] - cathode_old;
                vl_x_new[cathode_idx] = cathode_old + cathode_delta * scale;
            }
        }
    }
}

// --- Convergence check: parallel max-reduction of |x_new - x_old| ---

const CONV_WG_SIZE: u32 = 64u;

struct ConvParams {
    n: u32,
    tolerance: f32,
}

var<workgroup> conv_scratch: array<f32, 64>;

@group(0) @binding(0) var<storage, read> conv_x_new: array<f32>;
@group(0) @binding(1) var<storage, read> conv_x_old: array<f32>;
// Output: [0] = max_diff per workgroup (array of partial maxes)
@group(0) @binding(2) var<storage, read_write> conv_partial_max: array<f32>;
// Flag buffer: [0] = 1 if converged, 0 otherwise; [1] = 1 if NaN/Inf detected
@group(0) @binding(3) var<storage, read_write> conv_flags: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> conv_params: ConvParams;

@compute @workgroup_size(64)
fn convergence_check(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    var diff: f32 = 0.0;
    if i < conv_params.n {
        let xn = conv_x_new[i];
        let xo = conv_x_old[i];
        diff = abs(xn - xo);

        // Detect NaN or Inf: if xn is not finite, flag it
        if xn != xn || xn - xn != 0.0 {
            atomicOr(&conv_flags[1u], 1u);
            diff = 1.0e30;  // Force non-convergence
        }
    }

    conv_scratch[local_id] = diff;
    workgroupBarrier();

    // Tree reduction for max
    var stride = CONV_WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            conv_scratch[local_id] = max(conv_scratch[local_id], conv_scratch[local_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes workgroup max
    if local_id == 0u {
        conv_partial_max[wid.x] = conv_scratch[0];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference implementation of the smooth MOSFET evaluation shader.
    /// Returns (i_d, g_m, g_ds) matching the GPU shader logic exactly.
    fn mosfet_eval_reference(
        v_gs: f32,
        v_ds: f32,
        vto: f32,
        kp: f32,
        lambda: f32,
        polarity: f32,
    ) -> (f32, f32, f32) {
        let v_gs = polarity * v_gs;
        let v_ds = polarity * v_ds;

        let vt: f32 = 0.02585;
        let v_ov = v_gs - vto;
        let arg = v_ov / vt;

        let (v_ov_eff, sigmoid) = if arg > 20.0 {
            (v_ov, 1.0_f32)
        } else if arg < -20.0 {
            (vt * arg.exp(), arg.exp())
        } else {
            (vt * (1.0 + arg.exp()).ln(), 1.0 / (1.0 + (-arg).exp()))
        };

        let (i_d, g_m, g_ds);

        if v_ds < v_ov_eff {
            // Linear region
            let lam = 1.0 + lambda * v_ds;
            i_d = kp * (v_ov_eff * v_ds - v_ds * v_ds / 2.0) * lam;
            g_m = kp * v_ds * lam * sigmoid;
            g_ds = kp * (v_ov_eff - v_ds) * lam
                + kp * (v_ov_eff * v_ds - v_ds * v_ds / 2.0) * lambda;
        } else {
            // Saturation region
            let lam = 1.0 + lambda * v_ds;
            i_d = kp / 2.0 * v_ov_eff * v_ov_eff * lam;
            g_m = kp * v_ov_eff * lam * sigmoid;
            g_ds = kp / 2.0 * v_ov_eff * v_ov_eff * lambda;
        }

        (polarity * i_d, g_m, g_ds)
    }

    #[test]
    fn test_smooth_mosfet_deep_saturation() {
        // v_ov >> VT: smooth model should match original hard-cutoff model
        let vto = 0.7;
        let kp = 1.1e-4;
        let lambda = 0.04;
        let v_gs = 3.0; // v_ov = 2.3, well above VT
        let v_ds = 5.0; // saturation: v_ds > v_ov

        let (i_d, g_m, g_ds) = mosfet_eval_reference(v_gs, v_ds, vto, kp, lambda, 1.0);

        // Hard-cutoff reference: v_ov_eff ≈ v_ov when v_ov >> VT
        let v_ov = v_gs - vto;
        let lam = 1.0 + lambda * v_ds;
        let i_d_ref = kp / 2.0 * v_ov * v_ov * lam;
        let g_m_ref = kp * v_ov * lam;
        let g_ds_ref = kp / 2.0 * v_ov * v_ov * lambda;

        assert!(
            (i_d - i_d_ref).abs() / i_d_ref < 1e-3,
            "i_d={i_d}, ref={i_d_ref}"
        );
        assert!(
            (g_m - g_m_ref).abs() / g_m_ref < 1e-3,
            "g_m={g_m}, ref={g_m_ref}"
        );
        assert!(
            (g_ds - g_ds_ref).abs() / g_ds_ref < 1e-3,
            "g_ds={g_ds}, ref={g_ds_ref}"
        );
    }

    #[test]
    fn test_smooth_mosfet_deep_cutoff() {
        // v_ov << -VT: i_d ≈ 0, g_m ≈ 0, g_ds ≈ 0
        let (i_d, g_m, g_ds) = mosfet_eval_reference(0.0, 3.0, 0.7, 1.1e-4, 0.04, 1.0);

        assert!(i_d.abs() < 1e-10, "i_d={i_d}, expected ≈0 in cutoff");
        assert!(g_m.abs() < 1e-8, "g_m={g_m}, expected ≈0 in cutoff");
        assert!(g_ds.abs() < 1e-10, "g_ds={g_ds}, expected ≈0 in cutoff");
    }

    #[test]
    fn test_smooth_mosfet_at_threshold() {
        // v_gs = vto, v_ov = 0 → v_ov_eff = VT*ln(2) ≈ 0.0179
        let vt: f32 = 0.02585;
        let (i_d, g_m, _g_ds) = mosfet_eval_reference(0.7, 3.0, 0.7, 1.1e-4, 0.04, 1.0);

        // Should be non-zero but small
        let v_ov_eff = vt * 2.0_f32.ln();
        let expected_id = 1.1e-4 / 2.0 * v_ov_eff * v_ov_eff * (1.0 + 0.04 * 3.0);
        assert!(i_d > 0.0, "i_d should be positive at threshold");
        assert!(
            (i_d - expected_id).abs() / expected_id < 0.01,
            "i_d={i_d}, expected={expected_id}"
        );
        assert!(g_m > 0.0, "g_m should be positive at threshold");
    }

    #[test]
    fn test_smooth_mosfet_continuity_across_threshold() {
        // Sample v_gs from vto-0.1 to vto+0.1, verify i_d is monotonically increasing
        let vto = 0.7_f32;
        let mut prev_id = f32::NEG_INFINITY;
        for i in 0..=20 {
            let v_gs = (vto - 0.1) + (i as f32) * 0.01;
            let (i_d, _, _) = mosfet_eval_reference(v_gs, 3.0, vto, 1.1e-4, 0.04, 1.0);
            assert!(
                i_d >= prev_id,
                "i_d not monotonic at v_gs={v_gs}: {i_d} < {prev_id}"
            );
            prev_id = i_d;
        }
    }

    #[test]
    fn test_smooth_mosfet_linear_region() {
        // v_ds < v_ov_eff → linear region
        let vto = 0.7;
        let kp = 1.1e-4;
        let lambda = 0.04;
        let v_gs = 3.0; // v_ov = 2.3
        let v_ds = 0.5; // well below v_ov_eff ≈ 2.3

        let (i_d, g_m, g_ds) = mosfet_eval_reference(v_gs, v_ds, vto, kp, lambda, 1.0);

        // Hard-cutoff linear region reference
        let v_ov = v_gs - vto;
        let lam = 1.0 + lambda * v_ds;
        let i_d_ref = kp * (v_ov * v_ds - v_ds * v_ds / 2.0) * lam;
        let g_m_ref = kp * v_ds * lam; // sigmoid ≈ 1 for deep ON
        let g_ds_ref =
            kp * (v_ov - v_ds) * lam + kp * (v_ov * v_ds - v_ds * v_ds / 2.0) * lambda;

        assert!(
            (i_d - i_d_ref).abs() / i_d_ref < 1e-3,
            "linear i_d={i_d}, ref={i_d_ref}"
        );
        assert!(
            (g_m - g_m_ref).abs() / g_m_ref < 1e-3,
            "linear g_m={g_m}, ref={g_m_ref}"
        );
        assert!(
            (g_ds - g_ds_ref).abs() / g_ds_ref < 1e-3,
            "linear g_ds={g_ds}, ref={g_ds_ref}"
        );
    }

    #[test]
    fn test_smooth_mosfet_sigmoid_derivative_consistency() {
        // Verify g_m ≈ d(i_d)/d(v_gs) via finite differences
        let vto = 0.7;
        let kp = 1.1e-4;
        let lambda = 0.04;
        let v_ds = 3.0;
        let dv = 1e-4_f32;

        for v_gs in [0.5, 0.7, 0.9, 1.5, 2.5] {
            let (i_d_plus, _, _) =
                mosfet_eval_reference(v_gs + dv, v_ds, vto, kp, lambda, 1.0);
            let (i_d_minus, _, _) =
                mosfet_eval_reference(v_gs - dv, v_ds, vto, kp, lambda, 1.0);
            let (_, g_m, _) = mosfet_eval_reference(v_gs, v_ds, vto, kp, lambda, 1.0);

            let g_m_fd = (i_d_plus - i_d_minus) / (2.0 * dv);
            let rel_err = if g_m_fd.abs() > 1e-10 {
                (g_m - g_m_fd).abs() / g_m_fd.abs()
            } else {
                (g_m - g_m_fd).abs()
            };
            assert!(
                rel_err < 0.01,
                "g_m mismatch at v_gs={v_gs}: g_m={g_m}, fd={g_m_fd}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_smooth_mosfet_pmos_polarity() {
        // PMOS: polarity=-1, V_SG > |VTO| should conduct
        let vto = 0.7; // PMOS VTO is positive in Shichman-Hodges convention
        let kp = 5.5e-5;
        let lambda = 0.04;

        // Source at VDD=5, gate at 3 → physical V_SG=2V
        // With polarity=-1: v_gs = -1*(3-5) = 2, v_ds = -1*(drain-5)
        // Let drain be at 1V → v_ds = -1*(1-5) = 4
        let v_gs_physical = 3.0 - 5.0; // -2.0
        let v_ds_physical = 1.0 - 5.0; // -4.0

        let (i_d, g_m, g_ds) =
            mosfet_eval_reference(v_gs_physical, v_ds_physical, vto, kp, lambda, -1.0);

        // Current should flow source→drain (negative polarity means i_d < 0 in terms
        // of the physical drain current direction, but polarity*i_d gives the stamped
        // value). The returned i_d already has polarity applied.
        // For PMOS conducting: physical current flows from source to drain,
        // meaning conventional current into the drain node from outside is negative.
        assert!(i_d < 0.0, "PMOS i_d={i_d}, expected < 0 (current sink)");
        assert!(g_m > 0.0, "PMOS g_m={g_m}, expected > 0");
        assert!(g_ds > 0.0, "PMOS g_ds={g_ds}, expected > 0");
    }

    /// Validate that the WGSL shader source parses without errors.
    /// Uses naga's WGSL frontend directly so this works without a GPU.
    #[test]
    fn wgsl_parses_successfully() {
        let result = naga::front::wgsl::parse_str(SHADER_SOURCE);
        match result {
            Ok(module) => {
                // Verify expected entry points exist
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = [
                    "spmv_real",
                    "spmv_complex",
                    "axpy_real",
                    "axpy_complex",
                    "dot_real",
                    "dot_complex",
                    "scale_real",
                    "scale_complex",
                    "copy_real",
                    "copy_complex",
                    "subtract_real",
                    "subtract_complex",
                    "jacobi_real",
                    "jacobi_complex",
                ];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!("WGSL parse error:\n{}", e.emit_to_string(SHADER_SOURCE));
            }
        }
    }

    #[test]
    fn diode_eval_shader_parses() {
        let result = naga::front::wgsl::parse_str(DIODE_EVAL_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                assert!(
                    entry_names.contains(&"diode_eval"),
                    "missing entry point: diode_eval. Found: {entry_names:?}"
                );
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in DIODE_EVAL_SHADER:\n{}",
                    e.emit_to_string(DIODE_EVAL_SHADER)
                );
            }
        }
    }

    #[test]
    fn nonlinear_assemble_shader_parses() {
        let result = naga::front::wgsl::parse_str(NONLINEAR_ASSEMBLE_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = [
                    "assemble_matrix_copy",
                    "assemble_matrix_stamp",
                    "assemble_rhs_copy",
                    "assemble_rhs_stamp",
                ];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in NONLINEAR_ASSEMBLE_SHADER:\n{}",
                    e.emit_to_string(NONLINEAR_ASSEMBLE_SHADER)
                );
            }
        }
    }

    #[test]
    fn convergence_shader_parses() {
        let result = naga::front::wgsl::parse_str(CONVERGENCE_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = ["voltage_limit", "convergence_check"];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in CONVERGENCE_SHADER:\n{}",
                    e.emit_to_string(CONVERGENCE_SHADER)
                );
            }
        }
    }

    #[test]
    fn bjt_eval_shader_parses() {
        let result = naga::front::wgsl::parse_str(BJT_EVAL_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                assert!(
                    entry_names.contains(&"bjt_eval"),
                    "missing entry point: bjt_eval. Found: {entry_names:?}"
                );
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in BJT_EVAL_SHADER:\n{}",
                    e.emit_to_string(BJT_EVAL_SHADER)
                );
            }
        }
    }

    #[test]
    fn mosfet_eval_shader_parses() {
        let result = naga::front::wgsl::parse_str(MOSFET_EVAL_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                assert!(
                    entry_names.contains(&"mosfet_eval"),
                    "missing entry point: mosfet_eval. Found: {entry_names:?}"
                );
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in MOSFET_EVAL_SHADER:\n{}",
                    e.emit_to_string(MOSFET_EVAL_SHADER)
                );
            }
        }
    }

    #[test]
    fn bjt_assemble_shader_parses() {
        let result = naga::front::wgsl::parse_str(BJT_ASSEMBLE_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = ["assemble_bjt_matrix_stamp", "assemble_bjt_rhs_stamp"];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in BJT_ASSEMBLE_SHADER:\n{}",
                    e.emit_to_string(BJT_ASSEMBLE_SHADER)
                );
            }
        }
    }

    #[test]
    fn mosfet_assemble_shader_parses() {
        let result = naga::front::wgsl::parse_str(MOSFET_ASSEMBLE_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = ["assemble_mosfet_matrix_stamp", "assemble_mosfet_rhs_stamp"];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in MOSFET_ASSEMBLE_SHADER:\n{}",
                    e.emit_to_string(MOSFET_ASSEMBLE_SHADER)
                );
            }
        }
    }

    #[test]
    fn bjt_voltage_limit_shader_parses() {
        let result = naga::front::wgsl::parse_str(BJT_VOLTAGE_LIMIT_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                assert!(
                    entry_names.contains(&"bjt_voltage_limit"),
                    "missing entry point: bjt_voltage_limit. Found: {entry_names:?}"
                );
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in BJT_VOLTAGE_LIMIT_SHADER:\n{}",
                    e.emit_to_string(BJT_VOLTAGE_LIMIT_SHADER)
                );
            }
        }
    }

    #[test]
    fn mosfet_voltage_limit_shader_parses() {
        let result = naga::front::wgsl::parse_str(MOSFET_VOLTAGE_LIMIT_SHADER);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = ["mosfet_voltage_limit_reduce", "mosfet_voltage_limit_apply"];
                for name in &expected {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!(
                    "WGSL parse error in MOSFET_VOLTAGE_LIMIT_SHADER:\n{}",
                    e.emit_to_string(MOSFET_VOLTAGE_LIMIT_SHADER)
                );
            }
        }
    }
}
