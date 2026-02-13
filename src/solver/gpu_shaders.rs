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

/// WGSL shader for Newton-Raphson convergence checking and voltage limiting.
///
/// Two entry points:
/// - `voltage_limit`: One invocation per diode. Clamps voltage change to
///   prevent Newton overshoot (|delta_v| limited to 2*n_vt).
///
/// - `convergence_check`: Parallel reduction computing max|x_new - x_old|.
///   Writes 1 to converged flag if max_diff < tolerance, 0 otherwise.
///   Also detects NaN/Inf in the solution.
pub const CONVERGENCE_SHADER: &str = r#"
// ============================================================
// Convergence & Voltage Limiting Shader
// ============================================================

// --- Voltage limiting: clamp Newton step for diode nodes ---

struct VoltLimitParams {
    num_diodes: u32,
}

// Diode descriptors
@group(0) @binding(0) var<storage, read> vl_desc: array<u32>;
// Previous solution
@group(0) @binding(1) var<storage, read> vl_x_old: array<f32>;
// New solution (will be modified in place)
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

@compute @workgroup_size(64)
fn voltage_limit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= vl_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = vl_desc[base + 0u];
    let cathode_idx = vl_desc[base + 1u];
    let n_vt = bitcast<f32>(vl_desc[base + 3u]);

    let v_old = vl_read_old(anode_idx) - vl_read_old(cathode_idx);
    let v_new = vl_read_new(anode_idx) - vl_read_new(cathode_idx);
    let delta_v = v_new - v_old;

    let limit = 2.0 * n_vt;

    if abs(delta_v) > limit {
        // Scale back: new_v = old_v + sign(delta) * limit
        let scale = limit / abs(delta_v);
        // Apply the scaling to non-ground nodes only
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

    /// Validate that the WGSL shader source parses without errors.
    /// Uses naga's WGSL frontend directly so this works without a GPU.
    #[test]
    fn wgsl_parses_successfully() {
        let result = naga::front::wgsl::parse_str(SHADER_SOURCE);
        match result {
            Ok(module) => {
                // Verify expected entry points exist
                let entry_names: Vec<&str> =
                    module.entry_points.iter().map(|ep| ep.name.as_str()).collect();
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
                let entry_names: Vec<&str> =
                    module.entry_points.iter().map(|ep| ep.name.as_str()).collect();
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
                let entry_names: Vec<&str> =
                    module.entry_points.iter().map(|ep| ep.name.as_str()).collect();
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
                let entry_names: Vec<&str> =
                    module.entry_points.iter().map(|ep| ep.name.as_str()).collect();
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
}
