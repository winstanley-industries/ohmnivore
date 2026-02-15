//! WGSL compute shaders using double-single (DS) arithmetic for ~f64 precision.
//!
//! Each value is represented as a (hi, lo) pair of f32 values where
//! true_value = hi + lo. This gives ~48 bits of mantissa (vs f32's 24),
//! enabling BiCGSTAB convergence on ill-conditioned matrices (kappa ~ 1e12).
//!
//! DS primitives (TwoSum, TwoProd) rely on IEEE 754 rounding guarantees
//! for f32 add/sub/mul/fma, which WGSL provides.

/// DS primitive functions: two_sum, two_prod, ds_add, ds_mul, ds_sub, ds_exp,
/// ds_div, ds_abs, ds_max, ds_from_f32, ds_neg.
///
/// Shared by both linear algebra shaders (spmv, dot, axpy, etc.) and
/// nonlinear shaders (device eval, assembly, voltage limiting, convergence).
pub const DS_PRIMITIVES: &str = r#"
// ============================================================
// Double-Single (DS) Arithmetic Primitives
// ============================================================
//
// DS representation: value ≈ hi + lo (two f32s give ~48-bit mantissa)
// Primitives: TwoSum (Knuth), TwoProd (via fma)

// Error-free addition: s + err = a + b exactly
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let err = (a - (s - v)) + (b - v);
    return vec2(s, err);
}

// Error-free multiplication via fma: p + err = a * b exactly
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let err = fma(a, b, -p);
    return vec2(p, err);
}

// DS add: (a_hi + a_lo) + (b_hi + b_lo)
fn ds_add(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let s = two_sum(a_hi, b_hi);
    let e = a_lo + b_lo + s.y;
    return two_sum(s.x, e);
}

// DS multiply: (a_hi + a_lo) * (b_hi + b_lo)
fn ds_mul(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let p = two_prod(a_hi, b_hi);
    let e = fma(a_hi, b_lo, fma(a_lo, b_hi, p.y));
    return two_sum(p.x, e);
}

// DS subtract: (a_hi + a_lo) - (b_hi + b_lo)
fn ds_sub(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    return ds_add(a_hi, a_lo, -b_hi, -b_lo);
}

// DS negate
fn ds_neg(a_hi: f32, a_lo: f32) -> vec2<f32> {
    return vec2(-a_hi, -a_lo);
}

// DS from single f32
fn ds_from_f32(v: f32) -> vec2<f32> {
    return vec2(v, 0.0);
}

// DS absolute value
fn ds_abs(a_hi: f32, a_lo: f32) -> vec2<f32> {
    if a_hi < 0.0 || (a_hi == 0.0 && a_lo < 0.0) {
        return vec2(-a_hi, -a_lo);
    }
    return vec2(a_hi, a_lo);
}

// DS max: returns the larger of two DS values
fn ds_max(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    if a_hi > b_hi || (a_hi == b_hi && a_lo > b_lo) {
        return vec2(a_hi, a_lo);
    }
    return vec2(b_hi, b_lo);
}

// DS compare: returns true if a > b
fn ds_gt(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> bool {
    return a_hi > b_hi || (a_hi == b_hi && a_lo > b_lo);
}

// DS compare: returns true if a < b
fn ds_lt(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> bool {
    return a_hi < b_hi || (a_hi == b_hi && a_lo < b_lo);
}

// DS exp: exp(x_hi + x_lo) ≈ exp(x_hi) * (1 + x_lo)
// First-order correction is sufficient for DS precision since x_lo << 1.
// Clamps argument to [-80, 80] to avoid inf/nan.
fn ds_exp(x_hi: f32, x_lo: f32) -> vec2<f32> {
    let clamped_hi = clamp(x_hi, -80.0, 80.0);
    // If we clamped, zero out lo to avoid nonsense correction
    var eff_lo = x_lo;
    if clamped_hi != x_hi {
        eff_lo = 0.0;
    }
    let e_hi = exp(clamped_hi);
    // exp(hi + lo) = exp(hi) * exp(lo) ≈ exp(hi) * (1 + lo) for small lo
    // Result: hi_part = e_hi, correction = e_hi * eff_lo
    let p = two_prod(e_hi, eff_lo);
    return two_sum(e_hi + p.x, p.y);
}

// DS division: (a_hi + a_lo) / (b_hi + b_lo)
// Uses Newton refinement: q0 = a_hi/b_hi, then correct residual.
fn ds_div(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let q0 = a_hi / b_hi;
    // residual = a - q0 * b
    let prod = ds_mul(q0, 0.0, b_hi, b_lo);
    let r = ds_sub(a_hi, a_lo, prod.x, prod.y);
    // correction
    let q1 = r.x / b_hi;
    return two_sum(q0, q1);
}
"#;

/// DS linear algebra shader body (without primitives).
///
/// Entry points: spmv_ds, dot_ds, axpy_ds, scale_ds, copy_ds, jacobi_ds.
const DS_LINEAR_BODY: &str = r#"
// --- SpMV DS: y = A * x ---

struct SpmvParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> spmv_values_hi: array<f32>;
@group(0) @binding(1) var<storage, read> spmv_values_lo: array<f32>;
@group(0) @binding(2) var<storage, read> spmv_col_indices: array<u32>;
@group(0) @binding(3) var<storage, read> spmv_row_pointers: array<u32>;
@group(0) @binding(4) var<storage, read> spmv_x_hi: array<f32>;
@group(0) @binding(5) var<storage, read> spmv_x_lo: array<f32>;
@group(0) @binding(6) var<storage, read_write> spmv_y_hi: array<f32>;
@group(0) @binding(7) var<storage, read_write> spmv_y_lo: array<f32>;
@group(0) @binding(8) var<uniform> spmv_params: SpmvParams;

@compute @workgroup_size(64)
fn spmv_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= spmv_params.n {
        return;
    }
    let row_start = spmv_row_pointers[row];
    let row_end = spmv_row_pointers[row + 1u];
    var sum_hi: f32 = 0.0;
    var sum_lo: f32 = 0.0;
    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col = spmv_col_indices[idx];
        let p = ds_mul(spmv_values_hi[idx], spmv_values_lo[idx],
                       spmv_x_hi[col], spmv_x_lo[col]);
        let s = ds_add(sum_hi, sum_lo, p.x, p.y);
        sum_hi = s.x;
        sum_lo = s.y;
    }
    spmv_y_hi[row] = sum_hi;
    spmv_y_lo[row] = sum_lo;
}

// --- Dot product DS: partial reduction ---

const DOT_WG_SIZE: u32 = 64u;

var<workgroup> dot_scratch_hi: array<f32, 64>;
var<workgroup> dot_scratch_lo: array<f32, 64>;

struct VecParams {
    alpha_hi: f32,
    alpha_lo: f32,
    n: u32,
}

@group(0) @binding(0) var<storage, read> dot_x_hi: array<f32>;
@group(0) @binding(1) var<storage, read> dot_x_lo: array<f32>;
@group(0) @binding(2) var<storage, read> dot_y_hi: array<f32>;
@group(0) @binding(3) var<storage, read> dot_y_lo: array<f32>;
@group(0) @binding(4) var<storage, read_write> dot_out_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> dot_out_lo: array<f32>;
@group(0) @binding(6) var<uniform> dot_params: VecParams;

@compute @workgroup_size(64)
fn dot_ds(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    if i < dot_params.n {
        let p = ds_mul(dot_x_hi[i], dot_x_lo[i], dot_y_hi[i], dot_y_lo[i]);
        dot_scratch_hi[local_id] = p.x;
        dot_scratch_lo[local_id] = p.y;
    } else {
        dot_scratch_hi[local_id] = 0.0;
        dot_scratch_lo[local_id] = 0.0;
    }
    workgroupBarrier();

    var stride = DOT_WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            let s = ds_add(dot_scratch_hi[local_id], dot_scratch_lo[local_id],
                          dot_scratch_hi[local_id + stride], dot_scratch_lo[local_id + stride]);
            dot_scratch_hi[local_id] = s.x;
            dot_scratch_lo[local_id] = s.y;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if local_id == 0u {
        dot_out_hi[wid.x] = dot_scratch_hi[0];
        dot_out_lo[wid.x] = dot_scratch_lo[0];
    }
}

// --- AXPY DS: y = alpha * x + y ---

@group(0) @binding(0) var<storage, read> axpy_x_hi: array<f32>;
@group(0) @binding(1) var<storage, read> axpy_x_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> axpy_y_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> axpy_y_lo: array<f32>;
@group(0) @binding(4) var<uniform> axpy_params: VecParams;

@compute @workgroup_size(64)
fn axpy_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_params.n {
        return;
    }
    let p = ds_mul(axpy_params.alpha_hi, axpy_params.alpha_lo,
                   axpy_x_hi[i], axpy_x_lo[i]);
    let s = ds_add(p.x, p.y, axpy_y_hi[i], axpy_y_lo[i]);
    axpy_y_hi[i] = s.x;
    axpy_y_lo[i] = s.y;
}

// --- Scale DS: x = alpha * x ---

@group(0) @binding(0) var<storage, read_write> scale_x_hi: array<f32>;
@group(0) @binding(1) var<storage, read_write> scale_x_lo: array<f32>;
@group(0) @binding(2) var<uniform> scale_params: VecParams;

@compute @workgroup_size(64)
fn scale_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= scale_params.n {
        return;
    }
    let p = ds_mul(scale_params.alpha_hi, scale_params.alpha_lo,
                   scale_x_hi[i], scale_x_lo[i]);
    scale_x_hi[i] = p.x;
    scale_x_lo[i] = p.y;
}

// --- Copy DS: y = x ---

@group(0) @binding(0) var<storage, read> copy_x_hi: array<f32>;
@group(0) @binding(1) var<storage, read> copy_x_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> copy_y_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> copy_y_lo: array<f32>;
@group(0) @binding(4) var<uniform> copy_params: VecParams;

@compute @workgroup_size(64)
fn copy_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= copy_params.n {
        return;
    }
    copy_y_hi[i] = copy_x_hi[i];
    copy_y_lo[i] = copy_x_lo[i];
}

// --- Jacobi DS: z = inv_diag * r ---

@group(0) @binding(0) var<storage, read> jac_diag_hi: array<f32>;
@group(0) @binding(1) var<storage, read> jac_diag_lo: array<f32>;
@group(0) @binding(2) var<storage, read> jac_r_hi: array<f32>;
@group(0) @binding(3) var<storage, read> jac_r_lo: array<f32>;
@group(0) @binding(4) var<storage, read_write> jac_z_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> jac_z_lo: array<f32>;
@group(0) @binding(6) var<uniform> jac_params: VecParams;

@compute @workgroup_size(64)
fn jacobi_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= jac_params.n {
        return;
    }
    let p = ds_mul(jac_diag_hi[i], jac_diag_lo[i], jac_r_hi[i], jac_r_lo[i]);
    jac_z_hi[i] = p.x;
    jac_z_lo[i] = p.y;
}
"#;

/// Concatenated DS primitives + linear algebra body.
/// This is the full shader source used by the DS linear solver backend.
pub static DS_SHADER_SOURCE: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{}\n{}", DS_PRIMITIVES, DS_LINEAR_BODY));

/// Build a complete DS shader by concatenating DS_PRIMITIVES with a shader body.
pub fn ds_shader(body: &str) -> String {
    format!("{}\n{}", DS_PRIMITIVES, body)
}

/// Fused BiCGSTAB shader: runs the entire iterative solve in a single
/// workgroup (N ≤ 64). Eliminates all CPU-GPU synchronization overhead
/// by performing dot product reductions in shared memory instead of
/// readbacks.
const DS_BICGSTAB_FUSED_BODY: &str = r#"
struct BicgstabParams {
    n: u32,
    max_iters: u32,
    tol_hi: f32,
    tol_lo: f32,
}

@group(0) @binding(0) var<storage, read> a_val_hi: array<f32>;
@group(0) @binding(1) var<storage, read> a_val_lo: array<f32>;
@group(0) @binding(2) var<storage, read> a_col: array<u32>;
@group(0) @binding(3) var<storage, read> a_row: array<u32>;
@group(0) @binding(4) var<storage, read> rhs_hi: array<f32>;
@group(0) @binding(5) var<storage, read> rhs_lo: array<f32>;
@group(0) @binding(6) var<storage, read_write> sol_hi: array<f32>;
@group(0) @binding(7) var<storage, read_write> sol_lo: array<f32>;
@group(0) @binding(8) var<uniform> bp: BicgstabParams;
@group(0) @binding(9) var<storage, read_write> bresult: array<u32>;

// Shared memory for SpMV vector exchange and dot product reduction.
var<workgroup> sh_hi: array<f32, 64>;
var<workgroup> sh_lo: array<f32, 64>;

@compute @workgroup_size(64)
fn bicgstab_fused(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = bp.n;

    // Compute Jacobi inverse diagonal from matrix diagonal.
    var inv_d_hi: f32 = 0.0;
    var inv_d_lo: f32 = 0.0;
    if tid < n {
        let rs = a_row[tid];
        let re = a_row[tid + 1u];
        for (var idx = rs; idx < re; idx++) {
            if a_col[idx] == tid {
                let dh = a_val_hi[idx];
                let dl = a_val_lo[idx];
                if abs(dh) > 1e-30 {
                    let inv = ds_div(1.0, 0.0, dh, dl);
                    inv_d_hi = inv.x;
                    inv_d_lo = inv.y;
                }
                break;
            }
        }
    }

    // Initialize: x = 0, r = b, r_hat = b, p = 0, v = 0.
    var my_x_hi: f32 = 0.0;
    var my_x_lo: f32 = 0.0;
    var my_r_hi: f32 = 0.0;
    var my_r_lo: f32 = 0.0;
    var my_rh_hi: f32 = 0.0;
    var my_rh_lo: f32 = 0.0;
    var my_p_hi: f32 = 0.0;
    var my_p_lo: f32 = 0.0;
    var my_v_hi: f32 = 0.0;
    var my_v_lo: f32 = 0.0;

    if tid < n {
        my_r_hi = rhs_hi[tid];
        my_r_lo = rhs_lo[tid];
        my_rh_hi = my_r_hi;
        my_rh_lo = my_r_lo;
    }

    // ||b||^2 for relative tolerance.
    var prod: vec2<f32>;
    if tid < n {
        prod = ds_mul(my_r_hi, my_r_lo, my_r_hi, my_r_lo);
    } else {
        prod = vec2(0.0, 0.0);
    }
    sh_hi[tid] = prod.x;
    sh_lo[tid] = prod.y;
    workgroupBarrier();
    var stride = 32u;
    while stride > 0u {
        if tid < stride {
            let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
            sh_hi[tid] = s.x;
            sh_lo[tid] = s.y;
        }
        workgroupBarrier();
        stride /= 2u;
    }
    let bnorm_hi = sh_hi[0];
    let bnorm_lo = sh_lo[0];
    workgroupBarrier();

    if bnorm_hi == 0.0 && bnorm_lo == 0.0 {
        if tid < n { sol_hi[tid] = 0.0; sol_lo[tid] = 0.0; }
        if tid == 0u { bresult[0] = 1u; bresult[1] = 0u; }
        return;
    }

    let tol_sq = ds_mul(bp.tol_hi, bp.tol_lo, bp.tol_hi, bp.tol_lo);
    let thresh = ds_mul(tol_sq.x, tol_sq.y, bnorm_hi, bnorm_lo);

    var rho_hi: f32 = 1.0;
    var rho_lo: f32 = 0.0;
    var alpha_hi: f32 = 1.0;
    var alpha_lo: f32 = 0.0;
    var omega_hi: f32 = 1.0;
    var omega_lo: f32 = 0.0;

    for (var iter = 0u; iter < bp.max_iters; iter++) {
        // rho_new = dot(r_hat, r)
        if tid < n {
            prod = ds_mul(my_rh_hi, my_rh_lo, my_r_hi, my_r_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let rn_hi = sh_hi[0];
        let rn_lo = sh_lo[0];
        workgroupBarrier();

        if abs(rn_hi) < 1e-30 && abs(rn_lo) < 1e-30 {
            if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
            if tid == 0u { bresult[0] = 2u; bresult[1] = iter; }
            return;
        }

        // beta = (rho_new / rho) * (alpha / omega)
        let bp1 = ds_div(rn_hi, rn_lo, rho_hi, rho_lo);
        let bp2 = ds_div(alpha_hi, alpha_lo, omega_hi, omega_lo);
        let beta = ds_mul(bp1.x, bp1.y, bp2.x, bp2.y);
        rho_hi = rn_hi;
        rho_lo = rn_lo;

        // p = r + beta * (p - omega * v)
        if tid < n {
            let ov = ds_mul(omega_hi, omega_lo, my_v_hi, my_v_lo);
            let pov = ds_sub(my_p_hi, my_p_lo, ov.x, ov.y);
            let bp3 = ds_mul(beta.x, beta.y, pov.x, pov.y);
            let np = ds_add(my_r_hi, my_r_lo, bp3.x, bp3.y);
            my_p_hi = np.x;
            my_p_lo = np.y;
        }

        // p_hat = M^{-1} * p (Jacobi)
        var ph_hi: f32 = 0.0;
        var ph_lo: f32 = 0.0;
        if tid < n {
            let ph = ds_mul(inv_d_hi, inv_d_lo, my_p_hi, my_p_lo);
            ph_hi = ph.x;
            ph_lo = ph.y;
        }

        // v = A * p_hat (SpMV via shared memory)
        sh_hi[tid] = ph_hi;
        sh_lo[tid] = ph_lo;
        workgroupBarrier();
        my_v_hi = 0.0;
        my_v_lo = 0.0;
        if tid < n {
            let rs = a_row[tid];
            let re = a_row[tid + 1u];
            for (var idx = rs; idx < re; idx++) {
                let c = a_col[idx];
                let mp = ds_mul(a_val_hi[idx], a_val_lo[idx], sh_hi[c], sh_lo[c]);
                let sa = ds_add(my_v_hi, my_v_lo, mp.x, mp.y);
                my_v_hi = sa.x;
                my_v_lo = sa.y;
            }
        }
        workgroupBarrier();

        // alpha = rho / dot(r_hat, v)
        if tid < n {
            prod = ds_mul(my_rh_hi, my_rh_lo, my_v_hi, my_v_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let rhv_hi = sh_hi[0];
        let rhv_lo = sh_lo[0];
        workgroupBarrier();

        if abs(rhv_hi) < 1e-30 && abs(rhv_lo) < 1e-30 {
            if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
            if tid == 0u { bresult[0] = 2u; bresult[1] = iter; }
            return;
        }

        let na = ds_div(rho_hi, rho_lo, rhv_hi, rhv_lo);
        alpha_hi = na.x;
        alpha_lo = na.y;

        // s = r - alpha * v
        var my_s_hi: f32 = 0.0;
        var my_s_lo: f32 = 0.0;
        if tid < n {
            let av = ds_mul(alpha_hi, alpha_lo, my_v_hi, my_v_lo);
            let ns = ds_sub(my_r_hi, my_r_lo, av.x, av.y);
            my_s_hi = ns.x;
            my_s_lo = ns.y;
        }

        // Early convergence: ||s||^2
        if tid < n {
            prod = ds_mul(my_s_hi, my_s_lo, my_s_hi, my_s_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let sn_hi = sh_hi[0];
        let sn_lo = sh_lo[0];
        workgroupBarrier();

        if ds_lt(sn_hi, sn_lo, thresh.x, thresh.y) {
            if tid < n {
                let ap = ds_mul(alpha_hi, alpha_lo, ph_hi, ph_lo);
                let nx = ds_add(my_x_hi, my_x_lo, ap.x, ap.y);
                sol_hi[tid] = nx.x;
                sol_lo[tid] = nx.y;
            }
            if tid == 0u { bresult[0] = 1u; bresult[1] = iter + 1u; }
            return;
        }

        // s_hat = M^{-1} * s (Jacobi)
        var sh2_hi: f32 = 0.0;
        var sh2_lo: f32 = 0.0;
        if tid < n {
            let sh2 = ds_mul(inv_d_hi, inv_d_lo, my_s_hi, my_s_lo);
            sh2_hi = sh2.x;
            sh2_lo = sh2.y;
        }

        // t = A * s_hat (SpMV)
        sh_hi[tid] = sh2_hi;
        sh_lo[tid] = sh2_lo;
        workgroupBarrier();
        var my_t_hi: f32 = 0.0;
        var my_t_lo: f32 = 0.0;
        if tid < n {
            let rs = a_row[tid];
            let re = a_row[tid + 1u];
            for (var idx = rs; idx < re; idx++) {
                let c = a_col[idx];
                let mp = ds_mul(a_val_hi[idx], a_val_lo[idx], sh_hi[c], sh_lo[c]);
                let sa = ds_add(my_t_hi, my_t_lo, mp.x, mp.y);
                my_t_hi = sa.x;
                my_t_lo = sa.y;
            }
        }
        workgroupBarrier();

        // omega = dot(t, s) / dot(t, t)
        if tid < n {
            prod = ds_mul(my_t_hi, my_t_lo, my_s_hi, my_s_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let ts_hi = sh_hi[0];
        let ts_lo = sh_lo[0];
        workgroupBarrier();

        if tid < n {
            prod = ds_mul(my_t_hi, my_t_lo, my_t_hi, my_t_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let tt_hi = sh_hi[0];
        let tt_lo = sh_lo[0];
        workgroupBarrier();

        if abs(tt_hi) < 1e-30 && abs(tt_lo) < 1e-30 {
            // t ≈ 0: accept x + alpha*p_hat
            if tid < n {
                let ap = ds_mul(alpha_hi, alpha_lo, ph_hi, ph_lo);
                let nx = ds_add(my_x_hi, my_x_lo, ap.x, ap.y);
                sol_hi[tid] = nx.x;
                sol_lo[tid] = nx.y;
            }
            if tid == 0u { bresult[0] = 1u; bresult[1] = iter + 1u; }
            return;
        }

        let nw = ds_div(ts_hi, ts_lo, tt_hi, tt_lo);
        omega_hi = nw.x;
        omega_lo = nw.y;

        // x = x + alpha * p_hat + omega * s_hat
        if tid < n {
            let ap = ds_mul(alpha_hi, alpha_lo, ph_hi, ph_lo);
            let os = ds_mul(omega_hi, omega_lo, sh2_hi, sh2_lo);
            let nx1 = ds_add(my_x_hi, my_x_lo, ap.x, ap.y);
            let nx2 = ds_add(nx1.x, nx1.y, os.x, os.y);
            my_x_hi = nx2.x;
            my_x_lo = nx2.y;
        }

        // r = s - omega * t
        if tid < n {
            let ot = ds_mul(omega_hi, omega_lo, my_t_hi, my_t_lo);
            let nr = ds_sub(my_s_hi, my_s_lo, ot.x, ot.y);
            my_r_hi = nr.x;
            my_r_lo = nr.y;
        }

        // Convergence: ||r||^2
        if tid < n {
            prod = ds_mul(my_r_hi, my_r_lo, my_r_hi, my_r_lo);
        } else {
            prod = vec2(0.0, 0.0);
        }
        sh_hi[tid] = prod.x;
        sh_lo[tid] = prod.y;
        workgroupBarrier();
        stride = 32u;
        while stride > 0u {
            if tid < stride {
                let s = ds_add(sh_hi[tid], sh_lo[tid], sh_hi[tid + stride], sh_lo[tid + stride]);
                sh_hi[tid] = s.x;
                sh_lo[tid] = s.y;
            }
            workgroupBarrier();
            stride /= 2u;
        }
        let rn2_hi = sh_hi[0];
        let rn2_lo = sh_lo[0];
        workgroupBarrier();

        // NaN check
        if rn2_hi != rn2_hi {
            if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
            if tid == 0u { bresult[0] = 3u; bresult[1] = iter; }
            return;
        }

        if ds_lt(rn2_hi, rn2_lo, thresh.x, thresh.y) {
            if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
            if tid == 0u { bresult[0] = 1u; bresult[1] = iter + 1u; }
            return;
        }

        if abs(omega_hi) < 1e-30 && abs(omega_lo) < 1e-30 {
            if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
            if tid == 0u { bresult[0] = 2u; bresult[1] = iter; }
            return;
        }
    }

    // Not converged
    if tid < n { sol_hi[tid] = my_x_hi; sol_lo[tid] = my_x_lo; }
    if tid == 0u { bresult[0] = 0u; bresult[1] = bp.max_iters; }
}
"#;

/// Full source for the fused BiCGSTAB shader (DS primitives + fused body).
pub static DS_BICGSTAB_FUSED_SOURCE: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| ds_shader(DS_BICGSTAB_FUSED_BODY));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds_shader_parses_successfully() {
        let source = &*DS_SHADER_SOURCE;
        let result = naga::front::wgsl::parse_str(source);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                let expected = [
                    "spmv_ds",
                    "dot_ds",
                    "axpy_ds",
                    "scale_ds",
                    "copy_ds",
                    "jacobi_ds",
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
                    "DS WGSL parse error:\n{}",
                    e.emit_to_string(source)
                );
            }
        }
    }

    #[test]
    fn ds_bicgstab_fused_parses_successfully() {
        let source = &*DS_BICGSTAB_FUSED_SOURCE;
        let result = naga::front::wgsl::parse_str(source);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                assert!(
                    entry_names.contains(&"bicgstab_fused"),
                    "missing entry point: bicgstab_fused. Found: {entry_names:?}"
                );
            }
            Err(e) => {
                panic!(
                    "Fused BiCGSTAB WGSL parse error:\n{}",
                    e.emit_to_string(source)
                );
            }
        }
    }

    #[test]
    fn ds_primitives_parse_standalone() {
        // DS_PRIMITIVES alone has no entry points but should parse as valid WGSL.
        let result = naga::front::wgsl::parse_str(DS_PRIMITIVES);
        match result {
            Ok(_) => {}
            Err(e) => {
                panic!(
                    "DS_PRIMITIVES parse error:\n{}",
                    e.emit_to_string(DS_PRIMITIVES)
                );
            }
        }
    }
}
