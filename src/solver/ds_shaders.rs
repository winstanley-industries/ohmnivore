//! WGSL compute shaders using double-single (DS) arithmetic for ~f64 precision.
//!
//! Each value is represented as a (hi, lo) pair of f32 values where
//! true_value = hi + lo. This gives ~48 bits of mantissa (vs f32's 24),
//! enabling BiCGSTAB convergence on ill-conditioned matrices (kappa ~ 1e12).
//!
//! DS primitives (TwoSum, TwoProd) rely on IEEE 754 rounding guarantees
//! for f32 add/sub/mul/fma, which WGSL provides.

/// DS linear algebra shaders: spmv, dot, axpy, scale, copy, jacobi.
///
/// Entry points:
/// - `spmv_ds`: y = A * x (CSR, DS precision)
/// - `dot_ds`: partial dot product reduction (DS precision)
/// - `axpy_ds`: y = alpha * x + y (DS precision)
/// - `scale_ds`: x = alpha * x (DS precision)
/// - `copy_ds`: y = x (DS copy, both hi and lo)
/// - `jacobi_ds`: z = inv_diag * r (DS element-wise multiply)
pub const DS_SHADER_SOURCE: &str = r#"
// ============================================================
// Double-Single (DS) Arithmetic GPU Compute Shaders
// ============================================================
//
// DS representation: value ≈ hi + lo (two f32s give ~48-bit mantissa)
// Primitives: TwoSum (Knuth), TwoProd (via fma)

// --- DS Primitives ---

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds_shader_parses_successfully() {
        let result = naga::front::wgsl::parse_str(DS_SHADER_SOURCE);
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
                panic!("DS WGSL parse error:\n{}", e.emit_to_string(DS_SHADER_SOURCE));
            }
        }
    }
}
