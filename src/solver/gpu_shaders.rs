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
}
