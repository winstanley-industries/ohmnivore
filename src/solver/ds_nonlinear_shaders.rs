//! DS-precision WGSL shader bodies for nonlinear device evaluation, assembly,
//! voltage limiting, and convergence checking.
//!
//! Each constant contains only the shader body (bindings + entry points).
//! DS primitives are prepended at pipeline creation via `ds_shaders::ds_shader()`.
//!
//! Convention: solution vectors and eval outputs use hi/lo pairs. Descriptors
//! remain u32 (model parameters are inherently f32-precision from SPICE cards).

/// DS diode evaluation shader body.
///
/// Entry point: `diode_eval_ds`
/// Input: descriptors (u32), solution hi/lo
/// Output: 4 floats per diode: [i_d_hi, i_d_lo, g_d_hi, g_d_lo]
pub const DS_DIODE_EVAL_BODY: &str = r#"
// ============================================================
// Diode Evaluation Shader (DS precision)
// ============================================================

struct DiodeEvalParams {
    num_diodes: u32,
}

@group(0) @binding(0) var<storage, read> diode_desc: array<u32>;
@group(0) @binding(1) var<storage, read> eval_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> eval_x_lo: array<f32>;
// Output: 4 floats per diode [i_d_hi, i_d_lo, g_d_hi, g_d_lo]
@group(0) @binding(3) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(4) var<uniform> eval_params: DiodeEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(eval_x_hi[idx], eval_x_lo[idx]);
}

@compute @workgroup_size(64)
fn diode_eval_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= eval_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = diode_desc[base + 0u];
    let cathode_idx = diode_desc[base + 1u];
    let is_val = bitcast<f32>(diode_desc[base + 2u]);
    let n_vt = bitcast<f32>(diode_desc[base + 3u]);

    let v_a = read_voltage_ds(anode_idx);
    let v_c = read_voltage_ds(cathode_idx);
    let v_d = ds_sub(v_a.x, v_a.y, v_c.x, v_c.y);

    // exponent = v_d / n_vt (DS division)
    let exponent = ds_div(v_d.x, v_d.y, n_vt, 0.0);

    // e = exp(exponent) (DS exp with clamping)
    let e = ds_exp(exponent.x, exponent.y);

    // i_d = is_val * (e - 1.0)
    let e_m1 = ds_sub(e.x, e.y, 1.0, 0.0);
    let i_d = ds_mul(is_val, 0.0, e_m1.x, e_m1.y);

    // g_d = is_val * e / n_vt
    let is_e = ds_mul(is_val, 0.0, e.x, e.y);
    let g_d = ds_div(is_e.x, is_e.y, n_vt, 0.0);

    let out = 4u * i;
    eval_output[out + 0u] = i_d.x;
    eval_output[out + 1u] = i_d.y;
    eval_output[out + 2u] = g_d.x;
    eval_output[out + 3u] = g_d.y;
}
"#;

/// DS nonlinear assembly shader body.
///
/// Entry points:
/// - `assemble_matrix_copy_ds`: Copy base matrix values (DS hi/lo)
/// - `assemble_matrix_stamp_ds`: Stamp diode conductance (DS)
/// - `assemble_rhs_copy_ds`: Copy base RHS values (DS hi/lo)
/// - `assemble_rhs_stamp_ds`: Stamp Norton companion currents (DS)
pub const DS_NONLINEAR_ASSEMBLE_BODY: &str = r#"
// ============================================================
// Nonlinear Assembly Shader (DS precision)
// ============================================================

struct AssembleMatParams {
    nnz: u32,
}

// --- Matrix base copy: out = base (DS) ---

@group(0) @binding(0) var<storage, read> base_values_hi: array<f32>;
@group(0) @binding(1) var<storage, read> base_values_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_values_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_values_lo: array<f32>;
@group(0) @binding(4) var<uniform> assemble_mat_params: AssembleMatParams;

@compute @workgroup_size(64)
fn assemble_matrix_copy_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= assemble_mat_params.nnz {
        return;
    }
    out_values_hi[i] = base_values_hi[i];
    out_values_lo[i] = base_values_lo[i];
}

// --- Matrix stamp: add g_d (DS) into 4 CSR positions per diode ---

struct StampMatParams {
    num_diodes: u32,
}

@group(0) @binding(0) var<storage, read> stamp_desc: array<u32>;
// Diode eval results: [i_d_hi, i_d_lo, g_d_hi, g_d_lo] per diode
@group(0) @binding(1) var<storage, read> stamp_eval: array<f32>;
@group(0) @binding(2) var<storage, read_write> stamp_values_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> stamp_values_lo: array<f32>;
@group(0) @binding(4) var<uniform> stamp_mat_params: StampMatParams;

const STAMP_SENTINEL: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn assemble_matrix_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= stamp_mat_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let g_d_hi = stamp_eval[4u * i + 2u];
    let g_d_lo = stamp_eval[4u * i + 3u];

    let pos0 = stamp_desc[base + 4u];
    let pos1 = stamp_desc[base + 5u];
    let pos2 = stamp_desc[base + 6u];
    let pos3 = stamp_desc[base + 7u];

    if pos0 != STAMP_SENTINEL {
        let s = ds_add(stamp_values_hi[pos0], stamp_values_lo[pos0], g_d_hi, g_d_lo);
        stamp_values_hi[pos0] = s.x;
        stamp_values_lo[pos0] = s.y;
    }
    if pos1 != STAMP_SENTINEL {
        let s = ds_sub(stamp_values_hi[pos1], stamp_values_lo[pos1], g_d_hi, g_d_lo);
        stamp_values_hi[pos1] = s.x;
        stamp_values_lo[pos1] = s.y;
    }
    if pos2 != STAMP_SENTINEL {
        let s = ds_sub(stamp_values_hi[pos2], stamp_values_lo[pos2], g_d_hi, g_d_lo);
        stamp_values_hi[pos2] = s.x;
        stamp_values_lo[pos2] = s.y;
    }
    if pos3 != STAMP_SENTINEL {
        let s = ds_add(stamp_values_hi[pos3], stamp_values_lo[pos3], g_d_hi, g_d_lo);
        stamp_values_hi[pos3] = s.x;
        stamp_values_lo[pos3] = s.y;
    }
}

// --- RHS base copy (DS) ---

struct AssembleRhsParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> base_b_hi: array<f32>;
@group(0) @binding(1) var<storage, read> base_b_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_b_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_b_lo: array<f32>;
@group(0) @binding(4) var<uniform> assemble_rhs_params: AssembleRhsParams;

@compute @workgroup_size(64)
fn assemble_rhs_copy_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= assemble_rhs_params.n {
        return;
    }
    out_b_hi[i] = base_b_hi[i];
    out_b_lo[i] = base_b_lo[i];
}

// --- RHS stamp: Norton companion current per diode (DS) ---

struct StampRhsParams {
    num_diodes: u32,
}

@group(0) @binding(0) var<storage, read> rhs_desc: array<u32>;
@group(0) @binding(1) var<storage, read> rhs_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> rhs_x_lo: array<f32>;
@group(0) @binding(3) var<storage, read> rhs_eval: array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs_b_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs_b_lo: array<f32>;
@group(0) @binding(6) var<uniform> stamp_rhs_params: StampRhsParams;

const RHS_SENTINEL: u32 = 0xFFFFFFFFu;

fn rhs_read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == RHS_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(rhs_x_hi[idx], rhs_x_lo[idx]);
}

@compute @workgroup_size(64)
fn assemble_rhs_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= stamp_rhs_params.num_diodes {
        return;
    }

    let base = i * 12u;
    let anode_idx = rhs_desc[base + 0u];
    let cathode_idx = rhs_desc[base + 1u];

    let i_d_hi = rhs_eval[4u * i + 0u];
    let i_d_lo = rhs_eval[4u * i + 1u];
    let g_d_hi = rhs_eval[4u * i + 2u];
    let g_d_lo = rhs_eval[4u * i + 3u];

    let v_a = rhs_read_voltage_ds(anode_idx);
    let v_c = rhs_read_voltage_ds(cathode_idx);
    let v_d = ds_sub(v_a.x, v_a.y, v_c.x, v_c.y);

    // Norton companion: i_eq = i_d - g_d * v_d
    let gv = ds_mul(g_d_hi, g_d_lo, v_d.x, v_d.y);
    let i_eq = ds_sub(i_d_hi, i_d_lo, gv.x, gv.y);

    let b_idx_anode = rhs_desc[base + 8u];
    let b_idx_cathode = rhs_desc[base + 9u];

    if b_idx_anode != RHS_SENTINEL {
        let s = ds_sub(rhs_b_hi[b_idx_anode], rhs_b_lo[b_idx_anode], i_eq.x, i_eq.y);
        rhs_b_hi[b_idx_anode] = s.x;
        rhs_b_lo[b_idx_anode] = s.y;
    }
    if b_idx_cathode != RHS_SENTINEL {
        let s = ds_add(rhs_b_hi[b_idx_cathode], rhs_b_lo[b_idx_cathode], i_eq.x, i_eq.y);
        rhs_b_hi[b_idx_cathode] = s.x;
        rhs_b_lo[b_idx_cathode] = s.y;
    }
}
"#;

/// DS BJT evaluation shader body.
///
/// Entry point: `bjt_eval_ds`
/// Output: 12 floats per BJT (6 DS pairs: I_C, I_B, 4 Jacobians)
pub const DS_BJT_EVAL_BODY: &str = r#"
// ============================================================
// BJT Evaluation Shader (Ebers-Moll Level 1, DS precision)
// ============================================================

struct BjtEvalParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_desc: array<u32>;
@group(0) @binding(1) var<storage, read> eval_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> eval_x_lo: array<f32>;
// Output: 12 floats per BJT (6 DS pairs)
@group(0) @binding(3) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(4) var<uniform> eval_params: BjtEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(eval_x_hi[idx], eval_x_lo[idx]);
}

@compute @workgroup_size(64)
fn bjt_eval_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let v_b = read_voltage_ds(base_idx);
    let v_e = read_voltage_ds(emitter_idx);
    let v_c = read_voltage_ds(collector_idx);

    // v_be = polarity * (V_B - V_E)
    let vbe_raw = ds_sub(v_b.x, v_b.y, v_e.x, v_e.y);
    let v_be = ds_mul(polarity, 0.0, vbe_raw.x, vbe_raw.y);
    // v_bc = polarity * (V_B - V_C)
    let vbc_raw = ds_sub(v_b.x, v_b.y, v_c.x, v_c.y);
    let v_bc = ds_mul(polarity, 0.0, vbc_raw.x, vbc_raw.y);

    let exp_f_arg = ds_div(v_be.x, v_be.y, nf_vt, 0.0);
    let e_f = ds_exp(exp_f_arg.x, exp_f_arg.y);
    let exp_r_arg = ds_div(v_bc.x, v_bc.y, nr_vt, 0.0);
    let e_r = ds_exp(exp_r_arg.x, exp_r_arg.y);

    // i_f = is_val * (e_f - 1)
    let ef_m1 = ds_sub(e_f.x, e_f.y, 1.0, 0.0);
    let i_f = ds_mul(is_val, 0.0, ef_m1.x, ef_m1.y);
    // i_r = is_val * (e_r - 1)
    let er_m1 = ds_sub(e_r.x, e_r.y, 1.0, 0.0);
    let i_r = ds_mul(is_val, 0.0, er_m1.x, er_m1.y);

    // i_c = bf/(bf+1) * i_f - i_r/(br+1)
    let bf_frac = ds_div(bf, 0.0, bf + 1.0, 0.0);
    let br1 = br + 1.0;
    let term1 = ds_mul(bf_frac.x, bf_frac.y, i_f.x, i_f.y);
    let ir_br1 = ds_div(i_r.x, i_r.y, br1, 0.0);
    let i_c = ds_sub(term1.x, term1.y, ir_br1.x, ir_br1.y);

    // i_b = i_f/(bf+1) + i_r/(br+1)
    let bf1 = bf + 1.0;
    let if_bf1 = ds_div(i_f.x, i_f.y, bf1, 0.0);
    let i_b = ds_add(if_bf1.x, if_bf1.y, ir_br1.x, ir_br1.y);

    // Jacobians
    // di_c_dvbe = bf/(bf+1) * is * e_f / nf_vt
    let is_ef = ds_mul(is_val, 0.0, e_f.x, e_f.y);
    let is_ef_nfvt = ds_div(is_ef.x, is_ef.y, nf_vt, 0.0);
    let di_c_dvbe = ds_mul(bf_frac.x, bf_frac.y, is_ef_nfvt.x, is_ef_nfvt.y);

    // di_c_dvbc = -is * e_r / nr_vt / (br+1)
    let is_er = ds_mul(is_val, 0.0, e_r.x, e_r.y);
    let is_er_nrvt = ds_div(is_er.x, is_er.y, nr_vt, 0.0);
    let is_er_nrvt_br1 = ds_div(is_er_nrvt.x, is_er_nrvt.y, br1, 0.0);
    let di_c_dvbc = ds_neg(is_er_nrvt_br1.x, is_er_nrvt_br1.y);

    // di_b_dvbe = is * e_f / nf_vt / (bf+1)
    let di_b_dvbe = ds_div(is_ef_nfvt.x, is_ef_nfvt.y, bf1, 0.0);

    // di_b_dvbc = is * e_r / nr_vt / (br+1)
    let di_b_dvbc = is_er_nrvt_br1;

    // Write output with polarity applied to currents
    let out = 12u * i;
    let ic_pol = ds_mul(i_c.x, i_c.y, polarity, 0.0);
    let ib_pol = ds_mul(i_b.x, i_b.y, polarity, 0.0);
    eval_output[out + 0u] = ic_pol.x;
    eval_output[out + 1u] = ic_pol.y;
    eval_output[out + 2u] = ib_pol.x;
    eval_output[out + 3u] = ib_pol.y;
    eval_output[out + 4u] = di_c_dvbe.x;
    eval_output[out + 5u] = di_c_dvbe.y;
    eval_output[out + 6u] = di_c_dvbc.x;
    eval_output[out + 7u] = di_c_dvbc.y;
    eval_output[out + 8u] = di_b_dvbe.x;
    eval_output[out + 9u] = di_b_dvbe.y;
    eval_output[out + 10u] = di_b_dvbc.x;
    eval_output[out + 11u] = di_b_dvbc.y;
}
"#;

/// DS BJT assembly shader body.
///
/// Entry points: `assemble_bjt_matrix_stamp_ds`, `assemble_bjt_rhs_stamp_ds`
pub const DS_BJT_ASSEMBLE_BODY: &str = r#"
// ============================================================
// BJT Assembly Shader (DS precision)
// ============================================================

const SENTINEL: u32 = 0xFFFFFFFFu;

// --- Matrix stamp: 9 G-matrix entries per BJT (DS) ---

struct BjtStampMatParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_mat_desc: array<u32>;
// Eval output: 12 floats per BJT (6 DS pairs)
@group(0) @binding(1) var<storage, read> bjt_mat_eval: array<f32>;
@group(0) @binding(2) var<storage, read_write> bjt_mat_values_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> bjt_mat_values_lo: array<f32>;
@group(0) @binding(4) var<uniform> bjt_mat_params: BjtStampMatParams;

@compute @workgroup_size(64)
fn assemble_bjt_matrix_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bjt_mat_params.num_bjts {
        return;
    }

    let desc_base = i * 24u;
    let eval_base = 12u * i;

    // Read 4 Jacobian derivatives (DS pairs)
    let di_c_dvbe_hi = bjt_mat_eval[eval_base + 4u];
    let di_c_dvbe_lo = bjt_mat_eval[eval_base + 5u];
    let di_c_dvbc_hi = bjt_mat_eval[eval_base + 6u];
    let di_c_dvbc_lo = bjt_mat_eval[eval_base + 7u];
    let di_b_dvbe_hi = bjt_mat_eval[eval_base + 8u];
    let di_b_dvbe_lo = bjt_mat_eval[eval_base + 9u];
    let di_b_dvbc_hi = bjt_mat_eval[eval_base + 10u];
    let di_b_dvbc_lo = bjt_mat_eval[eval_base + 11u];

    // Row: Collector
    let g_cc = ds_neg(di_c_dvbc_hi, di_c_dvbc_lo);
    let g_cb = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_c_dvbc_hi, di_c_dvbc_lo);
    let g_ce = ds_neg(di_c_dvbe_hi, di_c_dvbe_lo);

    // Row: Base
    let g_bc = ds_neg(di_b_dvbc_hi, di_b_dvbc_lo);
    let g_bb = ds_add(di_b_dvbe_hi, di_b_dvbe_lo, di_b_dvbc_hi, di_b_dvbc_lo);
    let g_be = ds_neg(di_b_dvbe_hi, di_b_dvbe_lo);

    // Row: Emitter (KCL: I_E = -(I_C + I_B))
    let g_ec = ds_add(di_c_dvbc_hi, di_c_dvbc_lo, di_b_dvbc_hi, di_b_dvbc_lo);
    let sum_all = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_c_dvbc_hi, di_c_dvbc_lo);
    let sum_all2 = ds_add(sum_all.x, sum_all.y, di_b_dvbe_hi, di_b_dvbe_lo);
    let sum_all3 = ds_add(sum_all2.x, sum_all2.y, di_b_dvbc_hi, di_b_dvbc_lo);
    let g_eb = ds_neg(sum_all3.x, sum_all3.y);
    let g_ee = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_b_dvbe_hi, di_b_dvbe_lo);

    // Store stamps as arrays for loop
    var stamps_hi: array<f32, 9>;
    var stamps_lo: array<f32, 9>;
    stamps_hi[0] = g_cc.x; stamps_lo[0] = g_cc.y;
    stamps_hi[1] = g_cb.x; stamps_lo[1] = g_cb.y;
    stamps_hi[2] = g_ce.x; stamps_lo[2] = g_ce.y;
    stamps_hi[3] = g_bc.x; stamps_lo[3] = g_bc.y;
    stamps_hi[4] = g_bb.x; stamps_lo[4] = g_bb.y;
    stamps_hi[5] = g_be.x; stamps_lo[5] = g_be.y;
    stamps_hi[6] = g_ec.x; stamps_lo[6] = g_ec.y;
    stamps_hi[7] = g_eb.x; stamps_lo[7] = g_eb.y;
    stamps_hi[8] = g_ee.x; stamps_lo[8] = g_ee.y;

    for (var j = 0u; j < 9u; j = j + 1u) {
        let pos = bjt_mat_desc[desc_base + 12u + j];
        if pos != SENTINEL {
            let s = ds_add(bjt_mat_values_hi[pos], bjt_mat_values_lo[pos],
                          stamps_hi[j], stamps_lo[j]);
            bjt_mat_values_hi[pos] = s.x;
            bjt_mat_values_lo[pos] = s.y;
        }
    }
}

// --- RHS stamp: Norton companion current per BJT (DS) ---

struct BjtStampRhsParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bjt_rhs_desc: array<u32>;
@group(0) @binding(1) var<storage, read> bjt_rhs_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> bjt_rhs_x_lo: array<f32>;
@group(0) @binding(3) var<storage, read> bjt_rhs_eval: array<f32>;
@group(0) @binding(4) var<storage, read_write> bjt_rhs_b_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> bjt_rhs_b_lo: array<f32>;
@group(0) @binding(6) var<uniform> bjt_rhs_params: BjtStampRhsParams;

fn bjt_read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(bjt_rhs_x_hi[idx], bjt_rhs_x_lo[idx]);
}

@compute @workgroup_size(64)
fn assemble_bjt_rhs_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= bjt_rhs_params.num_bjts {
        return;
    }

    let desc_base = i * 24u;
    let eval_base = 12u * i;

    let collector_idx = bjt_rhs_desc[desc_base + 0u];
    let base_idx = bjt_rhs_desc[desc_base + 1u];
    let emitter_idx = bjt_rhs_desc[desc_base + 2u];

    // Read eval outputs (DS pairs)
    let i_c_hi = bjt_rhs_eval[eval_base + 0u];
    let i_c_lo = bjt_rhs_eval[eval_base + 1u];
    let i_b_hi = bjt_rhs_eval[eval_base + 2u];
    let i_b_lo = bjt_rhs_eval[eval_base + 3u];
    let i_e_raw = ds_add(i_c_hi, i_c_lo, i_b_hi, i_b_lo);
    let i_e = ds_neg(i_e_raw.x, i_e_raw.y);

    let di_c_dvbe_hi = bjt_rhs_eval[eval_base + 4u];
    let di_c_dvbe_lo = bjt_rhs_eval[eval_base + 5u];
    let di_c_dvbc_hi = bjt_rhs_eval[eval_base + 6u];
    let di_c_dvbc_lo = bjt_rhs_eval[eval_base + 7u];
    let di_b_dvbe_hi = bjt_rhs_eval[eval_base + 8u];
    let di_b_dvbe_lo = bjt_rhs_eval[eval_base + 9u];
    let di_b_dvbc_hi = bjt_rhs_eval[eval_base + 10u];
    let di_b_dvbc_lo = bjt_rhs_eval[eval_base + 11u];

    let v_c = bjt_read_voltage_ds(collector_idx);
    let v_b = bjt_read_voltage_ds(base_idx);
    let v_e = bjt_read_voltage_ds(emitter_idx);

    // Collector row stamps
    let g_cc = ds_neg(di_c_dvbc_hi, di_c_dvbc_lo);
    let g_cb = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_c_dvbc_hi, di_c_dvbc_lo);
    let g_ce = ds_neg(di_c_dvbe_hi, di_c_dvbe_lo);

    // Base row stamps
    let g_bc = ds_neg(di_b_dvbc_hi, di_b_dvbc_lo);
    let g_bb = ds_add(di_b_dvbe_hi, di_b_dvbe_lo, di_b_dvbc_hi, di_b_dvbc_lo);
    let g_be = ds_neg(di_b_dvbe_hi, di_b_dvbe_lo);

    // Emitter row stamps
    let g_ec = ds_add(di_c_dvbc_hi, di_c_dvbc_lo, di_b_dvbc_hi, di_b_dvbc_lo);
    let sum_all = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_c_dvbc_hi, di_c_dvbc_lo);
    let sum_all2 = ds_add(sum_all.x, sum_all.y, di_b_dvbe_hi, di_b_dvbe_lo);
    let sum_all3 = ds_add(sum_all2.x, sum_all2.y, di_b_dvbc_hi, di_b_dvbc_lo);
    let g_eb = ds_neg(sum_all3.x, sum_all3.y);
    let g_ee = ds_add(di_c_dvbe_hi, di_c_dvbe_lo, di_b_dvbe_hi, di_b_dvbe_lo);

    // Linearized currents: i_lin = G * v
    let t1 = ds_mul(g_cc.x, g_cc.y, v_c.x, v_c.y);
    let t2 = ds_mul(g_cb.x, g_cb.y, v_b.x, v_b.y);
    let t3 = ds_mul(g_ce.x, g_ce.y, v_e.x, v_e.y);
    let i_lin_c_12 = ds_add(t1.x, t1.y, t2.x, t2.y);
    let i_lin_c = ds_add(i_lin_c_12.x, i_lin_c_12.y, t3.x, t3.y);

    let t4 = ds_mul(g_bc.x, g_bc.y, v_c.x, v_c.y);
    let t5 = ds_mul(g_bb.x, g_bb.y, v_b.x, v_b.y);
    let t6 = ds_mul(g_be.x, g_be.y, v_e.x, v_e.y);
    let i_lin_b_45 = ds_add(t4.x, t4.y, t5.x, t5.y);
    let i_lin_b = ds_add(i_lin_b_45.x, i_lin_b_45.y, t6.x, t6.y);

    let t7 = ds_mul(g_ec.x, g_ec.y, v_c.x, v_c.y);
    let t8 = ds_mul(g_eb.x, g_eb.y, v_b.x, v_b.y);
    let t9 = ds_mul(g_ee.x, g_ee.y, v_e.x, v_e.y);
    let i_lin_e_78 = ds_add(t7.x, t7.y, t8.x, t8.y);
    let i_lin_e = ds_add(i_lin_e_78.x, i_lin_e_78.y, t9.x, t9.y);

    // Norton equivalents: i_eq = i - i_lin
    let i_eq_c = ds_sub(i_c_hi, i_c_lo, i_lin_c.x, i_lin_c.y);
    let i_eq_b = ds_sub(i_b_hi, i_b_lo, i_lin_b.x, i_lin_b.y);
    let i_eq_e = ds_sub(i_e.x, i_e.y, i_lin_e.x, i_lin_e.y);

    // RHS: b[node] -= i_eq
    let b_c = bjt_rhs_desc[desc_base + 21u];
    let b_b = bjt_rhs_desc[desc_base + 22u];
    let b_e = bjt_rhs_desc[desc_base + 23u];

    if b_c != SENTINEL {
        let s = ds_sub(bjt_rhs_b_hi[b_c], bjt_rhs_b_lo[b_c], i_eq_c.x, i_eq_c.y);
        bjt_rhs_b_hi[b_c] = s.x;
        bjt_rhs_b_lo[b_c] = s.y;
    }
    if b_b != SENTINEL {
        let s = ds_sub(bjt_rhs_b_hi[b_b], bjt_rhs_b_lo[b_b], i_eq_b.x, i_eq_b.y);
        bjt_rhs_b_hi[b_b] = s.x;
        bjt_rhs_b_lo[b_b] = s.y;
    }
    if b_e != SENTINEL {
        let s = ds_sub(bjt_rhs_b_hi[b_e], bjt_rhs_b_lo[b_e], i_eq_e.x, i_eq_e.y);
        bjt_rhs_b_hi[b_e] = s.x;
        bjt_rhs_b_lo[b_e] = s.y;
    }
}
"#;

/// DS BJT voltage limiting shader body.
///
/// Entry point: `bjt_voltage_limit_ds`
/// Uses SPICE3f5 `pnjlim` with logarithmic damping for forward-biased junctions.
/// Control flow uses f32 hi-parts; node adjustments use DS arithmetic.
pub const DS_BJT_VOLTAGE_LIMIT_BODY: &str = r#"
// ============================================================
// BJT Voltage Limiting Shader (DS precision, SPICE pnjlim)
// ============================================================

struct BjtVoltLimitParams {
    num_bjts: u32,
}

@group(0) @binding(0) var<storage, read> bvl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> bvl_x_old_hi: array<f32>;
@group(0) @binding(2) var<storage, read> bvl_x_old_lo: array<f32>;
@group(0) @binding(3) var<storage, read_write> bvl_x_new_hi: array<f32>;
@group(0) @binding(4) var<storage, read_write> bvl_x_new_lo: array<f32>;
@group(0) @binding(5) var<uniform> bvl_params: BjtVoltLimitParams;

const BVL_SENTINEL: u32 = 0xFFFFFFFFu;

fn bvl_read_old_ds(idx: u32) -> vec2<f32> {
    if idx == BVL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(bvl_x_old_hi[idx], bvl_x_old_lo[idx]);
}

fn bvl_read_new_ds(idx: u32) -> vec2<f32> {
    if idx == BVL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(bvl_x_new_hi[idx], bvl_x_new_lo[idx]);
}

// SPICE3f5 pnjlim: logarithmic damping for forward-biased PN junctions.
// Uses f32 precision for the control logic (sufficient for limiting decisions).
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
fn bjt_voltage_limit_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // V_BE limiting (SPICE pnjlim) — f32 hi-parts for control, DS for adjustment
    let vbe_old_f32 = polarity * (bvl_read_old_ds(base_idx).x - bvl_read_old_ds(emitter_idx).x);
    let vbe_new_f32 = polarity * (bvl_read_new_ds(base_idx).x - bvl_read_new_ds(emitter_idx).x);
    let vbe_limited = bvl_pnjlim(vbe_new_f32, vbe_old_f32, nf_vt, vcrit_f);

    if vbe_limited != vbe_new_f32 {
        let delta = vbe_new_f32 - vbe_old_f32;
        if abs(delta) > 1e-30 {
            let scale = (vbe_limited - vbe_old_f32) / delta;
            if base_idx != BVL_SENTINEL && base_fixed == 0u {
                let old_val = bvl_read_old_ds(base_idx);
                let new_val = bvl_read_new_ds(base_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                bvl_x_new_hi[base_idx] = result.x;
                bvl_x_new_lo[base_idx] = result.y;
            }
            if emitter_idx != BVL_SENTINEL && emitter_fixed == 0u {
                let old_val = bvl_read_old_ds(emitter_idx);
                let new_val = bvl_read_new_ds(emitter_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                bvl_x_new_hi[emitter_idx] = result.x;
                bvl_x_new_lo[emitter_idx] = result.y;
            }
        }
    }

    // V_BC limiting (SPICE pnjlim)
    let vbc_old_f32 = polarity * (bvl_read_old_ds(base_idx).x - bvl_read_old_ds(collector_idx).x);
    let vbc_new_f32 = polarity * (bvl_read_new_ds(base_idx).x - bvl_read_new_ds(collector_idx).x);
    let vbc_limited = bvl_pnjlim(vbc_new_f32, vbc_old_f32, nr_vt, vcrit_r);

    if vbc_limited != vbc_new_f32 {
        let delta = vbc_new_f32 - vbc_old_f32;
        if abs(delta) > 1e-30 {
            let scale = (vbc_limited - vbc_old_f32) / delta;
            if base_idx != BVL_SENTINEL && base_fixed == 0u {
                let old_val = bvl_read_old_ds(base_idx);
                let new_val = bvl_read_new_ds(base_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                bvl_x_new_hi[base_idx] = result.x;
                bvl_x_new_lo[base_idx] = result.y;
            }
            if collector_idx != BVL_SENTINEL && collector_fixed == 0u {
                let old_val = bvl_read_old_ds(collector_idx);
                let new_val = bvl_read_new_ds(collector_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                bvl_x_new_hi[collector_idx] = result.x;
                bvl_x_new_lo[collector_idx] = result.y;
            }
        }
    }
}
"#;

/// DS MOSFET evaluation shader body.
///
/// Entry point: `mosfet_eval_ds`
/// Output: 8 floats per MOSFET (4 DS pairs: I_D, g_m, g_ds, polarity)
pub const DS_MOSFET_EVAL_BODY: &str = r#"
// ============================================================
// MOSFET Evaluation Shader (Shichman-Hodges Level 1, DS precision)
// ============================================================

struct MosfetEvalParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_desc: array<u32>;
@group(0) @binding(1) var<storage, read> eval_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> eval_x_lo: array<f32>;
// Output: 8 floats per MOSFET (4 DS pairs)
@group(0) @binding(3) var<storage, read_write> eval_output: array<f32>;
@group(0) @binding(4) var<uniform> eval_params: MosfetEvalParams;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(eval_x_hi[idx], eval_x_lo[idx]);
}

@compute @workgroup_size(64)
fn mosfet_eval_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let v_g = read_voltage_ds(gate_idx);
    let v_s = read_voltage_ds(source_idx);
    let v_d = read_voltage_ds(drain_idx);

    // v_gs = polarity * (V_G - V_S)
    let vgs_raw = ds_sub(v_g.x, v_g.y, v_s.x, v_s.y);
    let v_gs = ds_mul(polarity, 0.0, vgs_raw.x, vgs_raw.y);
    // v_ds = polarity * (V_D - V_S)
    let vds_raw = ds_sub(v_d.x, v_d.y, v_s.x, v_s.y);
    let v_ds = ds_mul(polarity, 0.0, vds_raw.x, vds_raw.y);

    // Smooth subthreshold transition
    let VT: f32 = 0.02585;
    let v_ov = ds_sub(v_gs.x, v_gs.y, vto, 0.0);
    let arg = ds_div(v_ov.x, v_ov.y, VT, 0.0);

    var v_ov_eff: vec2<f32>;
    var sigmoid: vec2<f32>;
    if arg.x > 20.0 {
        v_ov_eff = v_ov;
        sigmoid = vec2(1.0, 0.0);
    } else if arg.x < -20.0 {
        let e_arg = ds_exp(arg.x, arg.y);
        v_ov_eff = ds_mul(VT, 0.0, e_arg.x, e_arg.y);
        sigmoid = e_arg;
    } else {
        // v_ov_eff = VT * log(1 + exp(arg))
        // sigmoid = 1 / (1 + exp(-arg))
        let e_arg = exp(arg.x);
        let log_val = log(1.0 + e_arg);
        v_ov_eff = ds_mul(VT, 0.0, log_val, 0.0);
        let sig_val = 1.0 / (1.0 + exp(-arg.x));
        sigmoid = vec2(sig_val, 0.0);
    }

    var i_d: vec2<f32> = vec2(0.0, 0.0);
    var g_m: vec2<f32> = vec2(0.0, 0.0);
    var g_ds: vec2<f32> = vec2(0.0, 0.0);

    if ds_lt(v_ds.x, v_ds.y, v_ov_eff.x, v_ov_eff.y) {
        // Linear region
        let lam = ds_add(1.0, 0.0, ds_mul(lambda, 0.0, v_ds.x, v_ds.y).x,
                         ds_mul(lambda, 0.0, v_ds.x, v_ds.y).y);
        // i_d = kp * (v_ov_eff * v_ds - v_ds^2 / 2) * lam
        let vov_vds = ds_mul(v_ov_eff.x, v_ov_eff.y, v_ds.x, v_ds.y);
        let vds2 = ds_mul(v_ds.x, v_ds.y, v_ds.x, v_ds.y);
        let vds2_h = ds_mul(0.5, 0.0, vds2.x, vds2.y);
        let inner = ds_sub(vov_vds.x, vov_vds.y, vds2_h.x, vds2_h.y);
        let kp_inner = ds_mul(kp, 0.0, inner.x, inner.y);
        i_d = ds_mul(kp_inner.x, kp_inner.y, lam.x, lam.y);
        // g_m = kp * v_ds * lam * sigmoid
        let kp_vds = ds_mul(kp, 0.0, v_ds.x, v_ds.y);
        let kp_vds_lam = ds_mul(kp_vds.x, kp_vds.y, lam.x, lam.y);
        g_m = ds_mul(kp_vds_lam.x, kp_vds_lam.y, sigmoid.x, sigmoid.y);
        // g_ds = kp*(v_ov_eff-v_ds)*lam + kp*(v_ov_eff*v_ds - v_ds^2/2)*lambda
        let vov_m_vds = ds_sub(v_ov_eff.x, v_ov_eff.y, v_ds.x, v_ds.y);
        let term1 = ds_mul(kp, 0.0, vov_m_vds.x, vov_m_vds.y);
        let term1_lam = ds_mul(term1.x, term1.y, lam.x, lam.y);
        let term2 = ds_mul(kp_inner.x, kp_inner.y, lambda, 0.0);
        g_ds = ds_add(term1_lam.x, term1_lam.y, term2.x, term2.y);
    } else {
        // Saturation region
        let lam = ds_add(1.0, 0.0, ds_mul(lambda, 0.0, v_ds.x, v_ds.y).x,
                         ds_mul(lambda, 0.0, v_ds.x, v_ds.y).y);
        // i_d = kp/2 * v_ov_eff^2 * lam
        let vov2 = ds_mul(v_ov_eff.x, v_ov_eff.y, v_ov_eff.x, v_ov_eff.y);
        let kp_h = kp / 2.0;
        let kp_h_vov2 = ds_mul(kp_h, 0.0, vov2.x, vov2.y);
        i_d = ds_mul(kp_h_vov2.x, kp_h_vov2.y, lam.x, lam.y);
        // g_m = kp * v_ov_eff * lam * sigmoid
        let kp_vov = ds_mul(kp, 0.0, v_ov_eff.x, v_ov_eff.y);
        let kp_vov_lam = ds_mul(kp_vov.x, kp_vov.y, lam.x, lam.y);
        g_m = ds_mul(kp_vov_lam.x, kp_vov_lam.y, sigmoid.x, sigmoid.y);
        // g_ds = kp/2 * v_ov_eff^2 * lambda
        g_ds = ds_mul(kp_h_vov2.x, kp_h_vov2.y, lambda, 0.0);
    }

    let out = 8u * i;
    let id_pol = ds_mul(polarity, 0.0, i_d.x, i_d.y);
    eval_output[out + 0u] = id_pol.x;
    eval_output[out + 1u] = id_pol.y;
    eval_output[out + 2u] = g_m.x;
    eval_output[out + 3u] = g_m.y;
    eval_output[out + 4u] = g_ds.x;
    eval_output[out + 5u] = g_ds.y;
    eval_output[out + 6u] = polarity;
    eval_output[out + 7u] = 0.0;
}
"#;

/// DS MOSFET assembly shader body.
///
/// Entry points: `assemble_mosfet_matrix_stamp_ds`, `assemble_mosfet_rhs_stamp_ds`
pub const DS_MOSFET_ASSEMBLE_BODY: &str = r#"
// ============================================================
// MOSFET Assembly Shader (DS precision)
// ============================================================

const SENTINEL: u32 = 0xFFFFFFFFu;

struct MosfetStampMatParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_mat_desc: array<u32>;
// Eval output: 8 floats per MOSFET (4 DS pairs)
@group(0) @binding(1) var<storage, read> mosfet_mat_eval: array<f32>;
@group(0) @binding(2) var<storage, read_write> mosfet_mat_values_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> mosfet_mat_values_lo: array<f32>;
@group(0) @binding(4) var<uniform> mosfet_mat_params: MosfetStampMatParams;

@compute @workgroup_size(64)
fn assemble_mosfet_matrix_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mosfet_mat_params.num_mosfets {
        return;
    }

    let desc_base = i * 16u;
    let eval_base = 8u * i;

    let g_m_hi = mosfet_mat_eval[eval_base + 2u];
    let g_m_lo = mosfet_mat_eval[eval_base + 3u];
    let g_ds_hi = mosfet_mat_eval[eval_base + 4u];
    let g_ds_lo = mosfet_mat_eval[eval_base + 5u];

    let gm_gds = ds_add(g_m_hi, g_m_lo, g_ds_hi, g_ds_lo);

    // Stamp order: [DD, DG, DS, SD, SG, SS]
    var stamps_hi: array<f32, 6>;
    var stamps_lo: array<f32, 6>;
    stamps_hi[0] = g_ds_hi; stamps_lo[0] = g_ds_lo;     // DD
    stamps_hi[1] = g_m_hi; stamps_lo[1] = g_m_lo;       // DG
    let neg_gm_gds = ds_neg(gm_gds.x, gm_gds.y);
    stamps_hi[2] = neg_gm_gds.x; stamps_lo[2] = neg_gm_gds.y; // DS
    let neg_gds = ds_neg(g_ds_hi, g_ds_lo);
    stamps_hi[3] = neg_gds.x; stamps_lo[3] = neg_gds.y; // SD
    let neg_gm = ds_neg(g_m_hi, g_m_lo);
    stamps_hi[4] = neg_gm.x; stamps_lo[4] = neg_gm.y;   // SG
    stamps_hi[5] = gm_gds.x; stamps_lo[5] = gm_gds.y;   // SS

    for (var j = 0u; j < 6u; j = j + 1u) {
        let pos = mosfet_mat_desc[desc_base + 8u + j];
        if pos != SENTINEL {
            let s = ds_add(mosfet_mat_values_hi[pos], mosfet_mat_values_lo[pos],
                          stamps_hi[j], stamps_lo[j]);
            mosfet_mat_values_hi[pos] = s.x;
            mosfet_mat_values_lo[pos] = s.y;
        }
    }
}

// --- RHS stamp ---

struct MosfetStampRhsParams {
    num_mosfets: u32,
}

@group(0) @binding(0) var<storage, read> mosfet_rhs_desc: array<u32>;
@group(0) @binding(1) var<storage, read> mosfet_rhs_x_hi: array<f32>;
@group(0) @binding(2) var<storage, read> mosfet_rhs_x_lo: array<f32>;
@group(0) @binding(3) var<storage, read> mosfet_rhs_eval: array<f32>;
@group(0) @binding(4) var<storage, read_write> mosfet_rhs_b_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> mosfet_rhs_b_lo: array<f32>;
@group(0) @binding(6) var<uniform> mosfet_rhs_params: MosfetStampRhsParams;

fn mosfet_read_voltage_ds(idx: u32) -> vec2<f32> {
    if idx == SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(mosfet_rhs_x_hi[idx], mosfet_rhs_x_lo[idx]);
}

@compute @workgroup_size(64)
fn assemble_mosfet_rhs_stamp_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mosfet_rhs_params.num_mosfets {
        return;
    }

    let desc_base = i * 16u;
    let eval_base = 8u * i;

    let drain_idx = mosfet_rhs_desc[desc_base + 0u];
    let gate_idx = mosfet_rhs_desc[desc_base + 1u];
    let source_idx = mosfet_rhs_desc[desc_base + 2u];

    let i_d_hi = mosfet_rhs_eval[eval_base + 0u];
    let i_d_lo = mosfet_rhs_eval[eval_base + 1u];
    let g_m_hi = mosfet_rhs_eval[eval_base + 2u];
    let g_m_lo = mosfet_rhs_eval[eval_base + 3u];
    let g_ds_hi = mosfet_rhs_eval[eval_base + 4u];
    let g_ds_lo = mosfet_rhs_eval[eval_base + 5u];

    let v_d = mosfet_read_voltage_ds(drain_idx);
    let v_g = mosfet_read_voltage_ds(gate_idx);
    let v_s = mosfet_read_voltage_ds(source_idx);

    let gm_gds = ds_add(g_m_hi, g_m_lo, g_ds_hi, g_ds_lo);
    let neg_gm_gds = ds_neg(gm_gds.x, gm_gds.y);

    // Drain: I_lin = g_ds * V_D + g_m * V_G - (g_m + g_ds) * V_S
    let t1 = ds_mul(g_ds_hi, g_ds_lo, v_d.x, v_d.y);
    let t2 = ds_mul(g_m_hi, g_m_lo, v_g.x, v_g.y);
    let t3 = ds_mul(neg_gm_gds.x, neg_gm_gds.y, v_s.x, v_s.y);
    let t12 = ds_add(t1.x, t1.y, t2.x, t2.y);
    let i_lin_d = ds_add(t12.x, t12.y, t3.x, t3.y);
    let i_eq_d = ds_sub(i_d_hi, i_d_lo, i_lin_d.x, i_lin_d.y);

    // Source: I_S = -I_D
    let neg_id = ds_neg(i_d_hi, i_d_lo);
    // I_lin_s = -g_ds*V_D - g_m*V_G + (g_m+g_ds)*V_S
    let neg_gds = ds_neg(g_ds_hi, g_ds_lo);
    let neg_gm = ds_neg(g_m_hi, g_m_lo);
    let s1 = ds_mul(neg_gds.x, neg_gds.y, v_d.x, v_d.y);
    let s2 = ds_mul(neg_gm.x, neg_gm.y, v_g.x, v_g.y);
    let s3 = ds_mul(gm_gds.x, gm_gds.y, v_s.x, v_s.y);
    let s12 = ds_add(s1.x, s1.y, s2.x, s2.y);
    let i_lin_s = ds_add(s12.x, s12.y, s3.x, s3.y);
    let i_eq_s = ds_sub(neg_id.x, neg_id.y, i_lin_s.x, i_lin_s.y);

    let b_d = mosfet_rhs_desc[desc_base + 14u];
    let b_s = mosfet_rhs_desc[desc_base + 15u];

    if b_d != SENTINEL {
        let s = ds_sub(mosfet_rhs_b_hi[b_d], mosfet_rhs_b_lo[b_d], i_eq_d.x, i_eq_d.y);
        mosfet_rhs_b_hi[b_d] = s.x;
        mosfet_rhs_b_lo[b_d] = s.y;
    }
    if b_s != SENTINEL {
        let s = ds_sub(mosfet_rhs_b_hi[b_s], mosfet_rhs_b_lo[b_s], i_eq_s.x, i_eq_s.y);
        mosfet_rhs_b_hi[b_s] = s.x;
        mosfet_rhs_b_lo[b_s] = s.y;
    }
}
"#;

/// DS MOSFET voltage limiting shader body.
///
/// Entry points: `mosfet_voltage_limit_reduce_ds`, `mosfet_voltage_limit_apply_ds`
/// Uses the same atomicMin trick as f32 version (scale is a ratio in [0,1]).
pub const DS_MOSFET_VOLTAGE_LIMIT_BODY: &str = r#"
// ============================================================
// MOSFET Voltage Limiting Shader (DS precision)
// ============================================================
// Uses same race-free atomicMin approach as f32 version.
// The scale factor is a simple ratio in [0,1] — single f32 encoded
// as u32 via integer scaling. No DS needed for the scale itself.

struct MosfetVoltLimitParams {
    num_mosfets: u32,
    n_nodes: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> mvl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> mvl_x_old_hi: array<f32>;
@group(0) @binding(2) var<storage, read> mvl_x_old_lo: array<f32>;
@group(0) @binding(3) var<storage, read_write> mvl_x_new_hi: array<f32>;
@group(0) @binding(4) var<storage, read_write> mvl_x_new_lo: array<f32>;
@group(0) @binding(5) var<storage, read_write> mvl_node_scale: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> mvl_params: MosfetVoltLimitParams;

const MVL_SENTINEL: u32 = 0xFFFFFFFFu;
const MVL_SCALE_ONE: u32 = 1000000u;
const MVL_STEP_LIMIT: f32 = 0.5;

fn mvl_read_old_ds(idx: u32) -> vec2<f32> {
    if idx == MVL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(mvl_x_old_hi[idx], mvl_x_old_lo[idx]);
}

fn mvl_read_new_ds(idx: u32) -> vec2<f32> {
    if idx == MVL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(mvl_x_new_hi[idx], mvl_x_new_lo[idx]);
}

fn mvl_scale_to_encoded(delta_v_hi: f32, delta_v_lo: f32, limit: f32) -> u32 {
    let abs_d = ds_abs(delta_v_hi, delta_v_lo);
    let scale = limit / abs_d.x;
    let encoded = u32(scale * f32(MVL_SCALE_ONE));
    return max(encoded, 1u);
}

@compute @workgroup_size(64)
fn mosfet_voltage_limit_reduce_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= mvl_params.num_mosfets {
        return;
    }

    let base = i * 16u;
    let drain_idx = mvl_desc[base + 0u];
    let gate_idx = mvl_desc[base + 1u];
    let source_idx = mvl_desc[base + 2u];

    let node_fixed_packed = mvl_desc[base + 7u];
    let drain_fixed = (node_fixed_packed & 1u) != 0u;
    let gate_fixed = (node_fixed_packed & 2u) != 0u;
    let source_fixed = (node_fixed_packed & 4u) != 0u;

    // V_GS limiting
    let vgs_old = ds_sub(mvl_read_old_ds(gate_idx).x, mvl_read_old_ds(gate_idx).y,
                         mvl_read_old_ds(source_idx).x, mvl_read_old_ds(source_idx).y);
    let vgs_new = ds_sub(mvl_read_new_ds(gate_idx).x, mvl_read_new_ds(gate_idx).y,
                         mvl_read_new_ds(source_idx).x, mvl_read_new_ds(source_idx).y);
    let delta_vgs = ds_sub(vgs_new.x, vgs_new.y, vgs_old.x, vgs_old.y);
    let abs_dvgs = ds_abs(delta_vgs.x, delta_vgs.y);

    if abs_dvgs.x > MVL_STEP_LIMIT {
        let scale_gs = mvl_scale_to_encoded(delta_vgs.x, delta_vgs.y, MVL_STEP_LIMIT);
        if gate_idx != MVL_SENTINEL && !gate_fixed {
            atomicMin(&mvl_node_scale[gate_idx], scale_gs);
        }
        if source_idx != MVL_SENTINEL && !source_fixed {
            atomicMin(&mvl_node_scale[source_idx], scale_gs);
        }
    }

    // V_DS limiting
    let vds_old = ds_sub(mvl_read_old_ds(drain_idx).x, mvl_read_old_ds(drain_idx).y,
                         mvl_read_old_ds(source_idx).x, mvl_read_old_ds(source_idx).y);
    let vds_new = ds_sub(mvl_read_new_ds(drain_idx).x, mvl_read_new_ds(drain_idx).y,
                         mvl_read_new_ds(source_idx).x, mvl_read_new_ds(source_idx).y);
    let delta_vds = ds_sub(vds_new.x, vds_new.y, vds_old.x, vds_old.y);
    let abs_dvds = ds_abs(delta_vds.x, delta_vds.y);

    if abs_dvds.x > MVL_STEP_LIMIT {
        let scale_ds = mvl_scale_to_encoded(delta_vds.x, delta_vds.y, MVL_STEP_LIMIT);
        if drain_idx != MVL_SENTINEL && !drain_fixed {
            atomicMin(&mvl_node_scale[drain_idx], scale_ds);
        }
        if source_idx != MVL_SENTINEL && !source_fixed {
            atomicMin(&mvl_node_scale[source_idx], scale_ds);
        }
    }
}

@compute @workgroup_size(64)
fn mosfet_voltage_limit_apply_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = gid.x;
    if node_idx >= mvl_params.n_nodes {
        return;
    }

    let keep_layout_word = mvl_desc[0u];
    if keep_layout_word == 0xFFFFFFFEu && node_idx == 0u {
        return;
    }

    let scale_encoded = atomicLoad(&mvl_node_scale[node_idx]);
    if scale_encoded >= MVL_SCALE_ONE {
        return;
    }

    let scale = f32(scale_encoded) / f32(MVL_SCALE_ONE);
    let old_hi = mvl_x_old_hi[node_idx];
    let old_lo = mvl_x_old_lo[node_idx];
    let delta = ds_sub(mvl_x_new_hi[node_idx], mvl_x_new_lo[node_idx], old_hi, old_lo);
    let scaled = ds_mul(delta.x, delta.y, scale, 0.0);
    let result = ds_add(old_hi, old_lo, scaled.x, scaled.y);
    mvl_x_new_hi[node_idx] = result.x;
    mvl_x_new_lo[node_idx] = result.y;
}
"#;

/// DS convergence checking shader body.
///
/// Entry points:
/// - `voltage_limit_ds`: Diode voltage limiting using SPICE pnjlim (DS)
/// - `convergence_check_ds`: Max-reduction of |x_new - x_old| (DS, output f32 partials)
pub const DS_CONVERGENCE_BODY: &str = r#"
// ============================================================
// Convergence & Voltage Limiting Shader (DS precision, SPICE pnjlim)
// ============================================================

// --- Voltage limiting: SPICE pnjlim for diode nodes (DS) ---

struct VoltLimitParams {
    num_diodes: u32,
}

@group(0) @binding(0) var<storage, read> vl_desc: array<u32>;
@group(0) @binding(1) var<storage, read> vl_x_old_hi: array<f32>;
@group(0) @binding(2) var<storage, read> vl_x_old_lo: array<f32>;
@group(0) @binding(3) var<storage, read_write> vl_x_new_hi: array<f32>;
@group(0) @binding(4) var<storage, read_write> vl_x_new_lo: array<f32>;
@group(0) @binding(5) var<uniform> vl_params: VoltLimitParams;

const VL_SENTINEL: u32 = 0xFFFFFFFFu;

fn vl_read_old_ds(idx: u32) -> vec2<f32> {
    if idx == VL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(vl_x_old_hi[idx], vl_x_old_lo[idx]);
}

fn vl_read_new_ds(idx: u32) -> vec2<f32> {
    if idx == VL_SENTINEL {
        return vec2(0.0, 0.0);
    }
    return vec2(vl_x_new_hi[idx], vl_x_new_lo[idx]);
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
fn voltage_limit_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Use f32 hi-parts for pnjlim control logic, DS for node adjustment
    let v_old_f32 = vl_read_old_ds(anode_idx).x - vl_read_old_ds(cathode_idx).x;
    let v_new_f32 = vl_read_new_ds(anode_idx).x - vl_read_new_ds(cathode_idx).x;
    let v_limited = vl_pnjlim(v_new_f32, v_old_f32, n_vt, vcrit);

    if v_limited != v_new_f32 {
        let delta = v_new_f32 - v_old_f32;
        if abs(delta) > 1e-30 {
            let scale = (v_limited - v_old_f32) / delta;
            if anode_idx != VL_SENTINEL {
                let old_val = vl_read_old_ds(anode_idx);
                let new_val = vl_read_new_ds(anode_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                vl_x_new_hi[anode_idx] = result.x;
                vl_x_new_lo[anode_idx] = result.y;
            }
            if cathode_idx != VL_SENTINEL {
                let old_val = vl_read_old_ds(cathode_idx);
                let new_val = vl_read_new_ds(cathode_idx);
                let d = ds_sub(new_val.x, new_val.y, old_val.x, old_val.y);
                let scaled = ds_mul(d.x, d.y, scale, 0.0);
                let result = ds_add(old_val.x, old_val.y, scaled.x, scaled.y);
                vl_x_new_hi[cathode_idx] = result.x;
                vl_x_new_lo[cathode_idx] = result.y;
            }
        }
    }
}

// --- Convergence check: parallel max-reduction of |x_new - x_old| (DS) ---
// Output: f32 partial maxes per workgroup (hi component is sufficient for tolerance)

const CONV_WG_SIZE: u32 = 64u;

struct ConvParams {
    n: u32,
    tolerance: f32,
}

var<workgroup> conv_scratch: array<f32, 64>;

@group(0) @binding(0) var<storage, read> conv_x_new_hi: array<f32>;
@group(0) @binding(1) var<storage, read> conv_x_new_lo: array<f32>;
@group(0) @binding(2) var<storage, read> conv_x_old_hi: array<f32>;
@group(0) @binding(3) var<storage, read> conv_x_old_lo: array<f32>;
@group(0) @binding(4) var<storage, read_write> conv_partial_max: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_flags: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> conv_params: ConvParams;

@compute @workgroup_size(64)
fn convergence_check_ds(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    var diff: f32 = 0.0;
    if i < conv_params.n {
        let xn_hi = conv_x_new_hi[i];
        let xn_lo = conv_x_new_lo[i];
        let xo_hi = conv_x_old_hi[i];
        let xo_lo = conv_x_old_lo[i];
        let d = ds_sub(xn_hi, xn_lo, xo_hi, xo_lo);
        let a = ds_abs(d.x, d.y);
        diff = a.x + a.y;

        // NaN/Inf detection
        if xn_hi != xn_hi || xn_hi - xn_hi != 0.0 {
            atomicOr(&conv_flags[1u], 1u);
            diff = 1.0e30;
        }
    }

    conv_scratch[local_id] = diff;
    workgroupBarrier();

    var stride = CONV_WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            conv_scratch[local_id] = max(conv_scratch[local_id], conv_scratch[local_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if local_id == 0u {
        conv_partial_max[wid.x] = conv_scratch[0];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ds_shaders;

    fn assert_shader_parses(body: &str, expected_entries: &[&str]) {
        let source = ds_shaders::ds_shader(body);
        let result = naga::front::wgsl::parse_str(&source);
        match result {
            Ok(module) => {
                let entry_names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|ep| ep.name.as_str())
                    .collect();
                for name in expected_entries {
                    assert!(
                        entry_names.contains(name),
                        "missing entry point: {name}. Found: {entry_names:?}"
                    );
                }
            }
            Err(e) => {
                panic!("WGSL parse error:\n{}", e.emit_to_string(&source));
            }
        }
    }

    #[test]
    fn ds_diode_eval_parses() {
        assert_shader_parses(DS_DIODE_EVAL_BODY, &["diode_eval_ds"]);
    }

    #[test]
    fn ds_nonlinear_assemble_parses() {
        assert_shader_parses(
            DS_NONLINEAR_ASSEMBLE_BODY,
            &[
                "assemble_matrix_copy_ds",
                "assemble_matrix_stamp_ds",
                "assemble_rhs_copy_ds",
                "assemble_rhs_stamp_ds",
            ],
        );
    }

    #[test]
    fn ds_bjt_eval_parses() {
        assert_shader_parses(DS_BJT_EVAL_BODY, &["bjt_eval_ds"]);
    }

    #[test]
    fn ds_bjt_assemble_parses() {
        assert_shader_parses(
            DS_BJT_ASSEMBLE_BODY,
            &["assemble_bjt_matrix_stamp_ds", "assemble_bjt_rhs_stamp_ds"],
        );
    }

    #[test]
    fn ds_bjt_voltage_limit_parses() {
        assert_shader_parses(DS_BJT_VOLTAGE_LIMIT_BODY, &["bjt_voltage_limit_ds"]);
    }

    #[test]
    fn ds_mosfet_eval_parses() {
        assert_shader_parses(DS_MOSFET_EVAL_BODY, &["mosfet_eval_ds"]);
    }

    #[test]
    fn ds_mosfet_assemble_parses() {
        assert_shader_parses(
            DS_MOSFET_ASSEMBLE_BODY,
            &[
                "assemble_mosfet_matrix_stamp_ds",
                "assemble_mosfet_rhs_stamp_ds",
            ],
        );
    }

    #[test]
    fn ds_mosfet_voltage_limit_parses() {
        assert_shader_parses(
            DS_MOSFET_VOLTAGE_LIMIT_BODY,
            &[
                "mosfet_voltage_limit_reduce_ds",
                "mosfet_voltage_limit_apply_ds",
            ],
        );
    }

    #[test]
    fn ds_convergence_parses() {
        assert_shader_parses(
            DS_CONVERGENCE_BODY,
            &["voltage_limit_ds", "convergence_check_ds"],
        );
    }
}
