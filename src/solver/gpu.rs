//! GPU-accelerated iterative solver using wgpu.
//!
//! Implements BiCGSTAB on the GPU for both real and complex linear systems.
//! Uses f32 precision on the GPU with f64 interface for compatibility.
//!
//! The real path delegates to `WgpuBackend` (via the `SolverBackend` trait).
//! The complex path uses its own GPU dispatch closures directly.

use crate::error::{OhmnivoreError, Result};
use crate::sparse::CsrMatrix;
use num_complex::Complex64;
use wgpu::util::DeviceExt;

use super::backend::{GpuCsrMatrix, SolverBackend, WgpuBackend, WgpuBuffer};
use super::bicgstab;
use super::preconditioner;

const MAX_ITERATIONS: u32 = 10000;
const TOLERANCE: f32 = 1e-5;
const WORKGROUP_SIZE: u32 = 64;

fn workgroup_count(n: u32) -> u32 {
    n.div_ceil(WORKGROUP_SIZE)
}

// Shader VecParams layout (must match WGSL struct)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VecParams {
    alpha_re: f32,
    alpha_im: f32,
    n: u32,
    _pad: u32,
}

/// GPU-accelerated linear solver using BiCGSTAB.
pub struct GpuSolver {
    backend: WgpuBackend,
}

impl GpuSolver {
    pub fn new() -> Result<Self> {
        Ok(Self {
            backend: WgpuBackend::new()?,
        })
    }
}

/// Read a GPU buffer back to CPU as f32 values.
fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender.send(r).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Upload a CsrMatrix<f64> to the GPU as f32.
fn upload_csr_f32(backend: &WgpuBackend, m: &CsrMatrix<f64>, n: usize) -> GpuCsrMatrix {
    let values: Vec<f32> = m.values.iter().map(|&v| v as f32).collect();
    let col_indices: Vec<u32> = m.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs: Vec<u32> = m.row_pointers.iter().map(|&r| r as u32).collect();
    backend.upload_matrix(&values, &col_indices, &row_ptrs, n)
}

impl super::LinearSolver for GpuSolver {
    fn solve_real(&self, a: &CsrMatrix<f64>, b: &[f64]) -> Result<Vec<f64>> {
        let n = a.nrows;
        if a.ncols != n || b.len() != n {
            return Err(OhmnivoreError::Solve(format!(
                "dimension mismatch: matrix is {}x{}, rhs length is {}",
                a.nrows, a.ncols,
                b.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let backend = &self.backend;

        // Convert matrix to f32
        let values_f32: Vec<f32> = a.values.iter().map(|&v| v as f32).collect();
        let col_indices_u32: Vec<u32> = a.col_indices.iter().map(|&c| c as u32).collect();
        let row_ptrs_u32: Vec<u32> = a.row_pointers.iter().map(|&r| r as u32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();

        // Upload matrix and RHS
        let gpu_matrix = backend.upload_matrix(&values_f32, &col_indices_u32, &row_ptrs_u32, n);
        let x_buf = backend.new_buffer(n);
        let b_buf = backend.new_buffer(n);
        backend.upload_vec(&b_f32, &b_buf);

        // Check if matrix has zero diagonals (MNA matrices with voltage sources/inductors)
        let has_zero_diag = (0..n).any(|row| {
            let mut diag = 0.0f64;
            for idx in a.row_pointers[row]..a.row_pointers[row + 1] {
                if a.col_indices[idx] == row {
                    diag = a.values[idx];
                    break;
                }
            }
            diag.abs() < 1e-30
        });

        if has_zero_diag {
            // ISAI preconditioner for matrices with zero diagonals
            let isai = preconditioner::compute_isai(a, 0);
            let ml_gpu = upload_csr_f32(backend, &isai.m_l, n);
            let mu_gpu = upload_csr_f32(backend, &isai.m_u, n);
            let tmp = backend.new_buffer(n);

            bicgstab::bicgstab(
                backend,
                &gpu_matrix,
                &b_buf,
                &x_buf,
                |b: &WgpuBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                    b.spmv(&ml_gpu, inp, &tmp);
                    b.spmv(&mu_gpu, &tmp, out);
                },
                n,
            )?;
        } else {
            // Jacobi preconditioner: fast path for matrices with nonzero diagonals
            let mut inv_diag = vec![0.0f32; n];
            for (row, diag) in inv_diag.iter_mut().enumerate() {
                for idx in a.row_pointers[row]..a.row_pointers[row + 1] {
                    if a.col_indices[idx] == row {
                        let val = a.values[idx] as f32;
                        if val.abs() > 1e-30 {
                            *diag = 1.0 / val;
                        }
                        break;
                    }
                }
            }
            let inv_diag_buf = backend.upload_storage_buffer(&inv_diag);

            bicgstab::bicgstab(
                backend,
                &gpu_matrix,
                &b_buf,
                &x_buf,
                |b: &WgpuBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                    b.jacobi_apply(&inv_diag_buf, inp, out);
                },
                n,
            )?;
        }

        // Read back solution
        let mut result_f32 = vec![0.0f32; n];
        backend.download_vec(&x_buf, &mut result_f32);
        Ok(result_f32.iter().map(|&v| v as f64).collect())
    }

    fn solve_complex(
        &self,
        a: &CsrMatrix<Complex64>,
        b: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        let n = a.nrows;
        if a.ncols != n || b.len() != n {
            return Err(OhmnivoreError::Solve(format!(
                "dimension mismatch: matrix is {}x{}, rhs length is {}",
                a.nrows, a.ncols,
                b.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let device = &self.backend.ctx.device;
        let queue = &self.backend.ctx.queue;
        let n_u32 = n as u32;
        let n_wg = workgroup_count(n_u32);

        // Convert to GPU format: vec2<f32> = [re, im]
        let values_f32: Vec<[f32; 2]> =
            a.values.iter().map(|v| [v.re as f32, v.im as f32]).collect();
        let col_indices_u32: Vec<u32> = a.col_indices.iter().map(|&c| c as u32).collect();
        let row_ptrs_u32: Vec<u32> = a.row_pointers.iter().map(|&r| r as u32).collect();
        let b_f32: Vec<[f32; 2]> = b.iter().map(|v| [v.re as f32, v.im as f32]).collect();

        let values_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_values"),
            contents: bytemuck::cast_slice(&values_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let col_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_col_indices"),
            contents: bytemuck::cast_slice(&col_indices_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row_ptrs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_row_ptrs"),
            contents: bytemuck::cast_slice(&row_ptrs_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let spmv_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_spmv_params"),
            contents: bytemuck::bytes_of(&n_u32),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let zeros_c: Vec<[f32; 2]> = vec![[0.0f32; 2]; n];

        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_x"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let r_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_r"),
            contents: bytemuck::cast_slice(&b_f32),
            usage: storage_rw,
        });
        let r_hat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_r_hat"),
            contents: bytemuck::cast_slice(&b_f32),
            usage: storage_rw,
        });
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_p"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_v"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let s_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_s"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let t_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_t"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let p_hat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_p_hat"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });
        let s_hat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_s_hat"),
            contents: bytemuck::cast_slice(&zeros_c),
            usage: storage_rw,
        });

        // Jacobi preconditioner: complex inverse diagonal
        let mut inv_diag = vec![[0.0f32; 2]; n];
        for (row, diag) in inv_diag.iter_mut().enumerate() {
            for idx in a.row_pointers[row]..a.row_pointers[row + 1] {
                if a.col_indices[idx] == row {
                    let re = a.values[idx].re as f32;
                    let im = a.values[idx].im as f32;
                    let norm_sq = re * re + im * im;
                    if norm_sq > 1e-30 {
                        *diag = [re / norm_sq, -im / norm_sq];
                    }
                    break;
                }
            }
        }
        let inv_diag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c_inv_diag"),
            contents: bytemuck::cast_slice(&inv_diag),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Dot product partial output: one vec2<f32> per workgroup
        let dot_out_size = (n_wg as usize * 2 * std::mem::size_of::<f32>()) as u64;
        let dot_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c_dot_out"),
            size: dot_out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Dispatch helpers ──

        let dispatch_spmv = |in_vec: &wgpu::Buffer, out_vec: &wgpu::Buffer| {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .backend
                    .ctx
                    .spmv_complex_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: col_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: row_ptrs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: in_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: spmv_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.backend.ctx.spmv_complex_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_wg, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        };

        // Complex dot product (conjugate): returns (re, im)
        let dispatch_dot = |a_vec: &wgpu::Buffer, b_vec: &wgpu::Buffer| -> (f32, f32) {
            let params = VecParams {
                alpha_re: 0.0,
                alpha_im: 0.0,
                n: n_u32,
                _pad: 0,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .backend
                    .ctx
                    .dot_complex_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dot_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.backend.ctx.dot_complex_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_wg, 1, 1);
            }
            queue.submit(Some(encoder.finish()));

            // Read partial sums (vec2 per workgroup) and reduce
            let partials = read_buffer_f32(device, queue, &dot_out_buf, n_wg as usize * 2);
            let mut re_sum = 0.0f32;
            let mut im_sum = 0.0f32;
            for i in 0..n_wg as usize {
                re_sum += partials[i * 2];
                im_sum += partials[i * 2 + 1];
            }
            (re_sum, im_sum)
        };

        // Complex axpy: y = alpha * x + y (in-place)
        let dispatch_axpy =
            |x_vec: &wgpu::Buffer, y_vec: &wgpu::Buffer, alpha_re: f32, alpha_im: f32| {
                let params = VecParams {
                    alpha_re,
                    alpha_im,
                    n: n_u32,
                    _pad: 0,
                };
                let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self
                        .backend
                        .ctx
                        .axpy_complex_pipeline
                        .get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_vec.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: y_vec.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.backend.ctx.axpy_complex_pipeline);
                    pass.set_bind_group(0, Some(&bg), &[]);
                    pass.dispatch_workgroups(n_wg, 1, 1);
                }
                queue.submit(Some(encoder.finish()));
            };

        // Complex scale: x = alpha * x
        let dispatch_scale = |x_vec: &wgpu::Buffer, alpha_re: f32, alpha_im: f32| {
            let params = VecParams {
                alpha_re,
                alpha_im,
                n: n_u32,
                _pad: 0,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .backend
                    .ctx
                    .scale_complex_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.backend.ctx.scale_complex_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_wg, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        };

        // Complex copy: dst = src
        let dispatch_copy = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            let params = VecParams {
                alpha_re: 0.0,
                alpha_im: 0.0,
                n: n_u32,
                _pad: 0,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .backend
                    .ctx
                    .copy_complex_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.backend.ctx.copy_complex_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_wg, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        };

        // Complex Jacobi: out[i] = inv_diag[i] * in_vec[i]
        let dispatch_jacobi = |in_vec: &wgpu::Buffer, out_vec: &wgpu::Buffer| {
            let params = VecParams {
                alpha_re: 0.0,
                alpha_im: 0.0,
                n: n_u32,
                _pad: 0,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .backend
                    .ctx
                    .jacobi_complex_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inv_diag_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_vec.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.backend.ctx.jacobi_complex_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_wg, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        };

        // Complex helpers
        let cdiv = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
            let d = b.0 * b.0 + b.1 * b.1;
            if d < 1e-30 {
                (0.0, 0.0)
            } else {
                (
                    (a.0 * b.0 + a.1 * b.1) / d,
                    (a.1 * b.0 - a.0 * b.1) / d,
                )
            }
        };
        let cmul = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
            (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
        };
        let cnorm_sq = |a: (f32, f32)| -> f32 { a.0 * a.0 + a.1 * a.1 };

        // Compute ||b||
        let (b_nsq_re, _) = dispatch_dot(&r_buf, &r_buf);
        let b_norm = b_nsq_re.sqrt();
        if b_norm < 1e-30 {
            return Ok(vec![Complex64::new(0.0, 0.0); n]);
        }
        let abs_tol = TOLERANCE * b_norm;

        // ── BiCGSTAB (complex) ──
        let mut rho = (1.0f32, 0.0f32);
        let mut alpha = (1.0f32, 0.0f32);
        let mut omega = (1.0f32, 0.0f32);

        for _iter in 0..MAX_ITERATIONS {
            let rho_new = dispatch_dot(&r_hat_buf, &r_buf);
            if cnorm_sq(rho_new) < 1e-30 {
                return Err(OhmnivoreError::Solve(
                    "BiCGSTAB breakdown: rho ~ 0".into(),
                ));
            }

            let beta = cmul(cdiv(rho_new, rho), cdiv(alpha, omega));
            rho = rho_new;

            // p = r + beta * (p - omega * v)
            dispatch_axpy(&v_buf, &p_buf, -omega.0, -omega.1);
            dispatch_scale(&p_buf, beta.0, beta.1);
            dispatch_axpy(&r_buf, &p_buf, 1.0, 0.0);

            // Right-preconditioned: p_hat = M^{-1}*p, v = A*p_hat
            dispatch_jacobi(&p_buf, &p_hat_buf);
            dispatch_spmv(&p_hat_buf, &v_buf);

            // alpha = rho / (r_hat . v)
            let rhv = dispatch_dot(&r_hat_buf, &v_buf);
            if cnorm_sq(rhv) < 1e-30 {
                return Err(OhmnivoreError::Solve(
                    "BiCGSTAB breakdown: r_hat.v ~ 0".into(),
                ));
            }
            alpha = cdiv(rho, rhv);

            // s = r - alpha * v
            dispatch_copy(&r_buf, &s_buf);
            dispatch_axpy(&v_buf, &s_buf, -alpha.0, -alpha.1);

            // Check ||s||
            let (s_nsq, _) = dispatch_dot(&s_buf, &s_buf);
            if s_nsq.sqrt() < abs_tol {
                dispatch_axpy(&p_hat_buf, &x_buf, alpha.0, alpha.1);
                break;
            }

            // Right-preconditioned: s_hat = M^{-1}*s, t = A*s_hat
            dispatch_jacobi(&s_buf, &s_hat_buf);
            dispatch_spmv(&s_hat_buf, &t_buf);

            // omega = (t.s) / (t.t)
            let tds = dispatch_dot(&t_buf, &s_buf);
            let tdt = dispatch_dot(&t_buf, &t_buf);
            if cnorm_sq(tdt) < 1e-30 {
                return Err(OhmnivoreError::Solve(
                    "BiCGSTAB breakdown: ||t|| ~ 0".into(),
                ));
            }
            omega = cdiv(tds, tdt);

            // x = x + alpha*p_hat + omega*s_hat
            dispatch_axpy(&p_hat_buf, &x_buf, alpha.0, alpha.1);
            dispatch_axpy(&s_hat_buf, &x_buf, omega.0, omega.1);

            // r = s - omega*t
            dispatch_copy(&s_buf, &r_buf);
            dispatch_axpy(&t_buf, &r_buf, -omega.0, -omega.1);

            // Check convergence
            let (r_nsq, _) = dispatch_dot(&r_buf, &r_buf);
            if r_nsq.sqrt() < abs_tol {
                break;
            }

            if cnorm_sq(omega) < 1e-30 {
                return Err(OhmnivoreError::Solve(
                    "BiCGSTAB breakdown: omega ~ 0".into(),
                ));
            }
        }

        // Read back
        let result_f32 = read_buffer_f32(device, queue, &x_buf, n * 2);
        let result: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(result_f32[i * 2] as f64, result_f32[i * 2 + 1] as f64))
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::LinearSolver;
    use crate::sparse::CsrMatrix;

    fn try_gpu_solver() -> Option<GpuSolver> {
        GpuSolver::new().ok()
    }

    #[test]
    fn gpu_device_creation() {
        if try_gpu_solver().is_none() {
            eprintln!("skipping GPU test: no GPU available");
            return;
        }
    }

    #[test]
    fn gpu_solve_real_identity_2x2() {
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![3.0, 7.0];
        let x = solver.solve_real(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 0.01, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0).abs() < 0.01, "x[1] = {}", x[1]);
    }

    #[test]
    fn gpu_solve_real_known_2x2() {
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 5.0), (1, 1, 7.0)],
        );
        let b = vec![11.0, 13.0];
        let x = solver.solve_real(&a, &b).unwrap();
        let expected_x0 = 64.0 / 9.0;
        let expected_x1 = -29.0 / 9.0;
        assert!(
            (x[0] - expected_x0).abs() < 0.05,
            "x[0]={}, expected {}",
            x[0],
            expected_x0
        );
        assert!(
            (x[1] - expected_x1).abs() < 0.05,
            "x[1]={}, expected {}",
            x[1],
            expected_x1
        );
    }

    #[test]
    fn gpu_solve_real_3x3_spd() {
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        );
        let b = vec![1.0, 5.0, 10.0];
        let x = solver.solve_real(&a, &b).unwrap();
        let ax = a.spmv(&x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 0.1,
                "ax[{}]={}, b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn gpu_solve_complex_identity() {
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let one = Complex64::new(1.0, 0.0);
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, one), (1, 1, one)]);
        let b = vec![Complex64::new(3.0, 1.0), Complex64::new(7.0, -2.0)];
        let x = solver.solve_complex(&a, &b).unwrap();
        assert!((x[0].re - 3.0).abs() < 0.01, "x[0].re={}", x[0].re);
        assert!((x[0].im - 1.0).abs() < 0.01, "x[0].im={}", x[0].im);
        assert!((x[1].re - 7.0).abs() < 0.01, "x[1].re={}", x[1].re);
        assert!((x[1].im + 2.0).abs() < 0.01, "x[1].im={}", x[1].im);
    }

    #[test]
    fn gpu_solve_complex_known_2x2() {
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 0, Complex64::new(1.0, 1.0)),
                (0, 1, Complex64::new(2.0, 0.0)),
                (1, 1, Complex64::new(1.0, -1.0)),
            ],
        );
        let b = vec![Complex64::new(5.0, 1.0), Complex64::new(2.0, -2.0)];
        let x = solver.solve_complex(&a, &b).unwrap();
        assert!((x[0].re - 1.0).abs() < 0.05, "x[0].re={}", x[0].re);
        assert!(x[0].im.abs() < 0.05, "x[0].im={}", x[0].im);
        assert!((x[1].re - 2.0).abs() < 0.05, "x[1].re={}", x[1].re);
        assert!(x[1].im.abs() < 0.05, "x[1].im={}", x[1].im);
    }

    // ── ISAI preconditioner integration tests ──

    #[test]
    fn gpu_isai_voltage_divider() {
        // Voltage divider: V1=10V, R1=1kΩ (node 1→2), R2=1kΩ (node 2→0)
        // MNA matrix has a zero-diagonal row from the voltage source.
        //
        // Nodes: 1, 2; Branch current: i_V1
        // MNA: [G1,   -G1,  1] [V1]   [0   ]
        //      [-G1, G1+G2, 0] [V2] = [0   ]
        //      [1,    0,    0] [i ]   [10.0 ]
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let g1 = 1.0 / 1000.0; // 1kΩ
        let g2 = 1.0 / 1000.0; // 1kΩ
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, g1),
                (0, 1, -g1),
                (0, 2, 1.0),
                (1, 0, -g1),
                (1, 1, g1 + g2),
                (2, 0, 1.0),
                // (2,2) intentionally absent — zero diagonal
            ],
        );
        let b = vec![0.0, 0.0, 10.0];
        let x = solver.solve_real(&a, &b).unwrap();
        // V(1) = 10.0, V(2) = 5.0
        assert!(
            (x[0] - 10.0).abs() < 0.1,
            "V(1)={}, expected 10.0",
            x[0]
        );
        assert!(
            (x[1] - 5.0).abs() < 0.1,
            "V(2)={}, expected 5.0",
            x[1]
        );
    }

    #[test]
    fn gpu_isai_two_voltage_sources() {
        // Two voltage sources: V1 sets node 1 = 10V, V2 sets node 2 = 5V
        // R1 = 1kΩ between nodes 1 and 2
        // MNA: [G1,  -G1, 1, 0] [V1  ]   [0 ]
        //      [-G1,  G1, 0, 1] [V2  ] = [0 ]
        //      [1,    0,  0, 0] [iV1 ]   [10]
        //      [0,    1,  0, 0] [iV2 ]   [5 ]
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let g1 = 1.0 / 1000.0;
        let a = CsrMatrix::from_triplets(
            4,
            4,
            &[
                (0, 0, g1),
                (0, 1, -g1),
                (0, 2, 1.0),
                (1, 0, -g1),
                (1, 1, g1),
                (1, 3, 1.0),
                (2, 0, 1.0),
                (3, 1, 1.0),
            ],
        );
        let b = vec![0.0, 0.0, 10.0, 5.0];
        let x = solver.solve_real(&a, &b).unwrap();
        // V(1) = 10.0, V(2) = 5.0
        assert!(
            (x[0] - 10.0).abs() < 0.1,
            "V(1)={}, expected 10.0",
            x[0]
        );
        assert!(
            (x[1] - 5.0).abs() < 0.1,
            "V(2)={}, expected 5.0",
            x[1]
        );
        // i_V1 = (V1 - V2) / R = 5mA
        assert!(
            (x[2] - (-0.005)).abs() < 0.001,
            "iV1={}, expected -0.005",
            x[2]
        );
    }

    #[test]
    fn gpu_isai_vs_cpu_voltage_divider() {
        // Verify GPU with ISAI agrees with CPU solver
        let Some(gpu_solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        let cpu_solver = crate::solver::cpu::CpuSolver;

        let g1 = 1.0 / 1000.0;
        let g2 = 1.0 / 1000.0;
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, g1),
                (0, 1, -g1),
                (0, 2, 1.0),
                (1, 0, -g1),
                (1, 1, g1 + g2),
                (2, 0, 1.0),
            ],
        );
        let b = vec![0.0, 0.0, 10.0];
        let x_cpu = cpu_solver.solve_real(&a, &b).unwrap();
        let x_gpu = gpu_solver.solve_real(&a, &b).unwrap();

        for i in 0..3 {
            assert!(
                (x_cpu[i] - x_gpu[i]).abs() < 0.1,
                "CPU vs GPU mismatch at [{}]: cpu={}, gpu={}",
                i, x_cpu[i], x_gpu[i]
            );
        }
    }

    #[test]
    fn gpu_jacobi_resistive_regression() {
        // Pure resistive circuit (no zero diagonals) should still work with Jacobi
        let Some(solver) = try_gpu_solver() else {
            eprintln!("skipping GPU test: no GPU available");
            return;
        };
        // 2-node resistor divider (all conductances, no voltage sources)
        // G = [G1+G2, -G2;  -G2, G2+G3]
        // Current source: 10mA into node 1
        let g1 = 0.001; // 1kΩ to ground
        let g2 = 0.002; // 500Ω between nodes
        let g3 = 0.001; // 1kΩ to ground
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 0, g1 + g2),
                (0, 1, -g2),
                (1, 0, -g2),
                (1, 1, g2 + g3),
            ],
        );
        let b = vec![0.01, 0.0]; // 10mA into node 1
        let x = solver.solve_real(&a, &b).unwrap();
        let ax = a.spmv(&x);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-3,
                "residual at [{}]: ax={}, b={}",
                i, ax[i], b[i]
            );
        }
    }
}
