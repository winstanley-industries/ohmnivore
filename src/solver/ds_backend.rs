//! Double-single precision wgpu backend for BiCGSTAB.
//!
//! Implements `SolverBackend` using DS arithmetic (~48-bit mantissa from
//! paired f32 values). This is a wgpu-specific workaround for the lack of
//! native f64 support; CUDA/ROCm backends use f64 directly.

use std::cell::Cell;

use wgpu::util::DeviceExt;

use crate::error::{OhmnivoreError, Result};

use super::backend::{GpuCsrMatrix, SolverBackend, WgpuBuffer};
use super::ds_nonlinear_shaders;
use super::ds_shaders;
use super::nonlinear::{ConvergenceResult, NonlinearBackend};
use crate::compiler::{GpuBjtDescriptor, GpuDiodeDescriptor, GpuMosfetDescriptor};

const WORKGROUP_SIZE: u32 = 64;

fn workgroup_count(n: u32) -> u32 {
    n.div_ceil(WORKGROUP_SIZE)
}

/// Uniform parameters for DS vector operations.
/// Layout matches the WGSL VecParams struct in ds_shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DsVecParams {
    alpha_hi: f32,
    alpha_lo: f32,
    n: u32,
    _pad: u32,
}

/// Split an f64 into a double-single (hi, lo) pair of f32 values.
pub fn f64_to_ds(v: f64) -> (f32, f32) {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    (hi, lo)
}

/// Recombine a double-single (hi, lo) pair back to f64.
pub fn ds_to_f64(hi: f32, lo: f32) -> f64 {
    hi as f64 + lo as f64
}

/// GPU pipelines for DS compute shaders.
struct DsPipelines {
    spmv: wgpu::ComputePipeline,
    dot: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
}

/// Double-single precision wgpu backend.
///
/// Each vector element and matrix value is stored as a (hi, lo) pair of f32
/// values, giving ~48 bits of mantissa. All `SolverBackend` operations dispatch
/// DS-arithmetic compute shaders.
pub struct WgpuDsBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: DsPipelines,
    dispatch_count: Cell<u32>,
    readback_count: Cell<u32>,
}

impl WgpuDsBackend {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| OhmnivoreError::Solve("no GPU adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ohmnivore_ds_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| OhmnivoreError::Solve(format!("failed to get GPU device: {e}")))?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ohmnivore_ds_shaders"),
            source: wgpu::ShaderSource::Wgsl(ds_shaders::DS_SHADER_SOURCE.as_str().into()),
        });

        let make_pipeline = |entry_point: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipes = DsPipelines {
            spmv: make_pipeline("spmv_ds"),
            dot: make_pipeline("dot_ds"),
            axpy: make_pipeline("axpy_ds"),
            scale: make_pipeline("scale_ds"),
            copy: make_pipeline("copy_ds"),
            jacobi: make_pipeline("jacobi_ds"),
        };

        Ok(Self {
            device,
            queue,
            pipes,
            dispatch_count: Cell::new(0),
            readback_count: Cell::new(0),
        })
    }

    /// Get total GPU dispatch count since creation.
    pub fn dispatch_count(&self) -> u32 {
        self.dispatch_count.get()
    }

    /// Get total GPU readback count since creation.
    pub fn readback_count(&self) -> u32 {
        self.readback_count.get()
    }

    /// Upload an f64 vector to a DS buffer with full precision splitting.
    pub fn upload_vec_f64(&self, data: &[f64], buffer: &WgpuBuffer) {
        let hi: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let lo: Vec<f32> = data
            .iter()
            .zip(hi.iter())
            .map(|(&v, &h)| (v - h as f64) as f32)
            .collect();
        self.queue
            .write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(&hi));
        if let Some(ref lo_buf) = buffer.buffer_lo {
            self.queue
                .write_buffer(lo_buf, 0, bytemuck::cast_slice(&lo));
        }
    }

    /// Download a DS buffer back to CPU as f64 values.
    pub fn download_vec_f64(&self, buffer: &WgpuBuffer, out: &mut [f64]) {
        let hi = read_buffer_f32(&self.device, &self.queue, &buffer.buffer, buffer.n);
        self.readback_count.set(self.readback_count.get() + 1);
        if let Some(ref lo_buf) = buffer.buffer_lo {
            let lo = read_buffer_f32(&self.device, &self.queue, lo_buf, buffer.n);
            self.readback_count.set(self.readback_count.get() + 1);
            for i in 0..buffer.n {
                out[i] = ds_to_f64(hi[i], lo[i]);
            }
        } else {
            for i in 0..buffer.n {
                out[i] = hi[i] as f64;
            }
        }
    }

    /// Upload a CSR matrix from f64 values with DS precision splitting.
    pub fn upload_matrix_f64(
        &self,
        values: &[f64],
        col_indices: &[u32],
        row_pointers: &[u32],
        n: usize,
    ) -> GpuCsrMatrix {
        let values_hi: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let values_lo: Vec<f32> = values
            .iter()
            .zip(values_hi.iter())
            .map(|(&v, &h)| (v - h as f64) as f32)
            .collect();

        let values_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_values_hi"),
                contents: bytemuck::cast_slice(&values_hi),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let values_lo_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_values_lo"),
                contents: bytemuck::cast_slice(&values_lo),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let col_indices_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_col_indices"),
                contents: bytemuck::cast_slice(col_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let row_pointers_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_row_ptrs"),
                contents: bytemuck::cast_slice(row_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let spmv_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_spmv_params"),
                contents: bytemuck::bytes_of(&(n as u32)),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        GpuCsrMatrix {
            values: values_buf,
            values_lo: Some(values_lo_buf),
            col_indices: col_indices_buf,
            row_pointers: row_pointers_buf,
            spmv_params: spmv_params_buf,
            n,
        }
    }

    /// Upload a read-only DS storage buffer (e.g., Jacobi inverse diagonal).
    pub fn upload_storage_buffer_f64(&self, data: &[f64]) -> WgpuBuffer {
        let hi: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let lo: Vec<f32> = data
            .iter()
            .zip(hi.iter())
            .map(|(&v, &h)| (v - h as f64) as f32)
            .collect();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_storage_hi"),
                contents: bytemuck::cast_slice(&hi),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buffer_lo = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_storage_lo"),
                contents: bytemuck::cast_slice(&lo),
                usage: wgpu::BufferUsages::STORAGE,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: Some(buffer_lo),
            n: data.len(),
        }
    }

    /// Apply Jacobi preconditioner on GPU using DS arithmetic.
    pub fn jacobi_apply(&self, inv_diag: &WgpuBuffer, input: &WgpuBuffer, output: &WgpuBuffer) {
        let n_u32 = input.n as u32;
        let n_wg = workgroup_count(n_u32);

        let params = DsVecParams {
            alpha_hi: 0.0,
            alpha_lo: 0.0,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.jacobi.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inv_diag.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inv_diag.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.jacobi);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
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
        label: Some("ds_read_staging"),
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

impl SolverBackend for WgpuDsBackend {
    type Buffer = WgpuBuffer;

    fn new_buffer(&self, n: usize) -> WgpuBuffer {
        let zeros = vec![0.0f32; n];
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_buffer_hi"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let buffer_lo = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_buffer_lo"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: Some(buffer_lo),
            n,
        }
    }

    fn upload_vec(&self, data: &[f32], buffer: &WgpuBuffer) {
        self.queue
            .write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(data));
        let zeros = vec![0.0f32; data.len()];
        self.queue.write_buffer(
            buffer.buffer_lo.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&zeros),
        );
    }

    fn upload_matrix(
        &self,
        values: &[f32],
        col_indices: &[u32],
        row_pointers: &[u32],
        n: usize,
    ) -> GpuCsrMatrix {
        let values_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_values_hi"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let zeros = vec![0.0f32; values.len()];
        let values_lo_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_values_lo"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let col_indices_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_col_indices"),
                contents: bytemuck::cast_slice(col_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let row_pointers_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_csr_row_ptrs"),
                contents: bytemuck::cast_slice(row_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let spmv_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_spmv_params"),
                contents: bytemuck::bytes_of(&(n as u32)),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        GpuCsrMatrix {
            values: values_buf,
            values_lo: Some(values_lo_buf),
            col_indices: col_indices_buf,
            row_pointers: row_pointers_buf,
            spmv_params: spmv_params_buf,
            n,
        }
    }

    fn download_vec(&self, buffer: &WgpuBuffer, out: &mut [f32]) {
        self.readback_count.set(self.readback_count.get() + 1);
        let result = read_buffer_f32(&self.device, &self.queue, &buffer.buffer, buffer.n);
        out[..buffer.n].copy_from_slice(&result);
    }

    fn spmv(&self, matrix: &GpuCsrMatrix, x: &WgpuBuffer, y: &WgpuBuffer) {
        let n_wg = workgroup_count(matrix.n as u32);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.spmv.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix.values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.values_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix.col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: matrix.row_pointers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: x.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: y.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: matrix.spmv_params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.spmv);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    fn dot(&self, x: &WgpuBuffer, y: &WgpuBuffer) -> f64 {
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let out_size = (n_wg as usize * std::mem::size_of::<f32>()) as u64;
        let dot_out_hi = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ds_dot_out_hi"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dot_out_lo = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ds_dot_out_lo"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = DsVecParams {
            alpha_hi: 0.0,
            alpha_lo: 0.0,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.dot.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dot_out_hi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dot_out_lo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.dot);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));

        let partials_hi = read_buffer_f32(&self.device, &self.queue, &dot_out_hi, n_wg as usize);
        let partials_lo = read_buffer_f32(&self.device, &self.queue, &dot_out_lo, n_wg as usize);
        self.readback_count.set(self.readback_count.get() + 1);

        let mut sum = 0.0f64;
        for i in 0..n_wg as usize {
            sum += ds_to_f64(partials_hi[i], partials_lo[i]);
        }
        sum
    }

    fn dot_n(&self, x: &WgpuBuffer, y: &WgpuBuffer, n: usize) -> f64 {
        if n == x.n {
            return self.dot(x, y);
        }
        let n_u32 = n as u32;
        let n_wg = workgroup_count(n_u32);

        let out_size = (n_wg as usize * std::mem::size_of::<f32>()) as u64;
        let dot_out_hi = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ds_dot_out_hi_n"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dot_out_lo = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ds_dot_out_lo_n"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = DsVecParams {
            alpha_hi: 0.0,
            alpha_lo: 0.0,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.dot.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dot_out_hi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dot_out_lo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.dot);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));

        let partials_hi = read_buffer_f32(&self.device, &self.queue, &dot_out_hi, n_wg as usize);
        let partials_lo = read_buffer_f32(&self.device, &self.queue, &dot_out_lo, n_wg as usize);
        self.readback_count.set(self.readback_count.get() + 1);

        let mut sum = 0.0f64;
        for i in 0..n_wg as usize {
            sum += ds_to_f64(partials_hi[i], partials_lo[i]);
        }
        sum
    }

    fn axpy(&self, alpha: f64, x: &WgpuBuffer, y: &WgpuBuffer) {
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let (alpha_hi, alpha_lo) = f64_to_ds(alpha);
        let params = DsVecParams {
            alpha_hi,
            alpha_lo,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.axpy.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.axpy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    fn scale(&self, alpha: f64, x: &WgpuBuffer) {
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let (alpha_hi, alpha_lo) = f64_to_ds(alpha);
        let params = DsVecParams {
            alpha_hi,
            alpha_lo,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.scale.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.scale);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    fn copy(&self, src: &WgpuBuffer, dst: &WgpuBuffer) {
        let n_u32 = src.n as u32;
        let n_wg = workgroup_count(n_u32);

        let params = DsVecParams {
            alpha_hi: 0.0,
            alpha_lo: 0.0,
            n: n_u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.copy.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.copy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }
}

// ── DS nonlinear pipelines ──────────────────────────────────────────

/// Compute pipelines for DS-precision nonlinear shaders.
struct DsNonlinearPipelines {
    diode_eval: wgpu::ComputePipeline,
    assemble_matrix_copy: wgpu::ComputePipeline,
    assemble_matrix_stamp: wgpu::ComputePipeline,
    assemble_rhs_copy: wgpu::ComputePipeline,
    assemble_rhs_stamp: wgpu::ComputePipeline,
    voltage_limit: wgpu::ComputePipeline,
    convergence_check: wgpu::ComputePipeline,
    bjt_eval: wgpu::ComputePipeline,
    assemble_bjt_matrix_stamp: wgpu::ComputePipeline,
    assemble_bjt_rhs_stamp: wgpu::ComputePipeline,
    bjt_voltage_limit: wgpu::ComputePipeline,
    mosfet_eval: wgpu::ComputePipeline,
    assemble_mosfet_matrix_stamp: wgpu::ComputePipeline,
    assemble_mosfet_rhs_stamp: wgpu::ComputePipeline,
    mosfet_voltage_limit_reduce: wgpu::ComputePipeline,
    mosfet_voltage_limit_apply: wgpu::ComputePipeline,
}

impl DsNonlinearPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let mk_mod = |label, body| {
            let src = ds_shaders::ds_shader(body);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let diode_eval_mod = mk_mod("ds_diode_eval", ds_nonlinear_shaders::DS_DIODE_EVAL_BODY);
        let assemble_mod =
            mk_mod("ds_assemble", ds_nonlinear_shaders::DS_NONLINEAR_ASSEMBLE_BODY);
        let conv_mod = mk_mod("ds_convergence", ds_nonlinear_shaders::DS_CONVERGENCE_BODY);
        let bjt_eval_mod = mk_mod("ds_bjt_eval", ds_nonlinear_shaders::DS_BJT_EVAL_BODY);
        let bjt_asm_mod = mk_mod("ds_bjt_assemble", ds_nonlinear_shaders::DS_BJT_ASSEMBLE_BODY);
        let bjt_vl_mod = mk_mod(
            "ds_bjt_vlimit",
            ds_nonlinear_shaders::DS_BJT_VOLTAGE_LIMIT_BODY,
        );
        let mos_eval_mod = mk_mod("ds_mosfet_eval", ds_nonlinear_shaders::DS_MOSFET_EVAL_BODY);
        let mos_asm_mod = mk_mod(
            "ds_mosfet_assemble",
            ds_nonlinear_shaders::DS_MOSFET_ASSEMBLE_BODY,
        );
        let mos_vl_mod = mk_mod(
            "ds_mosfet_vlimit",
            ds_nonlinear_shaders::DS_MOSFET_VOLTAGE_LIMIT_BODY,
        );

        let mk = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            diode_eval: mk(&diode_eval_mod, "diode_eval_ds"),
            assemble_matrix_copy: mk(&assemble_mod, "assemble_matrix_copy_ds"),
            assemble_matrix_stamp: mk(&assemble_mod, "assemble_matrix_stamp_ds"),
            assemble_rhs_copy: mk(&assemble_mod, "assemble_rhs_copy_ds"),
            assemble_rhs_stamp: mk(&assemble_mod, "assemble_rhs_stamp_ds"),
            voltage_limit: mk(&conv_mod, "voltage_limit_ds"),
            convergence_check: mk(&conv_mod, "convergence_check_ds"),
            bjt_eval: mk(&bjt_eval_mod, "bjt_eval_ds"),
            assemble_bjt_matrix_stamp: mk(&bjt_asm_mod, "assemble_bjt_matrix_stamp_ds"),
            assemble_bjt_rhs_stamp: mk(&bjt_asm_mod, "assemble_bjt_rhs_stamp_ds"),
            bjt_voltage_limit: mk(&bjt_vl_mod, "bjt_voltage_limit_ds"),
            mosfet_eval: mk(&mos_eval_mod, "mosfet_eval_ds"),
            assemble_mosfet_matrix_stamp: mk(&mos_asm_mod, "assemble_mosfet_matrix_stamp_ds"),
            assemble_mosfet_rhs_stamp: mk(&mos_asm_mod, "assemble_mosfet_rhs_stamp_ds"),
            mosfet_voltage_limit_reduce: mk(&mos_vl_mod, "mosfet_voltage_limit_reduce_ds"),
            mosfet_voltage_limit_apply: mk(&mos_vl_mod, "mosfet_voltage_limit_apply_ds"),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn read_buffer_u32_ds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> Vec<u32> {
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ds_read_staging_u32"),
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
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

impl WgpuDsBackend {
    fn nonlinear_pipelines(&self) -> DsNonlinearPipelines {
        DsNonlinearPipelines::new(&self.device)
    }

    /// Create a flat storage buffer without hi/lo split (for interleaved DS eval outputs).
    pub fn new_buffer_flat(&self, n: usize) -> WgpuBuffer {
        let zeros = vec![0.0f32; n];
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_flat_buffer"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n,
        }
    }

    /// Upload diode descriptors (u32 only, no lo buffer).
    pub fn upload_diode_descriptors(&self, descriptors: &[GpuDiodeDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_diode_desc"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 12,
        }
    }

    /// Upload BJT descriptors (u32 only, no lo buffer).
    pub fn upload_bjt_descriptors(&self, descriptors: &[GpuBjtDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_bjt_desc"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 24,
        }
    }

    /// Upload MOSFET descriptors (u32 only, no lo buffer).
    pub fn upload_mosfet_descriptors(&self, descriptors: &[GpuMosfetDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_mosfet_desc"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 16,
        }
    }

    /// Get a reference to the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the GPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

// ── NonlinearBackend for DS ─────────────────────────────────────────

impl NonlinearBackend for WgpuDsBackend {
    type Buffer = WgpuBuffer;

    fn evaluate_diodes(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_diodes: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_diodes),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes.diode_eval.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: solution.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.diode_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_nonlinear_system(
        &self,
        base_values: &WgpuBuffer,
        base_b: &WgpuBuffer,
        diode_output: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        out_values: &WgpuBuffer,
        out_b: &WgpuBuffer,
        n_diodes: u32,
        matrix_nnz: u32,
        system_size: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1a: copy base matrix values (DS)
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&matrix_nnz),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_matrix_copy.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: base_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: base_values
                            .buffer_lo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values
                            .buffer_lo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_matrix_copy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(matrix_nnz), 1, 1);
        }

        // Pass 1b: stamp diode conductance into matrix (DS)
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_diodes),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_matrix_stamp.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: diode_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values
                            .buffer_lo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        // Pass 2a: copy base RHS (DS)
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&system_size),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_rhs_copy.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: base_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: base_b.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_b.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_rhs_copy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(system_size), 1, 1);
        }

        // Pass 2b: stamp Norton companion currents (DS)
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_diodes),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_rhs_stamp.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: solution.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: diode_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_b.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        self.dispatch_count.set(self.dispatch_count.get() + 4);
        self.queue.submit(Some(encoder.finish()));
    }

    fn evaluate_bjts(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_bjts: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_bjts),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes.bjt_eval.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: solution.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.bjt_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    fn evaluate_mosfets(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_mosfets: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_mosfets),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes.mosfet_eval.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: solution.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.mosfet_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_bjt_stamps(
        &self,
        bjt_output: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        out_values: &WgpuBuffer,
        out_b: &WgpuBuffer,
        n_bjts: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Matrix stamp
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_bjts),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_bjt_matrix_stamp.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bjt_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values
                            .buffer_lo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_bjt_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }

        // RHS stamp
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_bjts),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_bjt_rhs_stamp.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: solution.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bjt_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_b.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_bjt_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }

        self.dispatch_count.set(self.dispatch_count.get() + 2);
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_mosfet_stamps(
        &self,
        mosfet_output: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        out_values: &WgpuBuffer,
        out_b: &WgpuBuffer,
        n_mosfets: u32,
    ) {
        let pipes = self.nonlinear_pipelines();
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Matrix stamp
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_mosfets),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes
                    .assemble_mosfet_matrix_stamp
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mosfet_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values
                            .buffer_lo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_mosfet_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }

        // RHS stamp
        {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_mosfets),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.assemble_mosfet_rhs_stamp.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: solution.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: solution.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mosfet_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_b.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.assemble_mosfet_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }

        self.dispatch_count.set(self.dispatch_count.get() + 2);
        self.queue.submit(Some(encoder.finish()));
    }

    fn limit_bjt_voltages(
        &self,
        x_old: &WgpuBuffer,
        x_new: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        n_bjts: u32,
    ) {
        if n_bjts == 0 {
            return;
        }
        let pipes = self.nonlinear_pipelines();
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_bjts),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes.bjt_voltage_limit.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_old.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_old.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_new.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: x_new.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.bjt_voltage_limit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        self.queue.submit(Some(encoder.finish()));
    }

    fn limit_mosfet_voltages(
        &self,
        x_old: &WgpuBuffer,
        x_new: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        n_mosfets: u32,
    ) {
        if n_mosfets == 0 {
            return;
        }
        let pipes = self.nonlinear_pipelines();

        let Ok(n_nodes) = u32::try_from(x_new.n) else {
            return;
        };

        const SCALE_ONE: u32 = 1_000_000;
        let node_scale_init = vec![SCALE_ONE; x_new.n];
        let node_scale_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_mosfet_node_limit_scale"),
                contents: bytemuck::cast_slice(&node_scale_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MosfetVoltLimitParams {
            num_mosfets: u32,
            n_nodes: u32,
            _pad0: u32,
            _pad1: u32,
        }
        let params = MosfetVoltLimitParams {
            num_mosfets: n_mosfets,
            n_nodes,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg_reduce = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes
                .mosfet_voltage_limit_reduce
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_old.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_old.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_new.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: x_new.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: node_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let bg_apply = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes
                .mosfet_voltage_limit_apply
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_old.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_old.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_new.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: x_new.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: node_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.mosfet_voltage_limit_reduce);
            pass.set_bind_group(0, Some(&bg_reduce), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.mosfet_voltage_limit_apply);
            pass.set_bind_group(0, Some(&bg_apply), &[]);
            pass.dispatch_workgroups(workgroup_count(n_nodes), 1, 1);
        }
        self.dispatch_count.set(self.dispatch_count.get() + 2);
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    fn limit_and_check_convergence(
        &self,
        x_old: &WgpuBuffer,
        x_new: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        _result_buf: &WgpuBuffer,
        tolerance: f64,
        n_diodes: u32,
        system_size: u32,
    ) -> ConvergenceResult {
        let pipes = self.nonlinear_pipelines();
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1: voltage limiting (diodes)
        if n_diodes > 0 {
            let p = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&n_diodes),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes.voltage_limit.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptors.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_old.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x_old.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: x_new.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: x_new.buffer_lo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.voltage_limit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        // Pass 2: convergence check
        let n_wg = workgroup_count(system_size);
        let partial_max_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ds_conv_partial_max"),
            size: (n_wg as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let flags_data: [u32; 2] = [0, 0];
        let flags_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ds_conv_flags"),
                contents: bytemuck::cast_slice(&flags_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvParams {
            n: u32,
            tolerance: f32,
        }
        let conv_params = ConvParams {
            n: system_size,
            tolerance: tolerance as f32,
        };
        let p = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&conv_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipes.convergence_check.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_new.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_new.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_old.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_old.buffer_lo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: partial_max_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: p.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.convergence_check);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }

        self.dispatch_count
            .set(self.dispatch_count.get() + if n_diodes > 0 { 2 } else { 1 });
        self.queue.submit(Some(encoder.finish()));

        // Read back partial maxes and flags
        let partials =
            read_buffer_f32(&self.device, &self.queue, &partial_max_buf, n_wg as usize);
        let flags = read_buffer_u32_ds(&self.device, &self.queue, &flags_buf, 2);
        self.readback_count.set(self.readback_count.get() + 2);

        if flags[1] != 0 {
            return ConvergenceResult::NumericalError;
        }

        let max_diff = partials.iter().cloned().fold(0.0f32, f32::max);

        if (max_diff as f64) < tolerance {
            ConvergenceResult::Converged
        } else {
            ConvergenceResult::NotConverged {
                max_diff: max_diff as f64,
            }
        }
    }
}
