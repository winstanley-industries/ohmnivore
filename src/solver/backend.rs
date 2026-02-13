//! Solver backend abstraction for GPU-accelerated linear algebra.
//!
//! Defines the `SolverBackend` trait for backend-agnostic iterative solvers,
//! and `WgpuBackend` which implements it using wgpu compute shaders.

use crate::error::{OhmnivoreError, Result};
use wgpu::util::DeviceExt;

use super::gpu_shaders;

const WORKGROUP_SIZE: u32 = 64;

fn workgroup_count(n: u32) -> u32 {
    n.div_ceil(WORKGROUP_SIZE)
}

/// Uploaded CSR matrix stored in GPU buffers.
pub struct GpuCsrMatrix {
    pub(crate) values: wgpu::Buffer,
    pub(crate) col_indices: wgpu::Buffer,
    pub(crate) row_pointers: wgpu::Buffer,
    pub(crate) spmv_params: wgpu::Buffer,
    pub(crate) n: usize,
}

/// A GPU buffer wrapping a `wgpu::Buffer` with element count metadata.
pub struct WgpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) n: usize,
}

// Shader VecParams layout: { alpha_re: f32, alpha_im: f32, n: u32 }
// With WGSL struct alignment, this is 12 bytes but uniform requires 16-byte alignment.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VecParams {
    alpha_re: f32,
    alpha_im: f32,
    n: u32,
    _pad: u32,
}

/// Abstract backend for linear algebra operations on f32 vectors.
///
/// All vector operations work on f32 precision. The `dot` method returns f64
/// for accumulation precision (partial f32 sums reduced into f64 on the CPU).
pub trait SolverBackend {
    type Buffer;

    /// Sparse matrix-vector multiply: y = A * x
    fn spmv(&self, matrix: &GpuCsrMatrix, x: &Self::Buffer, y: &Self::Buffer);

    /// Dot product: returns x . y as f64 for precision.
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> f64;

    /// AXPY: y = alpha * x + y (in-place on y)
    fn axpy(&self, alpha: f64, x: &Self::Buffer, y: &Self::Buffer);

    /// Copy: dst = src
    fn copy(&self, src: &Self::Buffer, dst: &Self::Buffer);

    /// Scale: x = alpha * x (in-place)
    fn scale(&self, alpha: f64, x: &Self::Buffer);

    /// Create a new zero-initialized buffer of size n.
    fn new_buffer(&self, n: usize) -> Self::Buffer;

    /// Upload a CSR matrix to GPU-ready format.
    fn upload_matrix(
        &self,
        values: &[f32],
        col_indices: &[u32],
        row_pointers: &[u32],
        n: usize,
    ) -> GpuCsrMatrix;

    /// Upload a vector to an existing buffer.
    fn upload_vec(&self, data: &[f32], buffer: &Self::Buffer);

    /// Download a buffer to CPU.
    fn download_vec(&self, buffer: &Self::Buffer, out: &mut [f32]);
}

/// GPU context holding the wgpu device, queue, and compute pipelines.
#[allow(dead_code)]
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    // Real pipelines
    pub(crate) spmv_real_pipeline: wgpu::ComputePipeline,
    pub(crate) dot_real_pipeline: wgpu::ComputePipeline,
    pub(crate) axpy_real_pipeline: wgpu::ComputePipeline,
    pub(crate) scale_real_pipeline: wgpu::ComputePipeline,
    pub(crate) copy_real_pipeline: wgpu::ComputePipeline,
    pub(crate) subtract_real_pipeline: wgpu::ComputePipeline,
    pub(crate) jacobi_real_pipeline: wgpu::ComputePipeline,
    // Complex pipelines
    pub(crate) spmv_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) dot_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) axpy_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) scale_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) copy_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) subtract_complex_pipeline: wgpu::ComputePipeline,
    pub(crate) jacobi_complex_pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
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
                label: Some("ohmnivore_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| OhmnivoreError::Solve(format!("failed to get GPU device: {e}")))?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ohmnivore_shaders"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::SHADER_SOURCE.into()),
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

        let spmv_real_pipeline = make_pipeline("spmv_real");
        let dot_real_pipeline = make_pipeline("dot_real");
        let axpy_real_pipeline = make_pipeline("axpy_real");
        let scale_real_pipeline = make_pipeline("scale_real");
        let copy_real_pipeline = make_pipeline("copy_real");
        let subtract_real_pipeline = make_pipeline("subtract_real");
        let jacobi_real_pipeline = make_pipeline("jacobi_real");
        let spmv_complex_pipeline = make_pipeline("spmv_complex");
        let dot_complex_pipeline = make_pipeline("dot_complex");
        let axpy_complex_pipeline = make_pipeline("axpy_complex");
        let scale_complex_pipeline = make_pipeline("scale_complex");
        let copy_complex_pipeline = make_pipeline("copy_complex");
        let subtract_complex_pipeline = make_pipeline("subtract_complex");
        let jacobi_complex_pipeline = make_pipeline("jacobi_complex");
        let _ = make_pipeline;

        Ok(Self {
            device,
            queue,
            spmv_real_pipeline,
            dot_real_pipeline,
            axpy_real_pipeline,
            scale_real_pipeline,
            copy_real_pipeline,
            subtract_real_pipeline,
            jacobi_real_pipeline,
            spmv_complex_pipeline,
            dot_complex_pipeline,
            axpy_complex_pipeline,
            scale_complex_pipeline,
            copy_complex_pipeline,
            subtract_complex_pipeline,
            jacobi_complex_pipeline,
        })
    }
}

/// Wgpu-based implementation of `SolverBackend`.
pub struct WgpuBackend {
    pub(crate) ctx: GpuContext,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ctx: GpuContext::new()?,
        })
    }

    /// Apply Jacobi preconditioner on GPU: output[i] = inv_diag[i] * input[i]
    pub fn jacobi_apply(
        &self,
        inv_diag: &WgpuBuffer,
        input: &WgpuBuffer,
        output: &WgpuBuffer,
    ) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_u32 = input.n as u32;
        let n_wg = workgroup_count(n_u32);

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
            layout: &self.ctx.jacobi_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inv_diag.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
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
            pass.set_pipeline(&self.ctx.jacobi_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Upload a read-only f32 storage buffer (e.g. for Jacobi inverse diagonal).
    pub fn upload_storage_buffer(&self, data: &[f32]) -> WgpuBuffer {
        let buffer =
            self.ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("storage_readonly"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        WgpuBuffer {
            buffer,
            n: data.len(),
        }
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

impl SolverBackend for WgpuBackend {
    type Buffer = WgpuBuffer;

    fn spmv(&self, matrix: &GpuCsrMatrix, x: &WgpuBuffer, y: &WgpuBuffer) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_wg = workgroup_count(matrix.n as u32);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.ctx.spmv_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix.values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix.row_pointers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: matrix.spmv_params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.ctx.spmv_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn dot(&self, x: &WgpuBuffer, y: &WgpuBuffer) -> f64 {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let dot_out_size = (n_wg as usize * std::mem::size_of::<f32>()) as u64;
        let dot_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dot_out"),
            size: dot_out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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
            layout: &self.ctx.dot_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y.buffer.as_entire_binding(),
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
            pass.set_pipeline(&self.ctx.dot_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        // Read back partial sums and reduce on CPU with f64 precision
        let partials = read_buffer_f32(device, queue, &dot_out_buf, n_wg as usize);
        partials.iter().map(|&v| v as f64).sum()
    }

    fn axpy(&self, alpha: f64, x: &WgpuBuffer, y: &WgpuBuffer) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let params = VecParams {
            alpha_re: alpha as f32,
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
            layout: &self.ctx.axpy_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y.buffer.as_entire_binding(),
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
            pass.set_pipeline(&self.ctx.axpy_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn copy(&self, src: &WgpuBuffer, dst: &WgpuBuffer) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_u32 = src.n as u32;
        let n_wg = workgroup_count(n_u32);

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
            layout: &self.ctx.copy_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.buffer.as_entire_binding(),
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
            pass.set_pipeline(&self.ctx.copy_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn scale(&self, alpha: f64, x: &WgpuBuffer) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_u32 = x.n as u32;
        let n_wg = workgroup_count(n_u32);

        let params = VecParams {
            alpha_re: alpha as f32,
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
            layout: &self.ctx.scale_real_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
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
            pass.set_pipeline(&self.ctx.scale_real_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn new_buffer(&self, n: usize) -> WgpuBuffer {
        let zeros = vec![0.0f32; n];
        let buffer =
            self.ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("backend_buffer"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });
        WgpuBuffer { buffer, n }
    }

    fn upload_matrix(
        &self,
        values: &[f32],
        col_indices: &[u32],
        row_pointers: &[u32],
        n: usize,
    ) -> GpuCsrMatrix {
        let device = &self.ctx.device;

        let values_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let col_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_col_indices"),
            contents: bytemuck::cast_slice(col_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row_pointers_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_row_ptrs"),
            contents: bytemuck::cast_slice(row_pointers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let spmv_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spmv_params"),
            contents: bytemuck::bytes_of(&(n as u32)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        GpuCsrMatrix {
            values: values_buf,
            col_indices: col_indices_buf,
            row_pointers: row_pointers_buf,
            spmv_params: spmv_params_buf,
            n,
        }
    }

    fn upload_vec(&self, data: &[f32], buffer: &WgpuBuffer) {
        self.ctx
            .queue
            .write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(data));
    }

    fn download_vec(&self, buffer: &WgpuBuffer, out: &mut [f32]) {
        let result = read_buffer_f32(&self.ctx.device, &self.ctx.queue, &buffer.buffer, buffer.n);
        out[..buffer.n].copy_from_slice(&result);
    }
}
