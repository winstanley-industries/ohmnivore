//! Double-single precision wgpu backend for BiCGSTAB.
//!
//! Implements `SolverBackend` using DS arithmetic (~48-bit mantissa from
//! paired f32 values). This is a wgpu-specific workaround for the lack of
//! native f64 support; CUDA/ROCm backends use f64 directly.

use std::cell::Cell;

use wgpu::util::DeviceExt;

use crate::error::{OhmnivoreError, Result};

use super::backend::{GpuCsrMatrix, SolverBackend, WgpuBuffer};
use super::ds_shaders;

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
            source: wgpu::ShaderSource::Wgsl(ds_shaders::DS_SHADER_SOURCE.into()),
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
