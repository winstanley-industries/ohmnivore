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
