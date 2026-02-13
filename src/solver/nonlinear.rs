//! Nonlinear solver backend for Newton-Raphson iteration on GPU.
//!
//! Extends `SolverBackend` with diode evaluation, nonlinear system assembly,
//! and convergence checking. The `WgpuBackend` implementation dispatches
//! WGSL compute shaders for each operation.

use wgpu::util::DeviceExt;

use super::backend::{WgpuBackend, WgpuBuffer};
use super::gpu_shaders;
use crate::compiler::{GpuBjtDescriptor, GpuMosfetDescriptor};
use crate::error::Result;

const WORKGROUP_SIZE: u32 = 64;

fn workgroup_count(n: u32) -> u32 {
    n.div_ceil(WORKGROUP_SIZE)
}

/// GPU-side diode descriptor. Layout must exactly match the WGSL flat u32
/// descriptor format (12 words = 48 bytes per diode).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDiodeDescriptor {
    pub anode_idx: u32,
    pub cathode_idx: u32,
    pub is_val: f32,
    pub n_vt: f32,
    /// CSR value indices for the 4 G-matrix conductance stamps:
    /// \[anode,anode\], \[anode,cathode\], \[cathode,anode\], \[cathode,cathode\]
    pub g_row_col: [u32; 4],
    /// RHS vector indices for anode and cathode nodes.
    pub b_idx: [u32; 2],
    pub _pad: [u32; 2],
}

/// Result of a Newton-Raphson convergence check.
#[derive(Debug, Clone)]
pub enum ConvergenceResult {
    Converged,
    NotConverged { max_diff: f32 },
    NumericalError,
}

/// Nonlinear solver operations for Newton-Raphson iteration.
///
/// Implementors must also implement [`super::backend::SolverBackend`].
pub trait NonlinearBackend {
    type Buffer;

    /// Evaluate all diodes: compute i_d and g_d for each diode given the
    /// current solution vector.
    ///
    /// Writes 2 floats per diode into `output`: \[i_d, g_d\].
    fn evaluate_diodes(
        &self,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        output: &Self::Buffer,
        n_diodes: u32,
    );

    /// Assemble the nonlinear system by copying base matrix/RHS values then
    /// stamping diode contributions.
    ///
    /// - Copies `base_values` -> `out_values`, then stamps g_d at CSR positions.
    /// - Copies `base_b` -> `out_b`, then stamps Norton companion currents.
    #[allow(clippy::too_many_arguments)]
    fn assemble_nonlinear_system(
        &self,
        base_values: &Self::Buffer,
        base_b: &Self::Buffer,
        diode_output: &Self::Buffer,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        out_values: &Self::Buffer,
        out_b: &Self::Buffer,
        n_diodes: u32,
        matrix_nnz: u32,
        system_size: u32,
    );

    /// Evaluate all BJTs: compute I_C, I_B, and 4 Jacobian derivatives.
    ///
    /// Writes 6 floats per BJT into `output`:
    /// \[I_C, I_B, dI_C/dV_BE, dI_C/dV_BC, dI_B/dV_BE, dI_B/dV_BC\].
    fn evaluate_bjts(
        &self,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        output: &Self::Buffer,
        n_bjts: u32,
    );

    /// Evaluate all MOSFETs: compute I_D, g_m, g_ds.
    ///
    /// Writes 4 floats per MOSFET into `output`: \[I_D, g_m, g_ds, polarity\].
    fn evaluate_mosfets(
        &self,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        output: &Self::Buffer,
        n_mosfets: u32,
    );

    /// Stamp BJT contributions into the assembled G-matrix and RHS vector.
    ///
    /// Does NOT copy base values — assumes they are already copied.
    #[allow(clippy::too_many_arguments)]
    fn assemble_bjt_stamps(
        &self,
        bjt_output: &Self::Buffer,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        out_values: &Self::Buffer,
        out_b: &Self::Buffer,
        n_bjts: u32,
    );

    /// Stamp MOSFET contributions into the assembled G-matrix and RHS vector.
    ///
    /// Does NOT copy base values — assumes they are already copied.
    #[allow(clippy::too_many_arguments)]
    fn assemble_mosfet_stamps(
        &self,
        mosfet_output: &Self::Buffer,
        descriptors: &Self::Buffer,
        solution: &Self::Buffer,
        out_values: &Self::Buffer,
        out_b: &Self::Buffer,
        n_mosfets: u32,
    );

    /// Apply voltage limiting to BJT junction voltages.
    fn limit_bjt_voltages(
        &self,
        x_old: &Self::Buffer,
        x_new: &Self::Buffer,
        descriptors: &Self::Buffer,
        n_bjts: u32,
    );

    /// Apply voltage limiting to MOSFET terminal voltages.
    fn limit_mosfet_voltages(
        &self,
        x_old: &Self::Buffer,
        x_new: &Self::Buffer,
        descriptors: &Self::Buffer,
        n_mosfets: u32,
    );

    /// Apply voltage limiting then check convergence of the Newton step.
    ///
    /// Modifies `x_new` in-place (voltage limiting), then computes the max
    /// absolute difference between old and new solutions.
    #[allow(clippy::too_many_arguments)]
    fn limit_and_check_convergence(
        &self,
        x_old: &Self::Buffer,
        x_new: &Self::Buffer,
        descriptors: &Self::Buffer,
        result_buf: &Self::Buffer,
        tolerance: f32,
        n_diodes: u32,
        system_size: u32,
    ) -> ConvergenceResult;
}

/// Lazily-initialized compute pipelines for nonlinear shaders.
struct NonlinearPipelines {
    // Diode evaluation
    diode_eval: wgpu::ComputePipeline,
    // Assembly: matrix copy + stamp, RHS copy + stamp
    assemble_matrix_copy: wgpu::ComputePipeline,
    assemble_matrix_stamp: wgpu::ComputePipeline,
    assemble_rhs_copy: wgpu::ComputePipeline,
    assemble_rhs_stamp: wgpu::ComputePipeline,
    // Convergence: voltage limit + convergence check
    voltage_limit: wgpu::ComputePipeline,
    convergence_check: wgpu::ComputePipeline,
    // BJT pipelines
    bjt_eval: wgpu::ComputePipeline,
    assemble_bjt_matrix_stamp: wgpu::ComputePipeline,
    assemble_bjt_rhs_stamp: wgpu::ComputePipeline,
    bjt_voltage_limit: wgpu::ComputePipeline,
    // MOSFET pipelines
    mosfet_eval: wgpu::ComputePipeline,
    assemble_mosfet_matrix_stamp: wgpu::ComputePipeline,
    assemble_mosfet_rhs_stamp: wgpu::ComputePipeline,
    mosfet_voltage_limit: wgpu::ComputePipeline,
}

impl NonlinearPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let diode_eval_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diode_eval_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::DIODE_EVAL_SHADER.into()),
        });
        let assemble_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nonlinear_assemble_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::NONLINEAR_ASSEMBLE_SHADER.into()),
        });
        let convergence_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("convergence_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::CONVERGENCE_SHADER.into()),
        });
        let bjt_eval_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bjt_eval_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::BJT_EVAL_SHADER.into()),
        });
        let bjt_assemble_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bjt_assemble_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::BJT_ASSEMBLE_SHADER.into()),
        });
        let bjt_vlimit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bjt_voltage_limit_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::BJT_VOLTAGE_LIMIT_SHADER.into()),
        });
        let mosfet_eval_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mosfet_eval_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::MOSFET_EVAL_SHADER.into()),
        });
        let mosfet_assemble_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mosfet_assemble_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::MOSFET_ASSEMBLE_SHADER.into()),
        });
        let mosfet_vlimit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mosfet_voltage_limit_shader"),
            source: wgpu::ShaderSource::Wgsl(gpu_shaders::MOSFET_VOLTAGE_LIMIT_SHADER.into()),
        });

        let make = |module: &wgpu::ShaderModule, entry: &str| -> wgpu::ComputePipeline {
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
            diode_eval: make(&diode_eval_module, "diode_eval"),
            assemble_matrix_copy: make(&assemble_module, "assemble_matrix_copy"),
            assemble_matrix_stamp: make(&assemble_module, "assemble_matrix_stamp"),
            assemble_rhs_copy: make(&assemble_module, "assemble_rhs_copy"),
            assemble_rhs_stamp: make(&assemble_module, "assemble_rhs_stamp"),
            voltage_limit: make(&convergence_module, "voltage_limit"),
            convergence_check: make(&convergence_module, "convergence_check"),
            bjt_eval: make(&bjt_eval_module, "bjt_eval"),
            assemble_bjt_matrix_stamp: make(&bjt_assemble_module, "assemble_bjt_matrix_stamp"),
            assemble_bjt_rhs_stamp: make(&bjt_assemble_module, "assemble_bjt_rhs_stamp"),
            bjt_voltage_limit: make(&bjt_vlimit_module, "bjt_voltage_limit"),
            mosfet_eval: make(&mosfet_eval_module, "mosfet_eval"),
            assemble_mosfet_matrix_stamp: make(
                &mosfet_assemble_module,
                "assemble_mosfet_matrix_stamp",
            ),
            assemble_mosfet_rhs_stamp: make(&mosfet_assemble_module, "assemble_mosfet_rhs_stamp"),
            mosfet_voltage_limit: make(&mosfet_vlimit_module, "mosfet_voltage_limit"),
        }
    }
}

/// Holds the lazily-initialized nonlinear pipelines for WgpuBackend.
pub struct NonlinearState {
    #[allow(dead_code)]
    pipelines: NonlinearPipelines,
}

impl WgpuBackend {
    /// Get or create the nonlinear pipelines.
    fn nonlinear_pipelines(&self) -> NonlinearPipelines {
        NonlinearPipelines::new(&self.ctx.device)
    }

    /// Upload diode descriptors as a raw u32 storage buffer.
    pub fn upload_diode_descriptors(&self, descriptors: &[GpuDiodeDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("diode_descriptors"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        // n is in u32 words (12 words per diode)
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 12,
        }
    }

    /// Upload BJT descriptors as a raw u32 storage buffer.
    pub fn upload_bjt_descriptors(&self, descriptors: &[GpuBjtDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bjt_descriptors"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        // 24 u32 words per BJT (96 bytes / 4)
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 24,
        }
    }

    /// Upload MOSFET descriptors as a raw u32 storage buffer.
    pub fn upload_mosfet_descriptors(&self, descriptors: &[GpuMosfetDescriptor]) -> WgpuBuffer {
        let data: &[u8] = bytemuck::cast_slice(descriptors);
        let buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mosfet_descriptors"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        // 16 u32 words per MOSFET (64 bytes / 4)
        WgpuBuffer {
            buffer,
            buffer_lo: None,
            n: descriptors.len() * 16,
        }
    }

    /// Initialize nonlinear state (pipelines). Call once before Newton iteration.
    pub fn init_nonlinear(&self) -> Result<NonlinearState> {
        Ok(NonlinearState {
            pipelines: self.nonlinear_pipelines(),
        })
    }
}

impl NonlinearBackend for WgpuBackend {
    type Buffer = WgpuBuffer;

    fn evaluate_diodes(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_diodes: u32,
    ) {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n_wg = workgroup_count(n_diodes);

        // DiodeEvalParams { num_diodes: u32 }
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&n_diodes),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.diode_eval.get_bind_group_layout(0),
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
            pass.set_pipeline(&pipelines.diode_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

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
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut encoder = device.create_command_encoder(&Default::default());

        // --- Pass 1a: copy base matrix values ---
        {
            // AssembleMatParams { nnz: u32 }
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&matrix_nnz),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_matrix_copy.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: base_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_values.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_matrix_copy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(matrix_nnz), 1, 1);
        }

        // --- Pass 1b: stamp diode conductance into matrix ---
        {
            // StampMatParams { num_diodes: u32 }
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_diodes),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_matrix_stamp.get_bind_group_layout(0),
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
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        // --- Pass 2a: copy base RHS ---
        {
            // AssembleRhsParams { n: u32 }
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&system_size),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_rhs_copy.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: base_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_rhs_copy);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(system_size), 1, 1);
        }

        // --- Pass 2b: stamp Norton companion currents into RHS ---
        {
            // StampRhsParams { num_diodes: u32 }
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_diodes),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_rhs_stamp.get_bind_group_layout(0),
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
                        resource: diode_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    fn limit_and_check_convergence(
        &self,
        x_old: &WgpuBuffer,
        x_new: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        _result_buf: &WgpuBuffer,
        tolerance: f32,
        n_diodes: u32,
        system_size: u32,
    ) -> ConvergenceResult {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut encoder = device.create_command_encoder(&Default::default());

        // --- Pass 1: voltage limiting ---
        if n_diodes > 0 {
            // VoltLimitParams { num_diodes: u32 }
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_diodes),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.voltage_limit.get_bind_group_layout(0),
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
                        resource: x_new.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.voltage_limit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_diodes), 1, 1);
        }

        // --- Pass 2: convergence check ---
        let n_wg = workgroup_count(system_size);
        let partial_max_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_partial_max"),
            size: (n_wg as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Flags buffer: [0] = converged (unused by shader, we compute on CPU),
        //               [1] = NaN/Inf detected
        // Initialize to zero.
        let flags_data: [u32; 2] = [0, 0];
        let flags_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv_flags"),
            contents: bytemuck::cast_slice(&flags_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // ConvParams { n: u32, tolerance: f32 }
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvParams {
            n: u32,
            tolerance: f32,
        }
        let conv_params = ConvParams {
            n: system_size,
            tolerance,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&conv_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.convergence_check.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_new.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_old.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partial_max_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.convergence_check);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }

        queue.submit(Some(encoder.finish()));

        // Read back partial maxes and flags
        let partials = read_buffer_raw(device, queue, &partial_max_buf, n_wg as usize);
        let flags = read_buffer_u32(device, queue, &flags_buf, 2);

        // Check for NaN/Inf
        if flags[1] != 0 {
            return ConvergenceResult::NumericalError;
        }

        // CPU reduction of workgroup partial maxes
        let max_diff = partials.iter().cloned().fold(0.0f32, f32::max);

        if max_diff < tolerance {
            ConvergenceResult::Converged
        } else {
            ConvergenceResult::NotConverged { max_diff }
        }
    }

    fn evaluate_bjts(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_bjts: u32,
    ) {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&n_bjts),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.bjt_eval.get_bind_group_layout(0),
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
            pass.set_pipeline(&pipelines.bjt_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn evaluate_mosfets(
        &self,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        output: &WgpuBuffer,
        n_mosfets: u32,
    ) {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&n_mosfets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.mosfet_eval.get_bind_group_layout(0),
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
            pass.set_pipeline(&pipelines.mosfet_eval);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn assemble_bjt_stamps(
        &self,
        bjt_output: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        out_values: &WgpuBuffer,
        out_b: &WgpuBuffer,
        n_bjts: u32,
    ) {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut encoder = device.create_command_encoder(&Default::default());

        // Matrix stamp
        {
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_bjts),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_bjt_matrix_stamp.get_bind_group_layout(0),
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
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_bjt_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }

        // RHS stamp
        {
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_bjts),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_bjt_rhs_stamp.get_bind_group_layout(0),
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
                        resource: bjt_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_bjt_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    fn assemble_mosfet_stamps(
        &self,
        mosfet_output: &WgpuBuffer,
        descriptors: &WgpuBuffer,
        solution: &WgpuBuffer,
        out_values: &WgpuBuffer,
        out_b: &WgpuBuffer,
        n_mosfets: u32,
    ) {
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut encoder = device.create_command_encoder(&Default::default());

        // Matrix stamp
        {
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_mosfets),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines
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
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_mosfet_matrix_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }

        // RHS stamp
        {
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&n_mosfets),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.assemble_mosfet_rhs_stamp.get_bind_group_layout(0),
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
                        resource: mosfet_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_b.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.assemble_mosfet_rhs_stamp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }

        queue.submit(Some(encoder.finish()));
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
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&n_bjts),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.bjt_voltage_limit.get_bind_group_layout(0),
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
                    resource: x_new.buffer.as_entire_binding(),
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
            pass.set_pipeline(&pipelines.bjt_voltage_limit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_bjts), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
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
        let pipelines = self.nonlinear_pipelines();
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&n_mosfets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.mosfet_voltage_limit.get_bind_group_layout(0),
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
                    resource: x_new.buffer.as_entire_binding(),
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
            pass.set_pipeline(&pipelines.mosfet_voltage_limit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n_mosfets), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// Read a GPU buffer back to CPU as f32 values.
fn read_buffer_raw(
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

/// Read a GPU buffer back to CPU as u32 values.
fn read_buffer_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> Vec<u32> {
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_staging_u32"),
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
