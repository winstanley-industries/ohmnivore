//! Performance statistics collection for `--stats` output.

use std::time::{Duration, Instant};

/// Collects performance counters and phase timings.
///
/// Created when `--stats` is passed, threaded as `Option<&mut Stats>`.
/// Zero cost when `None` — no timing calls, no counter increments.
pub struct Stats {
    total_start: Instant,
    phases: Vec<(&'static str, Duration)>,
    // Newton-Raphson
    pub newton_iterations: u32,
    pub bicgstab_iters_per_newton: Vec<u32>,
    // Sub-phase accumulators (set by callers via start/stop helpers)
    pub device_eval: Duration,
    pub assembly: Duration,
    pub matrix_dl_ul: Duration,
    pub linear_solve: Duration,
    pub convergence_check: Duration,
    // GPU counters (read from backend at end)
    pub gpu_dispatches: u32,
    pub gpu_readbacks: u32,
    // Transient
    pub timesteps_accepted: u32,
    pub timesteps_rejected: u32,
    // Linear solves (AC/transient)
    pub linear_solves: u32,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl Stats {
    pub fn new() -> Self {
        Self {
            total_start: Instant::now(),
            phases: Vec::new(),
            newton_iterations: 0,
            bicgstab_iters_per_newton: Vec::new(),
            device_eval: Duration::ZERO,
            assembly: Duration::ZERO,
            matrix_dl_ul: Duration::ZERO,
            linear_solve: Duration::ZERO,
            convergence_check: Duration::ZERO,
            gpu_dispatches: 0,
            gpu_readbacks: 0,
            timesteps_accepted: 0,
            timesteps_rejected: 0,
            linear_solves: 0,
        }
    }

    /// Record a completed phase with its duration.
    pub fn add_phase(&mut self, name: &'static str, duration: Duration) {
        self.phases.push((name, duration));
    }

    /// Print the stats table to stderr.
    pub fn display(&self) {
        let total = self.total_start.elapsed();
        eprintln!();
        eprintln!("=== Ohmnivore Performance Stats ===");

        for (name, dur) in &self.phases {
            eprintln!("  {:<24} {:>8.3}s", name, dur.as_secs_f64());
        }

        if self.newton_iterations > 0 {
            eprintln!("  Newton iterations:      {}", self.newton_iterations);
            eprintln!("    Device eval:          {:>8.3}s", self.device_eval.as_secs_f64());
            eprintln!("    Assembly:             {:>8.3}s", self.assembly.as_secs_f64());
            eprintln!("    Matrix DL/UL:         {:>8.3}s", self.matrix_dl_ul.as_secs_f64());
            eprintln!("    Linear solve:         {:>8.3}s", self.linear_solve.as_secs_f64());
            if !self.bicgstab_iters_per_newton.is_empty() {
                let avg: f64 = self.bicgstab_iters_per_newton.iter().map(|&i| i as f64).sum::<f64>()
                    / self.bicgstab_iters_per_newton.len() as f64;
                let iters_str: Vec<String> = self.bicgstab_iters_per_newton.iter().map(|i| i.to_string()).collect();
                eprintln!("      BiCGSTAB iters:     {}", iters_str.join(" / "));
                eprintln!("      BiCGSTAB avg:       {:.1}", avg);
            }
            eprintln!("    Convergence check:    {:>8.3}s", self.convergence_check.as_secs_f64());
        }

        if self.timesteps_accepted > 0 || self.timesteps_rejected > 0 {
            eprintln!("  Timesteps:              accepted={}  rejected={}", self.timesteps_accepted, self.timesteps_rejected);
        }

        if self.linear_solves > 0 {
            eprintln!("  Linear solves:          {}", self.linear_solves);
        }

        eprintln!("  ─────────────────────────────────");
        eprintln!("  Total:                  {:>8.3}s", total.as_secs_f64());
        eprintln!("  GPU dispatches:         {}", self.gpu_dispatches);
        eprintln!("  GPU readbacks:          {}", self.gpu_readbacks);
    }
}
