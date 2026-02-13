//! Transient source function evaluation.
//!
//! Evaluates PULSE, SIN, PWL, and EXP waveforms at a given time t.

use crate::ir::TransientFunc;

/// Evaluate a transient source function at time t.
pub fn evaluate(func: &TransientFunc, t: f64) -> f64 {
    match func {
        TransientFunc::Pulse {
            v1,
            v2,
            td,
            tr,
            tf,
            pw,
            per,
        } => eval_pulse(*v1, *v2, *td, *tr, *tf, *pw, *per, t),
        TransientFunc::Sin {
            vo,
            va,
            freq,
            td,
            theta,
        } => eval_sin(*vo, *va, *freq, *td, *theta, t),
        TransientFunc::Pwl { pairs } => eval_pwl(pairs, t),
        TransientFunc::Exp {
            v1,
            v2,
            td1,
            tau1,
            td2,
            tau2,
        } => eval_exp(*v1, *v2, *td1, *tau1, *td2, *tau2, t),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_pulse(v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64, t: f64) -> f64 {
    if t < td {
        return v1;
    }
    // Time within the current period
    let t_rel = if per > 0.0 && per < f64::MAX {
        (t - td) % per
    } else {
        t - td
    };

    if t_rel < tr {
        // Rising edge
        if tr > 0.0 {
            v1 + (v2 - v1) * t_rel / tr
        } else {
            v2
        }
    } else if t_rel < tr + pw {
        // Pulse high
        v2
    } else if t_rel < tr + pw + tf {
        // Falling edge
        if tf > 0.0 {
            v2 + (v1 - v2) * (t_rel - tr - pw) / tf
        } else {
            v1
        }
    } else {
        // Between pulses
        v1
    }
}

fn eval_sin(vo: f64, va: f64, freq: f64, td: f64, theta: f64, t: f64) -> f64 {
    if t < td {
        return vo;
    }
    let dt = t - td;
    let envelope = if theta != 0.0 {
        (-dt * theta).exp()
    } else {
        1.0
    };
    vo + va * (2.0 * std::f64::consts::PI * freq * dt).sin() * envelope
}

fn eval_pwl(pairs: &[(f64, f64)], t: f64) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    if t <= pairs[0].0 {
        return pairs[0].1;
    }
    if t >= pairs[pairs.len() - 1].0 {
        return pairs[pairs.len() - 1].1;
    }
    // Find the interval containing t
    for i in 1..pairs.len() {
        if t <= pairs[i].0 {
            let (t0, v0) = pairs[i - 1];
            let (t1, v1) = pairs[i];
            let frac = (t - t0) / (t1 - t0);
            return v0 + (v1 - v0) * frac;
        }
    }
    pairs[pairs.len() - 1].1
}

fn eval_exp(v1: f64, v2: f64, td1: f64, tau1: f64, td2: f64, tau2: f64, t: f64) -> f64 {
    if t < td1 {
        return v1;
    }
    let rise = (v2 - v1) * (1.0 - (-(t - td1) / tau1).exp());
    if t < td2 {
        return v1 + rise;
    }
    let fall = (v1 - v2) * (1.0 - (-(t - td2) / tau2).exp());
    v1 + rise + fall
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PULSE tests ----

    #[test]
    fn test_pulse_before_delay() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 1.0,
            tr: 0.1,
            tf: 0.1,
            pw: 1.0,
            per: 3.0,
        };
        assert!((evaluate(&f, 0.5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pulse_during_rise() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 0.0,
            tr: 1.0,
            tf: 1.0,
            pw: 2.0,
            per: 6.0,
        };
        assert!((evaluate(&f, 0.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_pulse_at_peak() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 0.0,
            tr: 1.0,
            tf: 1.0,
            pw: 2.0,
            per: 6.0,
        };
        assert!((evaluate(&f, 2.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_pulse_during_fall() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 0.0,
            tr: 1.0,
            tf: 1.0,
            pw: 2.0,
            per: 6.0,
        };
        assert!((evaluate(&f, 3.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_pulse_between_pulses() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 0.0,
            tr: 1.0,
            tf: 1.0,
            pw: 2.0,
            per: 6.0,
        };
        assert!((evaluate(&f, 5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pulse_period_wrap() {
        let f = TransientFunc::Pulse {
            v1: 0.0,
            v2: 5.0,
            td: 0.0,
            tr: 0.0,
            tf: 0.0,
            pw: 1.0,
            per: 2.0,
        };
        // t=2.5: period wraps to t_rel=0.5, in pulse width
        assert!((evaluate(&f, 2.5) - 5.0).abs() < 1e-12);
        // t=3.5: period wraps to t_rel=1.5, past pulse width
        assert!((evaluate(&f, 3.5) - 0.0).abs() < 1e-12);
    }

    // ---- SIN tests ----

    #[test]
    fn test_sin_before_delay() {
        let f = TransientFunc::Sin {
            vo: 1.0,
            va: 2.0,
            freq: 1.0,
            td: 1.0,
            theta: 0.0,
        };
        assert!((evaluate(&f, 0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sin_at_quarter_period() {
        let f = TransientFunc::Sin {
            vo: 0.0,
            va: 1.0,
            freq: 1.0,
            td: 0.0,
            theta: 0.0,
        };
        // sin(2*pi*1.0*0.25) = sin(pi/2) = 1.0
        assert!((evaluate(&f, 0.25) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sin_at_half_period() {
        let f = TransientFunc::Sin {
            vo: 0.0,
            va: 1.0,
            freq: 1.0,
            td: 0.0,
            theta: 0.0,
        };
        // sin(pi) ~ 0
        assert!((evaluate(&f, 0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_sin_with_damping() {
        let f = TransientFunc::Sin {
            vo: 0.0,
            va: 1.0,
            freq: 1.0,
            td: 0.0,
            theta: 1.0,
        };
        // At t=0.25: sin(pi/2)*exp(-0.25) = 1.0*exp(-0.25)
        let expected = (-0.25_f64).exp();
        assert!((evaluate(&f, 0.25) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sin_with_offset() {
        let f = TransientFunc::Sin {
            vo: 2.5,
            va: 1.0,
            freq: 1.0,
            td: 0.0,
            theta: 0.0,
        };
        assert!((evaluate(&f, 0.25) - 3.5).abs() < 1e-12);
    }

    // ---- PWL tests ----

    #[test]
    fn test_pwl_before_first() {
        let f = TransientFunc::Pwl {
            pairs: vec![(1.0, 0.0), (2.0, 5.0)],
        };
        assert!((evaluate(&f, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pwl_after_last() {
        let f = TransientFunc::Pwl {
            pairs: vec![(1.0, 0.0), (2.0, 5.0)],
        };
        assert!((evaluate(&f, 3.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_pwl_interpolation() {
        let f = TransientFunc::Pwl {
            pairs: vec![(0.0, 0.0), (1.0, 10.0), (2.0, 10.0), (3.0, 0.0)],
        };
        assert!((evaluate(&f, 0.5) - 5.0).abs() < 1e-12);
        assert!((evaluate(&f, 1.5) - 10.0).abs() < 1e-12);
        assert!((evaluate(&f, 2.5) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_pwl_at_breakpoint() {
        let f = TransientFunc::Pwl {
            pairs: vec![(0.0, 0.0), (1.0, 5.0), (2.0, 3.0)],
        };
        assert!((evaluate(&f, 1.0) - 5.0).abs() < 1e-12);
    }

    // ---- EXP tests ----

    #[test]
    fn test_exp_before_td1() {
        let f = TransientFunc::Exp {
            v1: 0.0,
            v2: 5.0,
            td1: 1.0,
            tau1: 1.0,
            td2: 3.0,
            tau2: 1.0,
        };
        assert!((evaluate(&f, 0.5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_exp_during_rise() {
        let f = TransientFunc::Exp {
            v1: 0.0,
            v2: 5.0,
            td1: 0.0,
            tau1: 1.0,
            td2: 10.0,
            tau2: 1.0,
        };
        // At t=1: 5*(1 - exp(-1)) = 5*0.6321 = 3.1606
        let expected = 5.0 * (1.0 - (-1.0_f64).exp());
        assert!((evaluate(&f, 1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_exp_during_fall() {
        let f = TransientFunc::Exp {
            v1: 0.0,
            v2: 5.0,
            td1: 0.0,
            tau1: 1.0,
            td2: 5.0,
            tau2: 1.0,
        };
        // At t=6: rise + fall
        let rise = 5.0 * (1.0 - (-6.0_f64).exp());
        let fall = -5.0 * (1.0 - (-1.0_f64).exp());
        let expected = rise + fall;
        assert!((evaluate(&f, 6.0) - expected).abs() < 1e-10);
    }
}
