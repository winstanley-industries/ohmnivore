use thiserror::Error;

#[derive(Debug, Error)]
pub enum OhmnivoreError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Compile error: {0}")]
    Compile(String),

    #[error("Solve error: {0}")]
    Solve(String),

    #[error("Newton solver did not converge after {iterations} iterations (max residual: {max_residual:.2e})")]
    NewtonNotConverged {
        iterations: usize,
        max_residual: f64,
    },

    #[error("Newton solver encountered numerical error at iteration {iteration}")]
    NewtonNumericalError { iteration: usize },

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OhmnivoreError>;
