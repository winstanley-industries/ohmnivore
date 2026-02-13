use thiserror::Error;

#[derive(Debug, Error)]
pub enum OhmnivoreError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Compile error: {0}")]
    Compile(String),

    #[error("Solve error: {0}")]
    Solve(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OhmnivoreError>;
