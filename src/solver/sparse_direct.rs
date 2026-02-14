//! Sparse-direct CPU fallback based on sparse LU factorization.
//!
//! This path is intended for difficult Newton subproblems where iterative
//! methods break down and the dense CPU fallback reports singular too early.

use crate::error::{OhmnivoreError, Result};
use crate::sparse::CsrMatrix;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

/// Solve a real-valued linear system with sparse LU on CPU.
pub fn solve_real_sparse_lu(a: &CsrMatrix<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.nrows;
    if a.ncols != n || b.len() != n {
        return Err(OhmnivoreError::Solve(format!(
            "dimension mismatch: matrix is {}x{}, rhs length is {}",
            a.nrows,
            a.ncols,
            b.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut triplets = Vec::with_capacity(a.values.len());
    for row in 0..n {
        for idx in a.row_pointers[row]..a.row_pointers[row + 1] {
            let val = a.values[idx];
            if !val.is_finite() {
                return Err(OhmnivoreError::Solve(
                    "sparse LU input contains NaN/Inf".into(),
                ));
            }
            triplets.push(Triplet::new(row, a.col_indices[idx], val));
        }
    }

    let a_sp = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|e| OhmnivoreError::Solve(format!("sparse matrix build failed: {e:?}")))?;

    let lu = a_sp
        .sp_lu()
        .map_err(|e| OhmnivoreError::Solve(format!("sparse LU factorization failed: {e:?}")))?;

    let rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let x = lu.solve(rhs);

    let mut out = vec![0.0; n];
    for i in 0..n {
        let xi = x[(i, 0)];
        if !xi.is_finite() {
            return Err(OhmnivoreError::Solve(
                "sparse LU produced NaN/Inf solution".into(),
            ));
        }
        out[i] = xi;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_lu_solves_known_2x2() {
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 5.0), (1, 1, 7.0)],
        );
        let b = vec![11.0, 13.0];
        let x = solve_real_sparse_lu(&a, &b).expect("sparse LU should solve");
        assert!((x[0] - 64.0 / 9.0).abs() < 1e-10);
        assert!((x[1] + 29.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn sparse_lu_reports_singular() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)]);
        let b = vec![1.0, 1.0];
        let err = solve_real_sparse_lu(&a, &b).expect_err("matrix should be singular");
        assert!(format!("{err}").contains("Solve error"));
    }
}

