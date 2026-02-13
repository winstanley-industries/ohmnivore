//! CPU direct solver using Gaussian elimination with partial pivoting.
//!
//! Converts CSR to dense and solves via LU decomposition.
//! Suitable for small-to-medium circuits. The GPU iterative solver
//! handles large circuits.

use crate::error::{OhmnivoreError, Result};
use crate::sparse::CsrMatrix;
use num_complex::Complex64;

/// CPU-based direct linear solver.
pub struct CpuSolver;

impl CpuSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl super::LinearSolver for CpuSolver {
    fn solve_real(&self, a: &CsrMatrix<f64>, b: &[f64]) -> Result<Vec<f64>> {
        let n = a.nrows;
        if a.ncols != n || b.len() != n {
            return Err(OhmnivoreError::Solve(format!(
                "dimension mismatch: matrix is {}x{}, rhs length is {}",
                a.nrows, a.ncols, b.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        // Build augmented matrix [A | b]
        let dense = a.to_dense();
        let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n + 1);
            row.extend_from_slice(&dense[i]);
            row.push(b[i]);
            aug.push(row);
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot row
            let mut max_val = aug[k][k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = aug[i][k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            if max_val < 1e-15 {
                return Err(OhmnivoreError::Solve("singular matrix".into()));
            }

            // Swap rows
            if max_row != k {
                aug.swap(k, max_row);
            }

            // Eliminate below
            let pivot = aug[k][k];
            for i in (k + 1)..n {
                let factor = aug[i][k] / pivot;
                aug[i][k] = 0.0;
                for j in (k + 1)..=n {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }

        Ok(x)
    }

    fn solve_complex(
        &self,
        a: &CsrMatrix<Complex64>,
        b: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        let n = a.nrows;
        if a.ncols != n || b.len() != n {
            return Err(OhmnivoreError::Solve(format!(
                "dimension mismatch: matrix is {}x{}, rhs length is {}",
                a.nrows, a.ncols, b.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let zero = Complex64::new(0.0, 0.0);

        // Build augmented matrix [A | b]
        let dense = a.to_dense();
        let mut aug: Vec<Vec<Complex64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n + 1);
            row.extend_from_slice(&dense[i]);
            row.push(b[i]);
            aug.push(row);
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot row (by magnitude)
            let mut max_val = aug[k][k].norm();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = aug[i][k].norm();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            if max_val < 1e-15 {
                return Err(OhmnivoreError::Solve("singular matrix".into()));
            }

            // Swap rows
            if max_row != k {
                aug.swap(k, max_row);
            }

            // Eliminate below
            let pivot = aug[k][k];
            for i in (k + 1)..n {
                let factor = aug[i][k] / pivot;
                aug[i][k] = zero;
                for j in (k + 1)..=n {
                    let akj = aug[k][j];
                    aug[i][j] -= factor * akj;
                }
            }
        }

        // Back substitution
        let mut x = vec![zero; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::LinearSolver;
    use crate::sparse::CsrMatrix;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex64;

    fn solver() -> CpuSolver {
        CpuSolver::new()
    }

    // ── Real solver tests ──

    #[test]
    fn solve_real_identity_2x2() {
        // I * x = [3, 7] => x = [3, 7]
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![3.0, 7.0];
        let x = solver().solve_real(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_real_known_2x2() {
        // [[2, 1], [5, 7]] * x = [11, 13]
        // det = 14 - 5 = 9
        // x1 = (77 - 13)/9 = 64/9, x2 = (26 - 55)/9 = -29/9
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 5.0), (1, 1, 7.0)],
        );
        let b = vec![11.0, 13.0];
        let x = solver().solve_real(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 64.0 / 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], -29.0 / 9.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_real_3x3_needs_pivoting() {
        // [[0, 2, 1], [1, 1, 1], [2, 1, 0]] * x = [5, 6, 5]
        // Row 0 has zero in pivot position, so pivoting is required.
        // Solution: x = [1, 2, 1]
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 1, 2.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
                (2, 0, 2.0),
                (2, 1, 1.0),
            ],
        );
        let b = vec![5.0, 4.0, 4.0];
        let x = solver().solve_real(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_real_singular_matrix() {
        // [[1, 2], [2, 4]] is singular (row2 = 2*row1)
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)],
        );
        let b = vec![3.0, 6.0];
        let result = solver().solve_real(&a, &b);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("singular"), "expected singular error, got: {err}");
    }

    #[test]
    fn solve_real_dimension_mismatch() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![1.0, 2.0, 3.0]; // wrong length
        let result = solver().solve_real(&a, &b);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dimension"),
            "expected dimension error, got: {err}"
        );
    }

    #[test]
    fn solve_real_nonsquare_matrix() {
        let a = CsrMatrix::from_triplets(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![1.0, 2.0];
        let result = solver().solve_real(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn solve_real_1x1() {
        let a = CsrMatrix::from_triplets(1, 1, &[(0, 0, 5.0)]);
        let b = vec![15.0];
        let x = solver().solve_real(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_real_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0);
        let b: Vec<f64> = vec![];
        let x = solver().solve_real(&a, &b).unwrap();
        assert!(x.is_empty());
    }

    // ── Complex solver tests ──

    #[test]
    fn solve_complex_identity_2x2() {
        let one = Complex64::new(1.0, 0.0);
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, one), (1, 1, one)]);
        let b = vec![Complex64::new(3.0, 1.0), Complex64::new(7.0, -2.0)];
        let x = solver().solve_complex(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0].re, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[0].im, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].re, 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].im, -2.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_complex_known_2x2() {
        // [[1+j, 2], [0, 1-j]] * x = [5+j, 2-2j]
        // From row 2: x2 = (2-2j)/(1-j) = (2-2j)(1+j)/((1-j)(1+j)) = (2-2j)(1+j)/2
        //   = (2+2j-2j-2j^2)/2 = (2+2)/2 = 2
        // From row 1: (1+j)*x1 + 2*2 = 5+j => (1+j)*x1 = 1+j => x1 = 1
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 0, Complex64::new(1.0, 1.0)),
                (0, 1, Complex64::new(2.0, 0.0)),
                (1, 1, Complex64::new(1.0, -1.0)),
            ],
        );
        let b = vec![Complex64::new(5.0, 1.0), Complex64::new(2.0, -2.0)];
        let x = solver().solve_complex(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[0].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].re, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_complex_singular_matrix() {
        // [[1+j, 2+2j], [1, 2]] — row2 is (1/(1+j)) * row1 in terms of column ratios
        // Actually: col ratio row1: (2+2j)/(1+j) = 2, col ratio row2: 2/1 = 2 → singular
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 0, Complex64::new(1.0, 1.0)),
                (0, 1, Complex64::new(2.0, 2.0)),
                (1, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(2.0, 0.0)),
            ],
        );
        let b = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let result = solver().solve_complex(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn solve_complex_dimension_mismatch() {
        let one = Complex64::new(1.0, 0.0);
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, one), (1, 1, one)]);
        let b = vec![one]; // wrong length
        let result = solver().solve_complex(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn solve_complex_needs_pivoting() {
        // [[0, 1+j], [1, 2]] * x = [1+j, 3]
        // After swap: [[1, 2], [0, 1+j]] * x = [3, 1+j]
        // x2 = (1+j)/(1+j) = 1
        // x1 = 3 - 2*1 = 1
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 1, Complex64::new(1.0, 1.0)),
                (1, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(2.0, 0.0)),
            ],
        );
        let b = vec![Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)];
        let x = solver().solve_complex(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[0].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_real_verifies_with_spmv() {
        // Solve Ax = b, then verify A*x ≈ b
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        );
        let b = vec![1.0, 5.0, 10.0];
        let x = solver().solve_real(&a, &b).unwrap();
        let ax = a.spmv(&x);
        for i in 0..3 {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }
}
