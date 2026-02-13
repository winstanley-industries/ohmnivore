//! Compressed Sparse Row (CSR) matrix.
//!
//! Used to represent MNA conductance (G) and capacitance (C) matrices.
//! Generic over value type to support both f64 (DC) and Complex64 (AC).

use num_complex::Complex64;
use std::ops::AddAssign;

/// Sparse matrix in Compressed Sparse Row format.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    pub nrows: usize,
    pub ncols: usize,
    /// Non-zero values, stored row by row.
    pub values: Vec<T>,
    /// Column index for each non-zero value.
    pub col_indices: Vec<usize>,
    /// `row_pointers[i]` is the index into values/col_indices where row i starts.
    /// `row_pointers[nrows]` = total number of non-zeros.
    pub row_pointers: Vec<usize>,
}

impl<T: Copy + Default + AddAssign> CsrMatrix<T> {
    /// Create an empty matrix with no non-zero entries.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers: vec![0; nrows + 1],
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Build CSR matrix from (row, col, value) triplets.
    /// Duplicate entries at the same (row, col) are summed.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, T)]) -> Self {
        if triplets.is_empty() {
            return Self::new(nrows, ncols);
        }

        // Sort indices by (row, col) without requiring T: Ord
        let mut indices: Vec<usize> = (0..triplets.len()).collect();
        indices.sort_by_key(|&i| (triplets[i].0, triplets[i].1));

        let mut values = Vec::with_capacity(triplets.len());
        let mut col_indices = Vec::with_capacity(triplets.len());
        let mut row_pointers = vec![0usize; nrows + 1];

        let first = indices[0];
        let mut cur_row = triplets[first].0;
        let mut cur_col = triplets[first].1;
        let mut cur_val = triplets[first].2;

        for &idx in &indices[1..] {
            let (row, col, val) = triplets[idx];
            if row == cur_row && col == cur_col {
                cur_val += val;
            } else {
                values.push(cur_val);
                col_indices.push(cur_col);
                row_pointers[cur_row + 1] += 1;
                cur_row = row;
                cur_col = col;
                cur_val = val;
            }
        }
        // Emit last accumulated entry
        values.push(cur_val);
        col_indices.push(cur_col);
        row_pointers[cur_row + 1] += 1;

        // Convert per-row counts to cumulative offsets
        for i in 1..=nrows {
            row_pointers[i] += row_pointers[i - 1];
        }

        Self {
            nrows,
            ncols,
            values,
            col_indices,
            row_pointers,
        }
    }

    /// Convert to dense matrix (row-major). For testing and small matrices only.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::default(); self.ncols]; self.nrows];
        for row in 0..self.nrows {
            for idx in self.row_pointers[row]..self.row_pointers[row + 1] {
                dense[row][self.col_indices[idx]] = self.values[idx];
            }
        }
        dense
    }
}

impl CsrMatrix<f64> {
    /// Sparse matrix-vector multiply: y = A * x
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols, "spmv dimension mismatch");
        let mut y = vec![0.0; self.nrows];
        for row in 0..self.nrows {
            let mut sum = 0.0;
            for idx in self.row_pointers[row]..self.row_pointers[row + 1] {
                sum += self.values[idx] * x[self.col_indices[idx]];
            }
            y[row] = sum;
        }
        y
    }
}

impl CsrMatrix<Complex64> {
    /// Sparse matrix-vector multiply: y = A * x (complex)
    pub fn spmv(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.ncols, "spmv dimension mismatch");
        let mut y = vec![Complex64::new(0.0, 0.0); self.nrows];
        for row in 0..self.nrows {
            let mut sum = Complex64::new(0.0, 0.0);
            for idx in self.row_pointers[row]..self.row_pointers[row + 1] {
                sum += self.values[idx] * x[self.col_indices[idx]];
            }
            y[row] = sum;
        }
        y
    }
}

/// Combine real G and C matrices into complex A = G + jωC.
/// Both matrices must have the same dimensions.
/// Entries present in one but not the other are treated as zero.
pub fn form_complex_matrix(
    g: &CsrMatrix<f64>,
    c: &CsrMatrix<f64>,
    omega: f64,
) -> CsrMatrix<Complex64> {
    assert_eq!(g.nrows, c.nrows);
    assert_eq!(g.ncols, c.ncols);
    let n = g.nrows;

    // Build via triplets for simplicity — both matrices are small for MVP
    let mut triplets = Vec::new();

    // Add G entries as real parts
    for row in 0..n {
        for idx in g.row_pointers[row]..g.row_pointers[row + 1] {
            let col = g.col_indices[idx];
            triplets.push((row, col, Complex64::new(g.values[idx], 0.0)));
        }
    }

    // Add jωC entries as imaginary parts
    for row in 0..n {
        for idx in c.row_pointers[row]..c.row_pointers[row + 1] {
            let col = c.col_indices[idx];
            triplets.push((row, col, Complex64::new(0.0, omega * c.values[idx])));
        }
    }

    CsrMatrix::from_triplets(n, n, &triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let m: CsrMatrix<f64> = CsrMatrix::new(3, 3);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.row_pointers, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_from_triplets_simple() {
        // 2x2 identity matrix
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let m = CsrMatrix::from_triplets(2, 2, &triplets);
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.to_dense(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn test_from_triplets_duplicates_summed() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0)];
        let m = CsrMatrix::from_triplets(2, 2, &triplets);
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.to_dense(), vec![vec![3.0, 0.0], vec![0.0, 3.0]]);
    }

    #[test]
    fn test_spmv() {
        // [[2, 1], [0, 3]] * [1, 2] = [4, 6]
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)];
        let m = CsrMatrix::from_triplets(2, 2, &triplets);
        let y = m.spmv(&[1.0, 2.0]);
        assert_eq!(y, vec![4.0, 6.0]);
    }

    #[test]
    fn test_form_complex_matrix() {
        // G = [[1, 0], [0, 2]], C = [[0, 0], [0, 1]]
        // At omega=1: A = [[1+0j, 0], [0, 2+1j]]
        let g = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        let c = CsrMatrix::from_triplets(2, 2, &[(1, 1, 1.0)]);
        let a = form_complex_matrix(&g, &c, 1.0);
        let dense = a.to_dense();
        assert_eq!(dense[0][0], Complex64::new(1.0, 0.0));
        assert_eq!(dense[1][1], Complex64::new(2.0, 1.0));
    }
}
