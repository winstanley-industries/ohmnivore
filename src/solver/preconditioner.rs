//! ISAI (Incomplete Sparse Approximate Inverse) preconditioner.
//!
//! Computes approximate inverses M_L ≈ L⁻¹ and M_U ≈ U⁻¹ from an ILU(0)
//! factorization, where each column of M_L and M_U is computed independently
//! via a small local least-squares solve restricted to a chosen sparsity pattern.

use crate::sparse::CsrMatrix;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Trait bounds needed for ISAI arithmetic.
pub trait IsaiScalar:
    Copy
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + std::ops::AddAssign
    + std::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs_val(self) -> f64;
}

impl IsaiScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn abs_val(self) -> f64 {
        self.abs()
    }
}

/// Result of ISAI preconditioner computation.
#[derive(Debug, Clone)]
pub struct IsaiPreconditionerData<T> {
    /// Approximate inverse of L (lower triangular).
    pub m_l: CsrMatrix<T>,
    /// Approximate inverse of U (upper triangular).
    pub m_u: CsrMatrix<T>,
    /// ISAI level used (0 or 1).
    pub level: usize,
}

/// Perform sparse LU factorization with partial pivoting.
///
/// Returns (L, U, perm) where L is unit lower triangular, U is upper
/// triangular, and `perm` tracks the row permutation such that PA = LU
/// (perm[i] = original row index of the i-th row after pivoting).
///
/// Fill-in is allowed so that the factors are non-singular even for MNA
/// matrices with zero diagonals (voltage sources, inductors). For the small
/// matrices typical of circuit simulation this has negligible CPU cost; the
/// ISAI benefit lies in the GPU apply (SpMV-only), not in restricting fill.
pub fn ilu0<T: IsaiScalar>(a: &CsrMatrix<T>) -> (CsrMatrix<T>, CsrMatrix<T>, Vec<usize>) {
    assert_eq!(a.nrows, a.ncols, "LU requires a square matrix");
    let n = a.nrows;

    // Track row permutation: perm[i] = original row index at position i.
    let mut perm: Vec<usize> = (0..n).collect();

    // Build a row-indexed working structure.
    let mut rows: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
    for i in 0..n {
        let start = a.row_pointers[i];
        let end = a.row_pointers[i + 1];
        let row: Vec<(usize, T)> = (start..end)
            .map(|idx| (a.col_indices[idx], a.values[idx]))
            .collect();
        rows.push(row);
    }

    // Column-major LU with partial pivoting and fill-in.
    // We need index-based access because rows[i] is mutated while rows[j] is read.
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        // Partial pivoting: find row with largest |entry| in column j.
        let mut best_row = j;
        let mut best_val = find_val(&rows[j], j).abs_val();
        for i in (j + 1)..n {
            let v = find_val(&rows[i], j).abs_val();
            if v > best_val {
                best_val = v;
                best_row = i;
            }
        }
        if best_row != j {
            rows.swap(j, best_row);
            perm.swap(j, best_row);
        }

        let pivot = find_val(&rows[j], j);
        if pivot.abs_val() < 1e-30 {
            continue;
        }

        let pivot_upper: Vec<(usize, T)> = rows[j]
            .iter()
            .filter(|&&(c, _)| c > j)
            .copied()
            .collect();

        for i in (j + 1)..n {
            let a_ij = find_val(&rows[i], j);
            if a_ij.abs_val() < 1e-30 {
                continue;
            }
            let l_ij = a_ij / pivot;

            set_or_insert(&mut rows[i], j, l_ij);

            for &(k, u_jk) in &pivot_upper {
                let fill = l_ij * u_jk;
                if let Some(pos) = rows[i].iter().position(|&(c, _)| c == k) {
                    rows[i][pos].1 = rows[i][pos].1 - fill;
                } else {
                    let insert_pos = rows[i].partition_point(|&(c, _)| c < k);
                    rows[i].insert(insert_pos, (k, T::zero() - fill));
                }
            }
        }
    }

    // Extract L and U.
    let mut l_triplets: Vec<(usize, usize, T)> = Vec::new();
    let mut u_triplets: Vec<(usize, usize, T)> = Vec::new();

    for (i, row) in rows.iter().enumerate() {
        for &(j, val) in row {
            if j < i {
                l_triplets.push((i, j, val));
            }
        }
        l_triplets.push((i, i, T::one()));

        for &(j, val) in row {
            if j >= i {
                u_triplets.push((i, j, val));
            }
        }
    }

    let l = CsrMatrix::from_triplets(n, n, &l_triplets);
    let u = CsrMatrix::from_triplets(n, n, &u_triplets);

    (l, u, perm)
}

/// Set a value at column `col` in a sorted row, or insert it if not present.
fn set_or_insert<T: IsaiScalar>(row: &mut Vec<(usize, T)>, col: usize, val: T) {
    if let Some(entry) = row.iter_mut().find(|&&mut (c, _)| c == col) {
        entry.1 = val;
    } else {
        let pos = row.partition_point(|&(c, _)| c < col);
        row.insert(pos, (col, val));
    }
}

/// Generate the sparsity pattern of a CSR matrix as sets of column indices per row.
fn sparsity_pattern<T: Copy + Default>(m: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let mut pattern = Vec::with_capacity(m.nrows);
    for i in 0..m.nrows {
        let start = m.row_pointers[i];
        let end = m.row_pointers[i + 1];
        let cols: Vec<usize> = m.col_indices[start..end].to_vec();
        pattern.push(cols);
    }
    pattern
}

/// Compute the symbolic sparsity pattern of A * B (no arithmetic, just structure).
fn symbolic_multiply_pattern<T: Copy + Default>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> Vec<Vec<usize>> {
    let a_pat = sparsity_pattern(a);
    let b_pat = sparsity_pattern(b);
    let n = a.nrows;

    let mut result = Vec::with_capacity(n);
    for a_row in &a_pat {
        let mut col_set = Vec::new();
        for &k in a_row {
            for &j in &b_pat[k] {
                col_set.push(j);
            }
        }
        col_set.sort_unstable();
        col_set.dedup();
        result.push(col_set);
    }
    result
}

/// Generate sparsity patterns for M_L and M_U at a given ISAI level.
///
/// - level=0: pattern(M_L) = pattern(L), pattern(M_U) = pattern(U)
/// - level=1: pattern(M_L) = pattern(L^2), pattern(M_U) = pattern(U^2)
fn generate_patterns<T: IsaiScalar + std::ops::AddAssign>(
    l: &CsrMatrix<T>,
    u: &CsrMatrix<T>,
    level: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    match level {
        0 => (sparsity_pattern(l), sparsity_pattern(u)),
        1 => (
            symbolic_multiply_pattern(l, l),
            symbolic_multiply_pattern(u, u),
        ),
        _ => panic!("ISAI level must be 0 or 1"),
    }
}

/// Transpose a CSR matrix (needed to work column-wise).
fn transpose<T: IsaiScalar + std::ops::AddAssign>(m: &CsrMatrix<T>) -> CsrMatrix<T> {
    let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(m.nnz());
    for i in 0..m.nrows {
        for idx in m.row_pointers[i]..m.row_pointers[i + 1] {
            triplets.push((m.col_indices[idx], i, m.values[idx]));
        }
    }
    CsrMatrix::from_triplets(m.ncols, m.nrows, &triplets)
}

/// Compute the ISAI approximate inverse preconditioner.
///
/// Given matrix A, performs LU factorization with partial pivoting (PA = LU),
/// then finds sparse approximate inverses M_L ≈ L⁻¹ and M_U ≈ U⁻¹ by solving
/// independent local least-squares problems for each column.
///
/// The row permutation P is folded into M_L by remapping its columns, so that
/// the preconditioner apply is simply: z = M_U * (M_L * r) ≈ A⁻¹ * r.
pub fn compute_isai<T: IsaiScalar + std::ops::AddAssign>(
    a: &CsrMatrix<T>,
    level: usize,
) -> IsaiPreconditionerData<T> {
    let (l, u, perm) = ilu0(a);
    let n = a.nrows;
    let (ml_pattern, mu_pattern) = generate_patterns(&l, &u, level);

    // Compute M_L: solve for each column k of M_L such that L * m_k ≈ e_k
    let l_t = transpose(&l);
    let ml_col_pattern = transpose_pattern(&ml_pattern, n);

    let ml_columns: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|k| solve_column_isai(&l, &l_t, k, &ml_col_pattern[k]))
        .collect();

    let m_l_raw = columns_to_csr(n, n, &ml_columns);

    // Fold the row permutation into M_L: M_L' = M_L * P.
    // Since PA = LU, we have A⁻¹ = U⁻¹ L⁻¹ P, so the apply becomes:
    //   z = M_U * (M_L' * r) = M_U * M_L * P * r ≈ U⁻¹ L⁻¹ P r = A⁻¹ r.
    // M_L * P permutes columns: column k of M_L' comes from column perm⁻¹[k].
    // Equivalently, entry at column c in M_L maps to column perm[c] in M_L'.
    let m_l = permute_csr_columns(&m_l_raw, &perm);

    // Compute M_U: solve for each column k of M_U such that U * m_k ≈ e_k
    let u_t = transpose(&u);
    let mu_col_pattern = transpose_pattern(&mu_pattern, n);

    let mu_columns: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|k| solve_column_isai(&u, &u_t, k, &mu_col_pattern[k]))
        .collect();

    let m_u = columns_to_csr(n, n, &mu_columns);

    IsaiPreconditionerData { m_l, m_u, level }
}

/// Remap column indices of a CSR matrix: new_col = perm[old_col].
fn permute_csr_columns<T: IsaiScalar + std::ops::AddAssign>(
    m: &CsrMatrix<T>,
    perm: &[usize],
) -> CsrMatrix<T> {
    let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(m.values.len());
    for i in 0..m.nrows {
        for idx in m.row_pointers[i]..m.row_pointers[i + 1] {
            triplets.push((i, perm[m.col_indices[idx]], m.values[idx]));
        }
    }
    CsrMatrix::from_triplets(m.nrows, m.ncols, &triplets)
}

/// Transpose a row-based pattern to column-based.
/// Given pattern[row] = [cols...], returns col_pattern[col] = [rows...].
fn transpose_pattern(pattern: &[Vec<usize>], ncols: usize) -> Vec<Vec<usize>> {
    let mut col_pattern: Vec<Vec<usize>> = vec![Vec::new(); ncols];
    for (row, cols) in pattern.iter().enumerate() {
        for &col in cols {
            col_pattern[col].push(row);
        }
    }
    for col in &mut col_pattern {
        col.sort_unstable();
    }
    col_pattern
}

/// Solve the local least-squares problem for column k:
///   minimize ‖A * m_k - e_k‖  subject to m_k having support on `col_indices`
///
/// `a` is the triangular factor (L or U), `a_t` is its transpose.
/// `col_indices` are the row indices where column k of M is nonzero.
///
/// This extracts the submatrix A[col_indices, col_indices] and solves
/// A_sub * m_sub = e_sub where e_sub is the unit vector at position k.
fn solve_column_isai<T: IsaiScalar>(
    a: &CsrMatrix<T>,
    _a_t: &CsrMatrix<T>,
    k: usize,
    col_indices: &[usize],
) -> Vec<(usize, T)> {
    let nz = col_indices.len();
    if nz == 0 {
        return Vec::new();
    }

    // Build a map from global index to local index
    let mut global_to_local = vec![usize::MAX; a.nrows];
    for (local, &global) in col_indices.iter().enumerate() {
        global_to_local[global] = local;
    }

    // Extract the dense submatrix A[col_indices, col_indices]
    let mut sub = vec![T::zero(); nz * nz];
    for (li, &row) in col_indices.iter().enumerate() {
        let start = a.row_pointers[row];
        let end = a.row_pointers[row + 1];
        for idx in start..end {
            let col = a.col_indices[idx];
            if col < global_to_local.len() && global_to_local[col] != usize::MAX {
                let lj = global_to_local[col];
                sub[li * nz + lj] = a.values[idx];
            }
        }
    }

    // Build rhs: e_k restricted to col_indices
    let mut rhs = vec![T::zero(); nz];
    if k < global_to_local.len() && global_to_local[k] != usize::MAX {
        rhs[global_to_local[k]] = T::one();
    }

    // Solve the small dense system via Gaussian elimination with partial pivoting
    let solution = dense_solve(nz, &mut sub, &mut rhs);

    // Restore global-to-local map
    for &global in col_indices {
        global_to_local[global] = usize::MAX;
    }

    // Pack results
    col_indices
        .iter()
        .zip(solution.iter())
        .map(|(&row, &val)| (row, val))
        .collect()
}

/// Solve a small dense system Ax = b via Gaussian elimination with partial pivoting.
/// Operates in-place on the provided arrays. `a` is row-major nxn, `b` is length n.
fn dense_solve<T: IsaiScalar>(n: usize, a: &mut [T], b: &mut [T]) -> Vec<T> {
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut best_row = col;
        let mut best_val = a[col * n + col].abs_val();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs_val();
            if v > best_val {
                best_val = v;
                best_row = row;
            }
        }

        if best_val < 1e-30 {
            // Singular or near-singular: leave zeros
            continue;
        }

        // Swap rows
        if best_row != col {
            for j in 0..n {
                a.swap(col * n + j, best_row * n + j);
            }
            b.swap(col, best_row);
        }

        let pivot = a[col * n + col];

        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            a[row * n + col] = T::zero();
            for j in (col + 1)..n {
                let above = a[col * n + j];
                a[row * n + j] = a[row * n + j] - factor * above;
            }
            b[row] = b[row] - factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for col in (0..n).rev() {
        let diag = a[col * n + col];
        if diag.abs_val() < 1e-30 {
            x[col] = T::zero();
            continue;
        }
        let mut sum = b[col];
        for j in (col + 1)..n {
            sum = sum - a[col * n + j] * x[j];
        }
        x[col] = sum / diag;
    }

    x
}

/// Convert column-wise entries into a CSR matrix.
fn columns_to_csr<T: IsaiScalar + std::ops::AddAssign>(
    nrows: usize,
    ncols: usize,
    columns: &[Vec<(usize, T)>],
) -> CsrMatrix<T> {
    let mut triplets: Vec<(usize, usize, T)> = Vec::new();
    for (col, entries) in columns.iter().enumerate() {
        for &(row, val) in entries {
            triplets.push((row, col, val));
        }
    }
    CsrMatrix::from_triplets(nrows, ncols, &triplets)
}

/// Helper: find value at a given column in a sorted (col, val) row.
fn find_val<T: IsaiScalar>(row: &[(usize, T)], col: usize) -> T {
    row.iter()
        .find(|&&(c, _)| c == col)
        .map(|&(_, v)| v)
        .unwrap_or_else(T::zero)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: multiply two small dense matrices (row-major).
    fn dense_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();
        let p = b.len();
        let mut c = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                for k in 0..p {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_ilu0_small_matrix() {
        // A = [[4, -1, 0],
        //      [-1, 4, -1],
        //      [0, -1, 4]]
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let (l, u, perm) = ilu0(&a);

        let l_dense = l.to_dense();
        let u_dense = u.to_dense();
        let lu = dense_matmul(&l_dense, &u_dense);

        // L*U should match P*A (permuted A)
        let a_dense = a.to_dense();
        let mut pa_dense = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            pa_dense[i] = a_dense[perm[i]].clone();
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(lu[i][j], pa_dense[i][j], 1e-10),
                    "L*U mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    lu[i][j],
                    pa_dense[i][j]
                );
            }
        }

        // L should have 1s on diagonal
        for i in 0..3 {
            assert!(
                approx_eq(l_dense[i][i], 1.0, 1e-14),
                "L diagonal at {} should be 1.0, got {}",
                i,
                l_dense[i][i]
            );
        }
    }

    #[test]
    fn test_isai0_exact_for_diagonal() {
        // Diagonal matrix: ISAI(0) should give exact inverse
        let triplets = vec![(0, 0, 2.0), (1, 1, 4.0), (2, 2, 5.0)];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let data = compute_isai(&a, 0);

        let (l, u, _perm) = ilu0(&a);

        // M_L * L should be exactly I (L is identity for diagonal A)
        let ml_dense = data.m_l.to_dense();
        let l_dense = l.to_dense();
        let product = dense_matmul(&ml_dense, &l_dense);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(product[i][j], expected, 1e-10),
                    "M_L*L not identity at ({},{}): got {}, expected {}",
                    i, j, product[i][j], expected,
                );
            }
        }

        // M_U * U should also be approximately I
        let mu_dense = data.m_u.to_dense();
        let u_dense = u.to_dense();
        let product_u = dense_matmul(&mu_dense, &u_dense);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(product_u[i][j], expected, 1e-10),
                    "M_U*U not identity at ({},{}): got {}, expected {}",
                    i, j, product_u[i][j], expected,
                );
            }
        }
    }

    #[test]
    fn test_isai1_tridiagonal_exact() {
        // Tridiagonal matrix: ISAI(1) with L^2 pattern should capture all fill-in
        // and give exact inverse for the 3x3 case.
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let data = compute_isai(&a, 1);

        let (l, _u, _perm) = ilu0(&a);

        // M_L * L should be approximately I with ISAI(1)
        let ml_dense = data.m_l.to_dense();
        let l_dense = l.to_dense();
        let product = dense_matmul(&ml_dense, &l_dense);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(product[i][j], expected, 1e-10),
                    "M_L*L not identity at ({},{}): got {}, expected {}",
                    i, j, product[i][j], expected,
                );
            }
        }
    }

    #[test]
    fn test_sparsity_pattern_isai0() {
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let (l, u, _perm) = ilu0(&a);

        let (ml_pat, mu_pat) = generate_patterns(&l, &u, 0);
        let l_pat = sparsity_pattern(&l);
        let u_pat = sparsity_pattern(&u);

        // ISAI(0): patterns should match L and U
        assert_eq!(ml_pat, l_pat, "M_L pattern should match L pattern at level 0");
        assert_eq!(mu_pat, u_pat, "M_U pattern should match U pattern at level 0");
    }

    #[test]
    fn test_sparsity_pattern_isai1() {
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let (l, u, _perm) = ilu0(&a);

        let (ml_pat, mu_pat) = generate_patterns(&l, &u, 1);

        // ISAI(1): patterns should be L^2 and U^2
        let l2_pat = symbolic_multiply_pattern(&l, &l);
        let u2_pat = symbolic_multiply_pattern(&u, &u);

        assert_eq!(ml_pat, l2_pat, "M_L pattern should match L^2 pattern at level 1");
        assert_eq!(mu_pat, u2_pat, "M_U pattern should match U^2 pattern at level 1");
    }

    #[test]
    fn test_zero_diagonal_mna_matrix() {
        // MNA matrix from voltage divider: has zero diagonal entries
        // [G1+G2, -G2, 1]   nodes: n1, n2, i_V1
        // [-G2,   G2,  0]
        // [1,     0,   0]   <- voltage source row, zero diagonal
        let g1 = 0.001; // 1kΩ
        let g2 = 0.002; // 500Ω
        let triplets = vec![
            (0, 0, g1 + g2),
            (0, 1, -g2),
            (0, 2, 1.0),
            (1, 0, -g2),
            (1, 1, g2),
            (2, 0, 1.0),
            // (2,2) is intentionally missing — zero diagonal
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);

        // Should not panic or produce NaN/Inf
        let data = compute_isai(&a, 0);

        // Verify no NaN or Inf in M_L
        for val in &data.m_l.values {
            assert!(val.is_finite(), "M_L contains non-finite value: {}", val);
        }

        // Verify no NaN or Inf in M_U
        for val in &data.m_u.values {
            assert!(val.is_finite(), "M_U contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_isai_level1_better_than_level0() {
        // 4x4 matrix where ISAI(1) should give a better approximate inverse
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 4.0),
        ];
        let a = CsrMatrix::from_triplets(4, 4, &triplets);
        let (l, _, _perm) = ilu0(&a);

        let data0 = compute_isai(&a, 0);
        let data1 = compute_isai(&a, 1);

        // Both should produce valid results
        for val in &data0.m_l.values {
            assert!(val.is_finite());
        }
        for val in &data1.m_l.values {
            assert!(val.is_finite());
        }

        // Compute M_L * L for both and check closeness to identity
        let ml0_dense = data0.m_l.to_dense();
        let ml1_dense = data1.m_l.to_dense();
        let l_dense = l.to_dense();
        let prod0 = dense_matmul(&ml0_dense, &l_dense);
        let prod1 = dense_matmul(&ml1_dense, &l_dense);

        let mut err0 = 0.0_f64;
        let mut err1 = 0.0_f64;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                err0 += (prod0[i][j] - expected).powi(2);
                err1 += (prod1[i][j] - expected).powi(2);
            }
        }

        // ISAI(1) should be at least as good as ISAI(0)
        assert!(
            err1 <= err0 + 1e-14,
            "ISAI(1) error ({}) should be <= ISAI(0) error ({})",
            err1,
            err0
        );
    }
}
