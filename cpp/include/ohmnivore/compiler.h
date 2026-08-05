#ifndef OHMNIVORE_COMPILER_H_
#define OHMNIVORE_COMPILER_H_

#include <complex>
#include <string>
#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr double kGminSiemens = 1e-12;

struct MnaSystem {
  CsrMatrix g;
  CsrMatrix c;
  std::vector<double> b_dc;
  std::vector<std::complex<double>> b_ac;
  std::vector<std::string> node_names;
  std::vector<std::string> branch_names;
};

[[nodiscard]] Result<MnaSystem> CompileMna(const Circuit &circuit);

// Forms A(omega) = G + j * omega * C by merging the independent canonical
// CSR patterns of G and C. Structural zeros are not required in either input.
[[nodiscard]] Result<ComplexCsrMatrix>
FormAcMatrix(const CsrMatrix &g, const CsrMatrix &c, double angular_frequency);

} // namespace ohmnivore

#endif // OHMNIVORE_COMPILER_H_
