#ifndef OHMNIVORE_COMPILER_H_
#define OHMNIVORE_COMPILER_H_

#include <complex>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr double kGminSiemens = 1e-12;

struct TransientRhsStamp {
  std::size_t index;
  double coefficient;
};

// A transient source replaces its own DC contribution at runtime. Starting
// from b_dc and adding coefficient * (waveform(t) - dc_value) therefore keeps
// all non-transient source contributions and the canonical source signs.
struct TransientSourceStamp {
  std::string name;
  double dc_value;
  TransientWaveform waveform;
  std::vector<TransientRhsStamp> rhs_stamps;
};

struct CapacitorInitialConstraint {
  std::string name;
  std::optional<std::size_t> positive_node_index;
  std::optional<std::size_t> negative_node_index;
};

struct InductorInitialConstraint {
  std::string name;
  std::size_t branch_index;
};

struct MnaSystem {
  CsrMatrix g;
  CsrMatrix c;
  std::vector<double> b_dc;
  std::vector<std::complex<double>> b_ac;
  std::vector<std::string> node_names;
  std::vector<std::string> branch_names;
  std::vector<TransientSourceStamp> transient_sources = {};
  std::vector<CapacitorInitialConstraint> capacitor_initial_constraints = {};
  std::vector<InductorInitialConstraint> inductor_initial_constraints = {};
};

[[nodiscard]] Result<MnaSystem> CompileMna(const Circuit &circuit);

// Forms A(omega) = G + j * omega * C by merging the independent canonical
// CSR patterns of G and C. Structural zeros are not required in either input.
[[nodiscard]] Result<ComplexCsrMatrix>
FormAcMatrix(const CsrMatrix &g, const CsrMatrix &c, double angular_frequency);

} // namespace ohmnivore

#endif // OHMNIVORE_COMPILER_H_
