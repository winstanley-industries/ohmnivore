#ifndef OHMNIVORE_COMPILER_H_
#define OHMNIVORE_COMPILER_H_

#include <array>
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

struct DiodeDescriptor {
  std::string name;
  std::optional<std::size_t> anode_node_index;
  std::optional<std::size_t> cathode_node_index;
  double saturation_current_amperes;
  double emission_voltage_volts;
  // Canonical Jacobian positions. Ground-related entries are absent.
  std::optional<std::size_t> anode_anode_value_index;
  std::optional<std::size_t> anode_cathode_value_index;
  std::optional<std::size_t> cathode_anode_value_index;
  std::optional<std::size_t> cathode_cathode_value_index;
};

struct BjtDescriptor {
  std::string name;
  std::optional<std::size_t> collector_node_index;
  std::optional<std::size_t> base_node_index;
  std::optional<std::size_t> emitter_node_index;
  double polarity;
  double saturation_current_amperes;
  double forward_current_gain;
  double reverse_current_gain;
  double forward_emission_voltage_volts;
  double reverse_emission_voltage_volts;
  // Row-major CC, CB, CE, BC, BB, BE, EC, EB, EE positions. Positions
  // involving ground are absent; aliased terminals may share a value index.
  std::array<std::optional<std::size_t>, 9> jacobian_value_indices;
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
  std::vector<DiodeDescriptor> diode_descriptors = {};
  std::vector<BjtDescriptor> bjt_descriptors = {};
};

[[nodiscard]] Result<MnaSystem> CompileMna(const Circuit &circuit);

// Forms A(omega) = G + j * omega * C by merging the independent canonical
// CSR patterns of G and C. Structural zeros are not required in either input.
[[nodiscard]] Result<ComplexCsrMatrix>
FormAcMatrix(const CsrMatrix &g, const CsrMatrix &c, double angular_frequency);

} // namespace ohmnivore

#endif // OHMNIVORE_COMPILER_H_
