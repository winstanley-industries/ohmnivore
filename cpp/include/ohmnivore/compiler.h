#ifndef OHMNIVORE_COMPILER_H_
#define OHMNIVORE_COMPILER_H_

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
  std::vector<std::string> node_names;
  std::vector<std::string> branch_names;
};

[[nodiscard]] Result<MnaSystem> CompileMna(const Circuit &circuit);

} // namespace ohmnivore

#endif // OHMNIVORE_COMPILER_H_
