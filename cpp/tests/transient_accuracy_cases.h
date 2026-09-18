#ifndef OHMNIVORE_TESTS_TRANSIENT_ACCURACY_CASES_H_
#define OHMNIVORE_TESTS_TRANSIENT_ACCURACY_CASES_H_

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "ohmnivore/transient.h"

namespace ohmnivore::test {

// Test-only physical circuits and continuous closed-form oracles. No production
// integration, error estimator or matrix helper is used to construct truth.
struct TransientAccuracyCase {
  std::string name;
  std::string deck;
  TranAnalysis analysis;
  double bias;
  double voltage_limit;
  double current_limit;
  bool has_inductor;
  std::function<std::array<double, 2>(double)> exact;
};

std::vector<TransientAccuracyCase> TransientAccuracyCases();

} // namespace ohmnivore::test
#endif
