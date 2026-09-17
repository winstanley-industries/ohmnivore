#ifndef OHMNIVORE_CPP_SRC_PREPARED_AC_INTERNAL_H_
#define OHMNIVORE_CPP_SRC_PREPARED_AC_INTERNAL_H_

#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::internal {

// Test/evidence-only numerical validation without fresh KLU certification.
// Production callers must use ValidatePreparedAcBatchResult instead.
[[nodiscard]] Result<bool>
ValidatePreparedAcBatchResultForEvidence(const PreparedAcBatch &batch,
                                         const PreparedAcBatchResult &result);

} // namespace ohmnivore::internal

#endif // OHMNIVORE_CPP_SRC_PREPARED_AC_INTERNAL_H_
