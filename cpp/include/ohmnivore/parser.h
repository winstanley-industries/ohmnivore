#ifndef OHMNIVORE_PARSER_H_
#define OHMNIVORE_PARSER_H_

#include <string_view>

#include "ohmnivore/ir.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Phase 1 recognizes resistors, independent DC voltage sources, .DC/.OP,
// .PRINT as a compatibility no-op, and .END. Other SPICE elements are rejected
// explicitly rather than weakened.
[[nodiscard]] Result<Circuit> ParseNetlist(std::string_view input);

} // namespace ohmnivore

#endif // OHMNIVORE_PARSER_H_
