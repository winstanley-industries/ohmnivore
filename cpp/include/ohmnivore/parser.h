#ifndef OHMNIVORE_PARSER_H_
#define OHMNIVORE_PARSER_H_

#include <string_view>

#include "ohmnivore/ir.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Phase 2A recognizes RLC elements, independent voltage/current sources with
// DC values, .DC/.OP, .PRINT as a compatibility no-op, and .END. AC,
// transient, nonlinear, and other SPICE forms are rejected explicitly rather
// than weakened.
[[nodiscard]] Result<Circuit> ParseNetlist(std::string_view input);

} // namespace ohmnivore

#endif // OHMNIVORE_PARSER_H_
