#ifndef OHMNIVORE_PARSER_H_
#define OHMNIVORE_PARSER_H_

#include <string_view>

#include "ohmnivore/ir.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Phase 2B recognizes RLC elements, independent voltage/current sources with
// DC and/or small-signal AC specifications, .DC/.OP, .AC, .PRINT as a
// compatibility no-op, and .END. Transient, nonlinear, and other SPICE forms
// are rejected explicitly rather than weakened.
[[nodiscard]] Result<Circuit> ParseNetlist(std::string_view input);

} // namespace ohmnivore

#endif // OHMNIVORE_PARSER_H_
