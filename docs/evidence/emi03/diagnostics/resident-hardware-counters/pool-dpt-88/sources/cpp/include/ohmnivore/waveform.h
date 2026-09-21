#ifndef OHMNIVORE_WAVEFORM_H_
#define OHMNIVORE_WAVEFORM_H_

#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Evaluates a validated transient waveform with the legacy PULSE/SIN/PWL/EXP
// boundary semantics. Invalid direct IR is a compile error; an invalid runtime
// time or a non-finite derived value is a solve error.
[[nodiscard]] Result<double>
EvaluateTransientWaveform(const TransientWaveform &waveform,
                          double time_seconds);

// Returns the sorted, unique waveform breakpoints in [0, stop_time_seconds].
// Periodic PULSE breakpoints are expanded deterministically and bounded.
[[nodiscard]] Result<std::vector<double>>
CollectTransientWaveformBreakpoints(const TransientWaveform &waveform,
                                    double stop_time_seconds);

} // namespace ohmnivore

#endif // OHMNIVORE_WAVEFORM_H_
