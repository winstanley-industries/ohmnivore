// Private native-FP64 device implementation for bounded transient jobs.
constexpr int N = 256;
constexpr int Threads = 256;
constexpr int Chunk = 256;
constexpr int Nonfinite = static_cast<int>(ErrorCode::kNonFinite) + 1;
constexpr int Singular = static_cast<int>(ErrorCode::kSingular) + 1;
constexpr int Nonconvergence = static_cast<int>(ErrorCode::kNonConvergence) + 1;
constexpr int Invalid = static_cast<int>(ErrorCode::kSolutionValidation) + 1;
constexpr int Budget = static_cast<int>(ErrorCode::kUnsupportedSize) + 1;

struct Pair {
  double time, value;
};
struct Source {
  int type, offset, count, stamp_offset, stamps;
  double dc;
  PulseWaveform pulse;
};
struct Stamp {
  int row;
  double coefficient;
};
struct RowSource {
  int source;
  double coefficient;
};
struct Reactive {
  int positive, negative;
  double absolute;
};
struct ExpressionStamp {
  int program;
  double coefficient;
  int offset;
};
struct Guard {
  int condition;
  unsigned int flags;
};
// The complete exported tree is validated before compacting its indices.
// Fourteen bits cover all 16,384 admitted nodes; guard stores condition + 1.
struct PackedExpressionNode {
  std::uint64_t first : 14, second : 14, third : 14, guard : 15;
  ExportedExpressionOp op : 4;
  std::uint64_t constant : 1, behavioral : 1, positive : 1;
};
static_assert(sizeof(PackedExpressionNode) == 8);
struct FactorTerm {
  std::uint16_t lower, upper;
};
struct FactorEntry {
  std::uint64_t begin : 24, count : 8;
  std::int64_t diagonal : 9, pivot : 9;
};
static_assert(sizeof(FactorEntry) == 8);
struct Model {
  bool refresh_enabled = true;
  bool shared_factor_metadata = false;
  bool shared_expression_metadata = false;
  bool shared_structure = false;
  int n, nodes, nnz, programs, sources, reactive, hard_count,
      expression_node_count;
  int expression_levels = 0, dependency_count = 0;
  const int *expression_jacobian_slots;
  const int *expression_order, *expression_level_offsets;
  const PackedExpressionNode *parallel_nodes;
  const double *literal_values;
  int factor_levels = 0, factor_terms = 0;
  const int *factor_level_offsets, *factor_order;
  const FactorEntry *factor_entries;
  const FactorTerm *factor_term;
  int factor_nonzeros = 0, forward_levels = 0, backward_levels = 0;
  const int *forward_offsets, *forward_rows, *backward_offsets, *backward_rows;
  const int *factor_rows, *factor_columns, *factor_diagonal;
  const int *row_permutation, *column_permutation, *factor_input_slots;
  const int *row_offsets, *columns, *expression_offsets;
  const double *g, *c, *b, *hard_points;
  const unsigned char *hard_wave;
  const ExpressionStamp *expression_stamps;
  const Source *source;
  const int *row_source_offsets;
  const RowSource *row_sources;
  const Pair *pwl;
  const Reactive *coordinates;
  const DeviceProgram *program;
  const std::uint32_t *dependencies;
  const int *gradient_leaf_offsets, *gradient_leaves;
  double maximum_step, minimum_step, stop, start;
  std::uint64_t maximum_attempts, maximum_accepted;
  int maximum_newton;
};
struct Progress {
  unsigned long long expression_cycles, factor_cycles, triangular_cycles,
      total_cycles;
  unsigned long long expression_value_cycles, expression_ad_cycles,
      factor_prepare_cycles, linear_residual_cycles, forward_cycles;
  std::uint64_t attempts, accepted, rejected, nonlinear;
  std::uint64_t factors, dense_factors, linear_retries, reuses, solves,
      refinements, expression_batches;
  std::uint64_t full_expressions, value_expressions, expression_cache_hits;
  std::uint64_t chord_iterations, chord_refreshes;
  std::uint64_t history_estimates, history_checks, doubling;
  std::uint64_t fallback_entries, fallback_recoveries;
  double time, proposed_step, older_time;
  int hard_index, error, output_count;
  bool recovery, has_current, has_older;
  bool first_audit, fallback;
  int since_audit, agreements;
};
struct Workspace {
  Progress progress;
  double *state, *full, *half, *second, *current, *proposed, *delta;
  double *companion_rhs, *affine_rhs, *residual, *row_scale, *solution,
      *correction, *work, *equil;
  double *inverse_equil, *linear_row_norm;
  double *base, *jacobian, *lu, *last_dynamic_jacobian, *factored_jacobian;
  int *permutation;
  double *history_current, *history_older, *history_trial;
  double *gradients;
  DeviceResult *expressions;
  double *output;
};
struct Shared {
  Workspace runtime;
  double reductions[Threads / 32], maximum, next_time, step, normalized_error;
  double validation_reductions[6][Threads / 32], validation_maxima[6];
  int error;
  bool sparse_factor, factor_valid;
  double active_scale, factored_scale;
  bool expression_cache_valid, expression_cache_derivatives;
  double *expression_state;
  int factor_changed;
  double *factor, *triangular, *inverse;
  double *expression_values, *expression_adjoints;
  const PackedExpressionNode *expression_nodes;
  unsigned char *expression_active;
  int pivot_failure;
  int dense_selected, dense_row_count, dense_column_count;
  int dense_rows[N], dense_columns[N];
  int *factor_order, *factor_level_offsets;
  const FactorEntry *factor_entries;
  const FactorTerm *factor_term;
  int *factor_columns;
  const int *matrix_rows, *matrix_columns;
  int factor_rows[N + 1];
  std::uint16_t factor_diagonal[N];
  std::uint16_t row_permutation[N], column_permutation[N];
  std::uint16_t forward_offsets[N + 1], forward_rows[N],
      backward_offsets[N + 1], backward_rows[N];

  bool backward, landing, waveform, audit, agreed, disable, checked,
      used_history;
};

// Error-free FP64 products and compensated FP64 accumulation protect affine
// RHS/residual cancellation. No FP32 arithmetic or reduced-precision solve.
struct Sum {
  double hi = 0, lo = 0;
  __device__ void Add(double value) {
    const double total = hi + value;
    const double displacement = total - hi;
    lo += (hi - (total - displacement)) + (value - displacement);
    hi = total;
  }
  __device__ void Product(double a, double b) {
    const double product = a * b;
    const double total = hi + product;
    const double displacement = total - hi;
    const double addition_error =
        (hi - (total - displacement)) + (product - displacement);
    lo += addition_error + fma(a, b, -product);
    hi = total;
  }
  __device__ double Value() const { return hi + lo; }
};
// Every caller supplies a nonnegative magnitude, scale or step. Retain fmax's
// NaN handling; no arithmetic precision is changed by this bitwise ordering.
__device__ double PositiveMaximum(double a, double b) {
  const auto ai = static_cast<unsigned long long>(__double_as_longlong(a)) &
                  0x7fffffffffffffffULL;
  const auto bi = static_cast<unsigned long long>(__double_as_longlong(b)) &
                  0x7fffffffffffffffULL;
  if (ai > 0x7ff0000000000000ULL)
    return b;
  if (bi > 0x7ff0000000000000ULL)
    return a;
  return __longlong_as_double(ai > bi ? ai : bi);
}
__device__ double Maximum(double value, Shared &shared) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  value = WarpMagnitudeMaximum(value);
  if (lane == 0)
    shared.reductions[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value =
        WarpMagnitudeMaximum(lane < Threads / 32 ? shared.reductions[lane] : 0);
    if (lane == 0)
      shared.maximum = value;
  }
  __syncthreads();
  return shared.maximum;
}
// All six validation maxima share one pair of block barriers. Their
// nonnegative inputs and reduction operations are unchanged.
__device__ void ValidationMaxima(double (&values)[6], Shared &s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int index = 0; index < 6; ++index) {
    const double value = WarpMagnitudeMaximum(values[index]);
    if (lane == 0)
      s.validation_reductions[index][warp] = value;
  }
  __syncthreads();
  if (warp < 6) {
    const double value = WarpMagnitudeMaximum(
        lane < Threads / 32 ? s.validation_reductions[warp][lane] : 0);
    if (lane == 0)
      s.validation_maxima[warp] = value;
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < 6; ++index)
    values[index] = s.validation_maxima[index];
}
__device__ void Reject(Shared &shared, int error) {
  atomicMax(&shared.error, error);
}
__device__ double Coordinate(const Reactive &coordinate, const double *state) {
  return (coordinate.positive < 0 ? 0 : state[coordinate.positive]) -
         (coordinate.negative < 0 ? 0 : state[coordinate.negative]);
}
__device__ double Waveform(const Model &model, const Source &source,
                           double time) {
  if (source.type == 0) {
    const auto *points = model.pwl + source.offset;
    if (time <= points[0].time)
      return points[0].value;
    for (int j = 1; j < source.count; ++j) {
      if (time <= points[j].time) {
        const double fraction =
            (time - points[j - 1].time) / (points[j].time - points[j - 1].time);
        return points[j - 1].value +
               (points[j].value - points[j - 1].value) * fraction;
      }
    }
    return points[source.count - 1].value;
  }
  const auto p = source.pulse;
  if (time < p.delay_seconds)
    return p.initial_value;
  const double elapsed = time - p.delay_seconds;
  double relative = elapsed;
  if (p.period_seconds < 1.7976931348623157e308) {
    relative = fmod(elapsed, p.period_seconds);
    if (relative < 0)
      relative += p.period_seconds;
    const double tolerance =
        16 * 2.2204460492503131e-16 *
        PositiveMaximum(PositiveMaximum(fabs(time), fabs(p.delay_seconds)),
                        PositiveMaximum(fabs(elapsed), p.period_seconds));
    if (relative <= tolerance || p.period_seconds - relative <= tolerance)
      relative = 0;
    double edge = p.rise_time_seconds;
    for (int i = 0; i < 3; ++i) {
      if (edge < p.period_seconds && fabs(relative - edge) <= tolerance)
        relative = edge;
      edge += i == 0 ? p.pulse_width_seconds : p.fall_time_seconds;
    }
  }
  if (relative < p.rise_time_seconds)
    return p.initial_value +
           (p.pulsed_value - p.initial_value) * relative / p.rise_time_seconds;
  if (relative < p.rise_time_seconds + p.pulse_width_seconds)
    return p.pulsed_value;
  if (relative <
      p.rise_time_seconds + p.pulse_width_seconds + p.fall_time_seconds)
    return p.pulsed_value +
           (p.initial_value - p.pulsed_value) *
               (relative - p.rise_time_seconds - p.pulse_width_seconds) /
               p.fall_time_seconds;
  return p.initial_value;
}
__device__ double Rhs(const Model &model, int row, double time) {
  double result = model.b[row];
  for (int at = model.row_source_offsets[row];
       at < model.row_source_offsets[row + 1]; ++at) {
    const auto stamp = model.row_sources[at];
    const auto source = model.source[stamp.source];
    result += stamp.coefficient * (Waveform(model, source, time) - source.dc);
  }
  return result;
}
__device__ void Expressions(const Model &m, Workspace &w, Shared &s,
                            const double *state, bool derivatives,
                            bool force_fresh = false) {
  const auto cycles = clock64();
  bool same = false;
  if (s.expression_cache_valid && !s.error) {
    double changed = 0;
    for (int index = threadIdx.x; index < m.n; index += Threads)
      if (__double_as_longlong(state[index]) !=
          __double_as_longlong(s.expression_state[index]))
        changed = 1;
    same = Maximum(changed, s) == 0;
  }
  const bool keep_derivatives = same && s.expression_cache_derivatives;
  if (same && !force_fresh && (!derivatives || keep_derivatives)) {
    if (threadIdx.x == 0) {
      ++w.progress.expression_cache_hits;
      w.progress.expression_cycles += clock64() - cycles;
    }
    __syncthreads();
    return;
  }
  const bool reuse_values = same && !force_fresh;
  if (threadIdx.x == 0)
    s.expression_cache_valid = false;
  __syncthreads();
  for (int i = threadIdx.x; i < m.expression_node_count; i += Threads) {
    if (!reuse_values) {
      s.expression_values[i] = 0;
      s.expression_active[i] = 0;
    }
    s.expression_adjoints[i] = 0;
  }
  for (int i = threadIdx.x; i < m.programs; i += Threads) {
    const auto program = m.program[i];
    for (unsigned j = 0; j < program.dependency_count; ++j) {
      const auto at = program.dependency_offset + j;
      if (!Bounded(state[m.dependencies[at]]))
        Reject(s, Nonfinite);
      if (derivatives)
        w.gradients[at] = 0;
    }
  }
  __syncthreads();
  const auto value_start = clock64();
  if (!reuse_values) {
    for (int level = 0; level < m.expression_levels; ++level) {
      for (int at = m.expression_level_offsets[level] + threadIdx.x;
           at < m.expression_level_offsets[level + 1]; at += Threads) {
        const int index = m.expression_order[at];
        if (index < 0)
          continue;
        const auto node = s.expression_nodes[index];
        const int condition = static_cast<int>(node.guard) - 1;
        const bool active =
            condition < 0 ||
            (s.expression_active[condition] &&
             ((s.expression_values[condition] != 0) == bool(node.positive)));
        s.expression_active[index] = active;
        if (!active)
          continue;
        const bool behavioral = node.behavioral;
        double a = 0, b = 0, value = 0;
        if (node.op != ExportedExpressionOp::kConstant &&
            node.op != ExportedExpressionOp::kState) {
          a = s.expression_values[node.first];
          if (node.op != ExportedExpressionOp::kNegate &&
              node.op != ExportedExpressionOp::kExp)
            b = s.expression_values[node.second];
        }
        switch (node.op) {
        case ExportedExpressionOp::kConstant:
          value = m.literal_values[index];
          break;
        case ExportedExpressionOp::kState:
          value = state[node.first];
          break;
        case ExportedExpressionOp::kNegate:
          value = -a;
          break;
        case ExportedExpressionOp::kAdd:
          value = a + b;
          break;
        case ExportedExpressionOp::kSubtract:
          value = a - b;
          break;
        case ExportedExpressionOp::kMultiply:
          value = a * b;
          break;
        case ExportedExpressionOp::kDivide:
          value = a / (behavioral ? Denominator(b) : b);
          break;
        case ExportedExpressionOp::kPower:
          value = pow(fabs(a), b);
          break;
        case ExportedExpressionOp::kExp:
          value = behavioral && a > 14 ? 1202604.284 * (a - 13) : exp(a);
          break;
        case ExportedExpressionOp::kLess:
          value = a < b ? 1.0 : 0.0;
          break;
        case ExportedExpressionOp::kGreater:
          value = a > b ? 1.0 : 0.0;
          break;
        case ExportedExpressionOp::kIf:
          value = s.expression_values[a != 0 ? node.second : node.third];
          break;
        }
        s.expression_values[index] = value;
        if (!Bounded(value))
          Reject(s, Nonfinite);
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < m.programs; i += Threads) {
    const auto program = m.program[i];
    w.expressions[i] = {s.expression_values[program.node_offset + program.root],
                        s.error ? 1U : 0U};
    s.expression_adjoints[program.node_offset + program.root] = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    w.progress.expression_value_cycles += clock64() - value_start;
  const auto ad_start = clock64();
  if (derivatives && !s.error) {
    for (int level = m.expression_levels - 1; level >= 0; --level) {
      for (int at = m.expression_level_offsets[level] + threadIdx.x;
           at < m.expression_level_offsets[level + 1]; at += Threads) {
        const int index = m.expression_order[at];
        if (index < 0)
          continue;
        const auto node = s.expression_nodes[index];
        const double adjoint = s.expression_adjoints[index];
        if (adjoint == 0 || node.constant ||
            node.op == ExportedExpressionOp::kState)
          continue;
        const double a = s.expression_values[node.first],
                     b = s.expression_values[node.second];
        const bool behavioral = node.behavioral;
        bool error = false;
        switch (node.op) {
        case ExportedExpressionOp::kConstant:
        case ExportedExpressionOp::kState:
          break;
        case ExportedExpressionOp::kNegate:
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.first,
                     adjoint, -1, &error);
          break;
        case ExportedExpressionOp::kAdd:
        case ExportedExpressionOp::kSubtract:
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.first,
                     adjoint, 1, &error);
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.second,
                     adjoint, node.op == ExportedExpressionOp::kAdd ? 1 : -1,
                     &error);
          break;
        case ExportedExpressionOp::kMultiply:
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.first,
                     adjoint, b, &error);
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.second,
                     adjoint, a, &error);
          break;
        case ExportedExpressionOp::kDivide: {
          const double denominator = behavioral ? Denominator(b) : b;
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.first,
                     adjoint, 1 / denominator, &error);
          AddAdjoint(s.expression_nodes, s.expression_adjoints, node.second,
                     adjoint, -(s.expression_values[index] / denominator),
                     &error);
          break;
        }
        case ExportedExpressionOp::kPower:
          if (!s.expression_nodes[node.first].constant) {
            double slope;
            if (b == 0 || (a == 0 && b > 1))
              slope = 0;
            else if (a == 0 && b == 1)
              slope = 1;
            else
              slope = b * pow(fabs(a), b - 1) * (a < 0 ? -1.0 : 1.0);
            AddAdjoint(s.expression_nodes, s.expression_adjoints, node.first,
                       adjoint, slope, &error);
          }
          if (!s.expression_nodes[node.second].constant)
            AddAdjoint(
                s.expression_nodes, s.expression_adjoints, node.second, adjoint,
                a == 0 && b > 0 ? 0 : s.expression_values[index] * log(fabs(a)),
                &error);
          break;
        case ExportedExpressionOp::kExp:
          AddAdjoint(
              s.expression_nodes, s.expression_adjoints, node.first, adjoint,
              behavioral && a > 14 ? 1202604.284 : s.expression_values[index],
              &error);
          break;
        case ExportedExpressionOp::kLess:
        case ExportedExpressionOp::kGreater:
          break;
        case ExportedExpressionOp::kIf:
          AddAdjoint(s.expression_nodes, s.expression_adjoints,
                     a != 0 ? node.second : node.third, adjoint, 1, &error);
          break;
        }
        if (error)
          Reject(s, Nonfinite);
      }
      __syncthreads();
    }
    // Each dependency owns its ordered list of state leaves. The list retains
    // the original descending AD accumulation order, including zero adjoints
    // from inactive branches, without rescanning non-state instructions.
    for (int dependency = threadIdx.x; dependency < m.dependency_count;
         dependency += Threads) {
      double gradient = 0;
      for (int at = m.gradient_leaf_offsets[dependency];
           at < m.gradient_leaf_offsets[dependency + 1]; ++at) {
        gradient += s.expression_adjoints[m.gradient_leaves[at]];
        if (!Bounded(gradient))
          Reject(s, Nonfinite);
      }
      w.gradients[dependency] = gradient;
    }
  }
  if (threadIdx.x == 0) {
    w.progress.expression_ad_cycles += clock64() - ad_start;
    ++w.progress.expression_batches;
    (derivatives ? w.progress.full_expressions
                 : w.progress.value_expressions) += m.programs;
  }
  for (int index = threadIdx.x; index < m.n; index += Threads)
    s.expression_state[index] = state[index];
  __syncthreads();
  if (threadIdx.x == 0) {
    s.expression_cache_valid = !s.error;
    s.expression_cache_derivatives =
        !s.error && (derivatives || keep_derivatives);
    w.progress.expression_cycles += clock64() - cycles;
  }
  __syncthreads();
}
__device__ double NonlinearHistory(const Model &m, const Workspace &w,
                                   int row) {
  double sum = 0;
  for (int k = m.expression_offsets[row]; k < m.expression_offsets[row + 1];
       ++k) {
    const auto stamp = m.expression_stamps[k];
    sum += stamp.coefficient * w.expressions[stamp.program].value;
  }
  return sum;
}
__device__ void Assemble(const Model &m, Workspace &w, Shared &s,
                         const double *state, bool derivatives,
                         bool force_fresh = false) {
  Expressions(m, w, s, state, derivatives, force_fresh);
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    double residual = 0, scale = fabs(w.companion_rhs[row]);
    Sum affine;
    affine.Add(w.companion_rhs[row]);
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
      const int col = s.matrix_columns[k];
      const double a = w.base[k], term = a * state[col];
      residual += term;
      scale += fabs(term);
      if (derivatives)
        w.jacobian[k] = a;
    }
    residual -= w.companion_rhs[row];
    for (int k = m.expression_offsets[row]; k < m.expression_offsets[row + 1];
         ++k) {
      const auto stamp = m.expression_stamps[k];
      const auto program = m.program[stamp.program];
      const double value =
          stamp.coefficient * w.expressions[stamp.program].value;
      residual += value;
      scale += fabs(value);
      if (derivatives) {
        Sum local;
        local.Add(-w.expressions[stamp.program].value);
        for (unsigned j = 0; j < program.dependency_count; ++j) {
          const auto at = program.dependency_offset + j;
          const auto col = m.dependencies[at];
          const double derivative = w.gradients[at];
          local.Product(derivative, state[col]);
          w.jacobian[m.expression_jacobian_slots[stamp.offset + j]] +=
              stamp.coefficient * derivative;
        }
        affine.Product(stamp.coefficient, local.hi);
        affine.Product(stamp.coefficient, local.lo);
      }
    }
    w.residual[row] = residual;
    w.row_scale[row] = scale;
    if (derivatives)
      w.affine_rhs[row] = affine.Value();
    if (!Bounded(residual) || !Bounded(scale) || !Bounded(affine.Value()))
      Reject(s, Nonfinite);
  }
  __syncthreads();
}

// A block cooperatively pivots and updates each private LU. Compact nonzero
// row/column lists skip structural zeros without changing the pivot sequence
// or per-entry arithmetic order. Pivots are never perturbed.
__device__ void DenseFactor(const Model &m, Workspace &w, Shared &s,
                            bool prefer_diagonal = false) {
  if (w.last_dynamic_jacobian) {
    for (int i = threadIdx.x; i < m.nnz; i += Threads)
      w.last_dynamic_jacobian[i] = w.jacobian[i];
  }
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    double scale = 0;
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k)
      scale = PositiveMaximum(scale, fabs(w.jacobian[k]));
    if (!(scale > 0) || !Bounded(scale)) {
      Reject(s, Singular);
      scale = 1;
    }
    w.equil[row] = scale;
    w.permutation[row] = row;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < m.n * N; i += Threads)
    w.lu[i] = 0;
  __syncthreads();
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
      const double value = w.jacobian[k] / w.equil[row];
      w.lu[row * N + s.matrix_columns[k]] = value;
      if (!Bounded(value) || (w.jacobian[k] != 0 && value == 0))
        Reject(s, Nonfinite);
    }
  }
  __syncthreads();
  for (int k = 0; k < m.n && !s.error; ++k) {
    double best = 0;
    for (int row = k + threadIdx.x; row < m.n; row += Threads)
      best = PositiveMaximum(best, fabs(w.lu[row * N + k]));
    best = Maximum(best, s);
    if (threadIdx.x == 0) {
      s.dense_selected = m.n;
      s.dense_row_count = 0;
      s.dense_column_count = 0;
      if (!(best >= 2.2250738585072014e-308) || !Bounded(best))
        Reject(s, Singular);
    }
    __syncthreads();
    if (s.error)
      break;
    for (int row = k + threadIdx.x; row < m.n; row += Threads)
      if (fabs(w.lu[row * N + k]) == best)
        atomicMin(&s.dense_selected, row);
    __syncthreads();
    if (threadIdx.x == 0) {
      const double diagonal = fabs(w.lu[k * N + k]);
      if (prefer_diagonal && diagonal >= .001 * best &&
          diagonal >= 2.2250738585072014e-308)
        s.dense_selected = k;
    }
    __syncthreads();
    const int selected = s.dense_selected;
    if (selected != k) {
      for (int col = threadIdx.x; col < m.n; col += Threads) {
        const double temp = w.lu[k * N + col];
        w.lu[k * N + col] = w.lu[selected * N + col];
        w.lu[selected * N + col] = temp;
      }
      if (threadIdx.x == 0) {
        const int value = w.permutation[k];
        w.permutation[k] = w.permutation[selected];
        w.permutation[selected] = value;
      }
    }
    __syncthreads();
    for (int row = k + 1 + threadIdx.x; row < m.n; row += Threads)
      if (w.lu[row * N + k] != 0)
        s.dense_rows[atomicAdd(&s.dense_row_count, 1)] = row;
    for (int col = k + 1 + threadIdx.x; col < m.n; col += Threads)
      if (w.lu[k * N + col] != 0)
        s.dense_columns[atomicAdd(&s.dense_column_count, 1)] = col;
    __syncthreads();
    const double pivot = w.lu[k * N + k];
    for (int at = threadIdx.x; at < s.dense_row_count; at += Threads) {
      const int row = s.dense_rows[at];
      w.lu[row * N + k] /= pivot;
    }
    __syncthreads();
    // Each destination is updated exactly once per pivot. Compacting the
    // nonzero row/column lists changes ownership, never arithmetic order.
    for (int at = threadIdx.x; at < s.dense_row_count * s.dense_column_count;
         at += Threads) {
      const int row = s.dense_rows[at / s.dense_column_count],
                col = s.dense_columns[at % s.dense_column_count];
      w.lu[row * N + col] =
          fma(-w.lu[row * N + k], w.lu[k * N + col], w.lu[row * N + col]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ++w.progress.factors;
    ++w.progress.dense_factors;
  }
  __syncthreads();
}
// Numeric pivot discovery executes on the device; the host only constructs the
// symbolic fill graph for the resulting row/column permutation.
__global__ void DiscoverPivots(Model m, Workspace *workspace, int *error,
                               bool prefer_diagonal) {
  __shared__ Shared shared;
  if (threadIdx.x == 0) {
    shared.error = 0;
    shared.matrix_rows = m.row_offsets;
    shared.matrix_columns = m.columns;
  }
  __syncthreads();
  DenseFactor(m, *workspace, shared, prefer_diagonal);
  if (threadIdx.x == 0)
    *error = shared.error;
}
__device__ void DenseTriangular(const Model &m, Workspace &w, Shared &s,
                                const double *rhs, double *result) {
  if (threadIdx.x < 32 && !s.error) {
    const int lane = threadIdx.x;
    for (int row = lane; row < m.n; row += 32)
      w.work[row] = rhs[w.permutation[row]] / w.equil[w.permutation[row]];
    __syncwarp();
    for (int row = 0; row < m.n; ++row) {
      double sum = 0;
      for (int col = lane; col < row; col += 32)
        sum = fma(w.lu[row * N + col], w.work[col], sum);
      for (int off = 16; off; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
      if (lane == 0)
        w.work[row] -= sum;
      __syncwarp();
    }
    for (int row = m.n - 1; row >= 0; --row) {
      double sum = 0;
      for (int col = row + 1 + lane; col < m.n; col += 32)
        sum = fma(w.lu[row * N + col], result[col], sum);
      for (int off = 16; off; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
      if (lane == 0) {
        result[row] = (w.work[row] - sum) / w.lu[row * N + row];
        if (!Bounded(result[row]))
          Reject(s, Nonfinite);
      }
      __syncwarp();
    }
  }
  __syncthreads();
}
__device__ void FactorImpl(const Model &m, Workspace &w, Shared &s) {
  const auto prepare_start = clock64();
  if (threadIdx.x == 0)
    s.sparse_factor = m.factor_nonzeros > 0;
  __syncthreads();
  if (!s.sparse_factor) {
    DenseFactor(m, w, s);
    return;
  }
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor[i] = 0;
  __syncthreads();
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    double scale = 0;
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k)
      scale = PositiveMaximum(scale, fabs(w.jacobian[k]));
    if (!(scale > 0) || !Bounded(scale)) {
      Reject(s, Singular);
      scale = 1;
    }
    w.equil[row] = scale;
    w.inverse_equil[row] = __drcp_rn(scale);
    double row_norm = 0;
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
      const double original = w.jacobian[k];
      const double scaled = isfinite(w.inverse_equil[row])
                                ? original * w.inverse_equil[row]
                                : original / scale;
      row_norm += fabs(original);
      if (original != 0 && scaled == 0)
        Reject(s, Nonfinite);
      s.factor[m.factor_input_slots[k]] = scaled;
    }
    w.linear_row_norm[row] = isfinite(w.inverse_equil[row])
                                 ? row_norm * w.inverse_equil[row]
                                 : row_norm / scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s.pivot_failure = 0;
    w.progress.factor_prepare_cycles += clock64() - prepare_start;
  }
  __syncthreads();
  for (int level = 0; level < m.factor_levels && !s.error; ++level) {
    for (int at = s.factor_level_offsets[level] + threadIdx.x;
         at < s.factor_level_offsets[level + 1]; at += Threads) {
      const int slot = s.factor_order[at];
      const auto entry = s.factor_entries[slot];
      double value = s.factor[slot];
      for (int j = entry.begin; j < entry.begin + entry.count; ++j) {
        const auto term = s.factor_term[j];
        value = fma(-s.factor[term.lower], s.factor[term.upper], value);
      }
      if (entry.diagonal >= 0)
        value *= s.inverse[entry.diagonal];
      s.factor[slot] = value;
      if (!Bounded(value))
        atomicExch(&s.pivot_failure, 1);
      if (entry.pivot >= 0) {
        if (!(fabs(value) >= 2.2250738585072014e-308))
          atomicExch(&s.pivot_failure, 1);
        s.inverse[entry.pivot] = __drcp_rn(value);
        if (!isfinite(s.inverse[entry.pivot]))
          atomicExch(&s.pivot_failure, 1);
      }
    }
    __syncthreads();
    if (s.pivot_failure)
      break;
  }
  if (threadIdx.x == 0 && s.pivot_failure)
    s.sparse_factor = false;
  if (threadIdx.x == 0)
    ++w.progress.factors;
  __syncthreads();
  if (!s.sparse_factor && !s.error)
    DenseFactor(m, w, s);
}
__device__ void Factor(const Model &m, Workspace &w, Shared &s) {
  const auto cycles = clock64();
  if (threadIdx.x == 0)
    s.factor_changed = !s.factor_valid;
  __syncthreads();
  for (int k = threadIdx.x; k < m.nnz; k += Threads)
    if (s.factor_valid && __double_as_longlong(w.jacobian[k]) !=
                              __double_as_longlong(w.factored_jacobian[k]))
      atomicExch(&s.factor_changed, 1);
  __syncthreads();
  if (!s.factor_changed) {
    if (threadIdx.x == 0)
      ++w.progress.reuses;
  } else {
    FactorImpl(m, w, s);
    for (int k = threadIdx.x; k < m.nnz; k += Threads)
      w.factored_jacobian[k] = w.jacobian[k];
    __syncthreads();
    if (threadIdx.x == 0) {
      s.factor_valid = !s.error;
      s.factored_scale = s.active_scale;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    w.progress.factor_cycles += clock64() - cycles;
}
__device__ double Equilibrated(double value, int row, const Workspace &w,
                               const Shared &s) {
  return s.sparse_factor && isfinite(w.inverse_equil[row])
             ? value * w.inverse_equil[row]
             : value / w.equil[row];
}
__device__ void TriangularImpl(const Model &m, Workspace &w, Shared &s,
                               const double *rhs, double *result) {
  if (!s.sparse_factor) {
    DenseTriangular(m, w, s, rhs, result);
    return;
  }
  if (threadIdx.x < 32 && !s.error) {
    const int lane = threadIdx.x;
    const auto forward_start = clock64();
    for (int level = 0; level < m.forward_levels; ++level) {
      for (int at = s.forward_offsets[level] + lane;
           at < s.forward_offsets[level + 1]; at += 32) {
        const int row = s.forward_rows[at], original = s.row_permutation[row];
        double value = Equilibrated(rhs[original], original, w, s);
        for (int j = s.factor_rows[row]; j < s.factor_diagonal[row]; ++j)
          value = fma(-s.factor[j], s.triangular[s.factor_columns[j]], value);
        s.triangular[row] = value;
      }
      __syncwarp();
    }
    if (threadIdx.x == 0)
      w.progress.forward_cycles += clock64() - forward_start;
    for (int level = 0; level < m.backward_levels; ++level) {
      for (int at = s.backward_offsets[level] + lane;
           at < s.backward_offsets[level + 1]; at += 32) {
        const int row = s.backward_rows[at];
        double value = s.triangular[row];
        for (int j = s.factor_diagonal[row] + 1; j < s.factor_rows[row + 1];
             ++j)
          value = fma(-s.factor[j], s.triangular[s.factor_columns[j]], value);
        s.triangular[row] = value * s.inverse[row];
        if (!Bounded(s.triangular[row]))
          Reject(s, Nonfinite);
      }
      __syncwarp();
    }
    for (int row = lane; row < m.n; row += 32)
      result[s.column_permutation[row]] = s.triangular[row];
  }
  __syncthreads();
}
__device__ void Triangular(const Model &m, Workspace &w, Shared &s,
                           const double *rhs, double *result) {
  const auto cycles = clock64();
  TriangularImpl(m, w, s, rhs, result);
  if (threadIdx.x == 0)
    w.progress.triangular_cycles += clock64() - cycles;
}
__device__ void Linear(const Model &m, Workspace &w, Shared &s) {
  Factor(m, w, s);
  if (s.error)
    return;
  for (int retry = 0; retry < 2 && !s.error; ++retry) {
    Triangular(m, w, s, w.affine_rhs, w.solution);
    bool corrected = false, accepted = false;
    for (int iteration = 0; iteration <= 4 && !s.error; ++iteration) {
      const auto residual_start = clock64();
      double component = 0, normalized = 0, matrix_norm = 0, rhs_norm = 0,
             solution_norm = 0, nonzero = 0;
      for (int row = threadIdx.x; row < m.n; row += Threads) {
        Sum residual;
        residual.Add(w.affine_rhs[row]);
        double scale = fabs(w.affine_rhs[row]),
               row_norm = s.sparse_factor ? w.linear_row_norm[row] : 0;
        for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
          const int col = s.matrix_columns[k];
          const double a = w.jacobian[k];
          residual.Product(-a, w.solution[col]);
          scale += fabs(a * w.solution[col]);
          if (!s.sparse_factor)
            row_norm += fabs(a);
        }
        if (!s.sparse_factor)
          row_norm /= w.equil[row];
        const double r = residual.Value();
        w.correction[row] = r;
        if (!Bounded(r))
          Reject(s, Nonfinite);
        if (r != 0)
          nonzero = 1;
        component = PositiveMaximum(component, scale == 0 ? (r == 0 ? 0 : 1e100)
                                                          : fabs(r) / scale);
        normalized =
            PositiveMaximum(normalized, Equilibrated(fabs(r), row, w, s));
        matrix_norm = PositiveMaximum(matrix_norm, row_norm);
        rhs_norm = PositiveMaximum(
            rhs_norm, Equilibrated(fabs(w.affine_rhs[row]), row, w, s));
        solution_norm = PositiveMaximum(solution_norm, fabs(w.solution[row]));
      }
      double maxima[]{component, normalized,    matrix_norm,
                      rhs_norm,  solution_norm, nonzero};
      ValidationMaxima(maxima, s);
      component = maxima[0];
      normalized = maxima[1];
      matrix_norm = maxima[2];
      rhs_norm = maxima[3];
      solution_norm = maxima[4];
      nonzero = maxima[5];
      if (threadIdx.x == 0)
        w.progress.linear_residual_cycles += clock64() - residual_start;
      const double denom = matrix_norm * solution_norm + rhs_norm;
      const bool valid =
          component <= 1e-5 &&
          (denom == 0 ? normalized == 0 : normalized / denom <= 1e-10);
      if ((corrected || nonzero == 0) && valid) {
        accepted = true;
        break;
      }
      if (iteration == 4)
        break;
      Triangular(m, w, s, w.correction, w.delta);
      for (int j = threadIdx.x; j < m.n; j += Threads) {
        w.solution[j] += w.delta[j];
        if (!Bounded(w.solution[j]))
          Reject(s, Nonfinite);
      }
      if (threadIdx.x == 0)
        ++w.progress.refinements;
      corrected = true;
      __syncthreads();
    }
    if (accepted || s.error)
      break;
    if (retry == 0 && s.sparse_factor) {
      if (threadIdx.x == 0) {
        s.sparse_factor = false;
        ++w.progress.linear_retries;
      }
      __syncthreads();
      DenseFactor(m, w, s);
    } else {
      if (threadIdx.x == 0)
        Reject(s, Invalid);
      __syncthreads();
    }
  }
  if (threadIdx.x == 0 && !s.error)
    ++w.progress.solves;
  __syncthreads();
}
__device__ double ResidualNorm(const Model &m, Workspace &w, Shared &s) {
  double norm = 0;
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    const double tolerance =
        (row < m.nodes ? 1e-9 : 1e-7) + 1e-5 * w.row_scale[row];
    const double value = fabs(w.residual[row]) / tolerance;
    if (!Bounded(value) || !(tolerance > 0))
      Reject(s, Nonfinite);
    norm = PositiveMaximum(norm, value);
  }
  return Maximum(norm, s);
}
__device__ double UpdateNorm(const Model &m, Workspace &w, Shared &s) {
  double norm = 0;
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    const double a = w.current[i], b = w.proposed[i];
    const double scale =
        (i < m.nodes ? 1e-7 : 1e-9) + 1e-5 * PositiveMaximum(fabs(a), fabs(b));
    const double value = fabs(b - a) / scale;
    if (!Bounded(value))
      Reject(s, Nonfinite);
    norm = PositiveMaximum(norm, value);
  }
  for (int i = threadIdx.x; i < m.reactive; i += Threads) {
    const auto coordinate = m.coordinates[i];
    const double a = Coordinate(coordinate, w.current),
                 b = Coordinate(coordinate, w.proposed);
    const double scale =
        .01 * (coordinate.absolute + 1e-4 * PositiveMaximum(fabs(a), fabs(b)));
    const double value = fabs(b - a) / scale;
    if (!Bounded(value))
      Reject(s, Nonfinite);
    norm = PositiveMaximum(norm, value);
  }
  return Maximum(norm, s);
}
// A chord iteration solves with a previously validated Jacobian at the exact
// same companion scale. Form J_old*x - F(x) with compensated products; retain
// the full fresh nonlinear residual and the linear certification of J_old.
__device__ void ChordRhs(const Model &m, Workspace &w, Shared &s,
                         const double *state) {
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    Sum affine;
    affine.Add(w.companion_rhs[row]);
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
      const double old = w.factored_jacobian[k];
      const int col = s.matrix_columns[k];
      affine.Product(old, state[col]);
      affine.Product(-w.base[k], state[col]);
      w.jacobian[k] = old;
    }
    for (int k = m.expression_offsets[row]; k < m.expression_offsets[row + 1];
         ++k) {
      const auto stamp = m.expression_stamps[k];
      affine.Product(-stamp.coefficient, w.expressions[stamp.program].value);
    }
    w.affine_rhs[row] = affine.Value();
    if (!Bounded(w.affine_rhs[row]))
      Reject(s, Nonfinite);
  }
  if (threadIdx.x == 0)
    ++w.progress.chord_iterations;
  __syncthreads();
}
__device__ void Newton(const Model &m, Workspace &w, Shared &s,
                       const double *initial, double *result) {
  for (int i = threadIdx.x; i < m.n; i += Threads)
    w.current[i] = initial[i];
  __syncthreads();
  bool chord = s.factor_valid && s.active_scale == s.factored_scale;
  Assemble(m, w, s, w.current, !chord);
  if (chord && !s.error)
    ChordRhs(m, w, s, w.current);
  double previous_residual = ResidualNorm(m, w, s);
  for (int iteration = 0; iteration < m.maximum_newton && !s.error;
       ++iteration) {
    Linear(m, w, s);
    if (s.error && chord) {
      // Refresh at the same current state if the lagged linear system fails.
      // This remains inside the bounded Newton solve, with no job retry.
      if (threadIdx.x == 0) {
        s.error = 0;
        s.factor_valid = false;
        ++w.progress.chord_refreshes;
      }
      __syncthreads();
      Assemble(m, w, s, w.current, true);
      if (!s.error)
        Linear(m, w, s);
      chord = false;
    }
    if (s.error)
      return;
    for (int i = threadIdx.x; i < m.n; i += Threads)
      w.delta[i] = w.solution[i] - w.current[i];
    __syncthreads();
    double scale = 1;
    for (int backtrack = 0; backtrack <= 16; ++backtrack) {
      if (threadIdx.x == 0)
        s.error = 0;
      __syncthreads();
      for (int i = threadIdx.x; i < m.n; i += Threads) {
        w.proposed[i] = w.current[i] + scale * w.delta[i];
        if (!Bounded(w.proposed[i]))
          Reject(s, Nonfinite);
      }
      __syncthreads();
      if (!s.error)
        Assemble(m, w, s, w.proposed, false);
      if (!s.error)
        break;
      if (s.error != Nonfinite)
        return;
      scale *= .5;
    }
    if (s.error) {
      if (threadIdx.x == 0)
        s.error = Nonconvergence;
      __syncthreads();
      return;
    }
    const double update = UpdateNorm(m, w, s), residual = ResidualNorm(m, w, s);
    if (s.error)
      return;
    if (update <= 1 && residual <= 1) {
      // Acceptance always checks the actual current analytic Jacobian, even
      // if every preceding chord residual and update is exactly zero.
      Assemble(m, w, s, w.proposed, true);
      if (s.error)
        return;
      double value =
          threadIdx.x < m.programs ? w.expressions[threadIdx.x].value : 0;
      Assemble(m, w, s, w.proposed, false, true);
      if (threadIdx.x < m.programs &&
          __double_as_longlong(value) !=
              __double_as_longlong(w.expressions[threadIdx.x].value))
        Reject(s, Invalid);
      const double final_residual = ResidualNorm(m, w, s);
      if (final_residual > 1 && threadIdx.x == 0)
        Reject(s, Invalid);
      __syncthreads();
      if (s.error)
        return;
      // A zero RHS cannot conceal singularity in the accepted Jacobian.
      Factor(m, w, s);
      if (threadIdx.x == 0 && !s.error)
        ++w.progress.solves;
      for (int i = threadIdx.x; i < m.n; i += Threads)
        result[i] = w.proposed[i];
      __syncthreads();
      return;
    }
    for (int i = threadIdx.x; i < m.n; i += Threads)
      w.current[i] = w.proposed[i];
    __syncthreads();
    chord = s.factor_valid && iteration % 3 != 2 &&
            residual < .5 * previous_residual;
    previous_residual = residual;
    if (chord)
      ChordRhs(m, w, s, w.current);
    else {
      if (threadIdx.x == 0)
        ++w.progress.chord_refreshes;
      Assemble(m, w, s, w.current, true);
    }
  }
  if (threadIdx.x == 0 && !s.error)
    s.error = Nonconvergence;
  __syncthreads();
}
__device__ void Step(const Model &m, Workspace &w, Shared &s,
                     const double *before, double t0, double t1, double h,
                     bool backward, bool left_limit, double *result) {
  if (!backward)
    Expressions(m, w, s, before, false);
  if (s.error)
    return;
  const double factor = (backward ? 1.0 : 2.0) / h;
  if (threadIdx.x == 0)
    s.active_scale = factor;
  const double source_time = left_limit ? nextafter(t1, t0) : t1;
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    double gp = 0, cp = 0;
    for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
      const int col = s.matrix_columns[k];
      gp += m.g[k] * before[col];
      cp += m.c[k] * before[col];
      w.base[k] = m.g[k] + factor * m.c[k];
      if (!Bounded(w.base[k]))
        Reject(s, Nonfinite);
    }
    w.companion_rhs[row] = backward ? Rhs(m, row, source_time) + factor * cp
                                    : Rhs(m, row, t1) + Rhs(m, row, t0) +
                                          factor * cp - gp -
                                          NonlinearHistory(m, w, row);
    if (!Bounded(w.companion_rhs[row]))
      Reject(s, Nonfinite);
  }
  __syncthreads();
  if (!s.error)
    Newton(m, w, s, before, result);
}
__device__ double LocalError(const Model &m, Workspace &w, Shared &s,
                             const double *high, const double *low,
                             double multiplier) {
  double norm = 0;
  for (int i = threadIdx.x; i < m.reactive; i += Threads) {
    const auto c = m.coordinates[i];
    const double a = Coordinate(c, high), b = Coordinate(c, low);
    const double scale = c.absolute + 1e-4 * PositiveMaximum(fabs(a), fabs(b));
    const double value = multiplier * fabs(a - b) / scale;
    if (!Bounded(value))
      Reject(s, Nonfinite);
    norm = PositiveMaximum(norm, value);
  }
  (void)w;
  return Maximum(norm, s);
}
__device__ void Integrate(const Model &m, Workspace &w, Shared &s) {
  const double t = w.progress.time, next = s.next_time, h = s.step;
  const double mid = t + h * .5;
  if (!(mid > t && mid < next)) {
    if (threadIdx.x == 0)
      Reject(s, Invalid);
    __syncthreads();
    return;
  }
  Step(m, w, s, w.state, t, next, h, s.backward, s.waveform, w.full);
  if (s.error)
    return;
  double history_error = 0, positive_feedback = 0;
  if (!s.backward && w.progress.has_current) {
    for (int i = threadIdx.x; i < m.reactive; i += Threads) {
      const auto c = m.coordinates[i];
      const double a = Coordinate(c, w.state), b = Coordinate(c, w.full);
      const double difference = b - a;
      const double derivative = 2.0 * difference / h - w.history_current[i];
      w.history_trial[i] = derivative;
      if (!Bounded(derivative))
        Reject(s, Nonfinite);
      if (w.progress.has_older) {
        const double k = t - w.progress.older_time;
        const double first = (derivative - w.history_current[i]) / h;
        const double previous = (w.history_current[i] - w.history_older[i]) / k;
        const double second = (first - previous) / (h + k);
        const double error = h * h * h * second / 6.0;
        const double refined = b - .75 * error;
        const double scale =
            c.absolute + 1e-4 * PositiveMaximum(fabs(b), fabs(refined));
        const double normalized = fabs(error) / scale;
        if (!Bounded(normalized))
          Reject(s, Nonfinite);
        history_error = PositiveMaximum(history_error, normalized);
        const double tolerance =
            c.absolute + 1e-4 * PositiveMaximum(fabs(a), fabs(b));
        if (fabs(difference) > .01 * tolerance) {
          const double gain =
              h * (derivative - w.history_current[i]) / difference;
          if (!Bounded(gain))
            Reject(s, Nonfinite);
          if (gain >= .5)
            positive_feedback = 1;
        }
      }
    }
  }
  history_error = Maximum(history_error, s);
  positive_feedback = Maximum(positive_feedback, s);
  if (s.error)
    return;
  const bool history_available =
      !s.backward && w.progress.has_current && w.progress.has_older;
  const bool doubling = s.backward || !history_available ||
                        w.progress.fallback || w.progress.first_audit ||
                        w.progress.since_audit >= 31 || history_error == 0 ||
                        positive_feedback != 0;
  if (threadIdx.x == 0) {
    s.normalized_error = history_error;
    s.audit = false;
    s.agreed = false;
    s.disable = false;
    s.checked = false;
    s.used_history = !doubling;
  }
  __syncthreads();
  if (doubling) {
    Step(m, w, s, w.state, t, mid, mid - t, s.backward, false, w.half);
    if (s.error)
      return;
    Step(m, w, s, w.half, mid, next, next - mid, s.backward,
         s.backward && s.waveform, w.second);
    if (s.error)
      return;
    const double error = s.backward
                             ? LocalError(m, w, s, w.second, w.full, 1.0)
                             : LocalError(m, w, s, w.full, w.second, 4.0 / 3.0);
    if (threadIdx.x == 0) {
      s.normalized_error = error;
      ++w.progress.doubling;
      s.checked = !s.backward;
      w.progress.history_checks += s.checked;
      s.audit = history_available;
      s.disable =
          history_available && error > .01 && error > 2.0 * history_error;
      s.agreed = history_available && !s.disable;
    }
    if (s.backward) {
      for (int i = threadIdx.x; i < m.reactive; i += Threads) {
        const auto c = m.coordinates[i];
        w.history_trial[i] =
            (Coordinate(c, w.second) - Coordinate(c, w.half)) / (next - mid);
        if (!Bounded(w.history_trial[i]))
          Reject(s, Nonfinite);
      }
      for (int i = threadIdx.x; i < m.n; i += Threads)
        w.full[i] = w.second[i];
    }
  }
  if (threadIdx.x == 0)
    w.progress.history_estimates += s.used_history;
  __syncthreads();
}
extern "C" __global__ void Emi03Advance(Model m, Workspace *workspace) {
  __shared__ Shared s;
  auto &w = s.runtime;
  const auto cycles = clock64();
  extern __shared__ double storage[];
  if (threadIdx.x == 0) {
    w = *workspace;
    s.error = 0;
    s.factor_valid = false;
    s.active_scale = 0;
    s.factored_scale = -1;
    s.expression_cache_valid = false;
    s.expression_cache_derivatives = false;
    w.progress.output_count = 0;
    s.factor = storage;
    s.triangular = storage + m.factor_nonzeros;
    s.inverse = s.triangular + m.n;
    double *vectors = s.inverse + m.n;
    w.state = vectors + 0 * m.n;
    w.full = vectors + 1 * m.n;
    w.half = vectors + 2 * m.n;
    w.current = vectors + 3 * m.n;
    w.proposed = vectors + 4 * m.n;
    w.delta = vectors + 5 * m.n;
    w.companion_rhs = vectors + 6 * m.n;
    w.affine_rhs = vectors + 7 * m.n;
    w.residual = vectors + 8 * m.n;
    w.row_scale = vectors + 9 * m.n;
    w.solution = vectors + 10 * m.n;
    w.equil = vectors + 11 * m.n;
    w.inverse_equil = vectors + 12 * m.n;
    w.linear_row_norm = vectors + 13 * m.n;
    s.expression_state = vectors + 14 * m.n;
    // Linear correction/forward work finish before the next nonlinear assembly
    // replaces residual/scale. The second half-step result is the final Newton
    // proposal and survives until its error/history checks have consumed it.
    w.correction = w.residual;
    w.work = w.row_scale;
    w.second = w.proposed;
    w.history_current = vectors + 15 * m.n;
    w.history_older = w.history_current + m.reactive;
    w.history_trial = w.history_older + m.reactive;
    w.base = w.history_trial + m.reactive;
    w.jacobian = w.base + m.nnz;
    w.gradients = w.jacobian + m.nnz;
    w.expressions =
        reinterpret_cast<DeviceResult *>(w.gradients + m.dependency_count);
    s.expression_values =
        reinterpret_cast<double *>(w.expressions + m.programs);
    s.expression_adjoints = s.expression_values + m.expression_node_count;
    s.expression_active = reinterpret_cast<unsigned char *>(
        s.expression_adjoints + m.expression_node_count);
    s.factor_columns = reinterpret_cast<int *>(
        (reinterpret_cast<std::uintptr_t>(s.expression_active +
                                          m.expression_node_count) +
         7) &
        ~std::uintptr_t{7});
    s.factor_order = s.factor_columns + m.factor_nonzeros;
    s.factor_level_offsets = s.factor_order + m.factor_nonzeros;
    s.factor_entries = m.factor_entries;
    s.factor_term = m.factor_term;
    auto end = (reinterpret_cast<std::uintptr_t>(s.factor_level_offsets +
                                                 m.factor_levels + 1) +
                7) &
               ~std::uintptr_t{7};
    if (m.shared_factor_metadata) {
      s.factor_entries = reinterpret_cast<const FactorEntry *>(end);
      s.factor_term = reinterpret_cast<const FactorTerm *>(s.factor_entries +
                                                           m.factor_nonzeros);
      end = (reinterpret_cast<std::uintptr_t>(s.factor_term + m.factor_terms) +
             7) &
            ~std::uintptr_t{7};
    }
    s.expression_nodes =
        m.shared_expression_metadata
            ? reinterpret_cast<const PackedExpressionNode *>(end)
            : m.parallel_nodes;
    if (m.shared_expression_metadata)
      end += m.expression_node_count * sizeof(PackedExpressionNode);
    s.matrix_rows =
        m.shared_structure ? reinterpret_cast<const int *>(end) : m.row_offsets;
    s.matrix_columns = m.shared_structure ? s.matrix_rows + m.n + 1 : m.columns;
  }
  __syncthreads();
  if (m.shared_structure) {
    for (int i = threadIdx.x; i <= m.n; i += Threads)
      const_cast<int *>(s.matrix_rows)[i] = m.row_offsets[i];
    for (int i = threadIdx.x; i < m.nnz; i += Threads)
      const_cast<int *>(s.matrix_columns)[i] = m.columns[i];
  }
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    w.state[i] = workspace->state[i];
    if (i < m.reactive) {
      w.history_current[i] = workspace->history_current[i];
      w.history_older[i] = workspace->history_older[i];
      w.history_trial[i] = workspace->history_trial[i];
    }
    s.factor_diagonal[i] = m.factor_diagonal[i];
    s.row_permutation[i] = m.row_permutation[i];
    s.column_permutation[i] = m.column_permutation[i];
    s.forward_rows[i] = m.forward_rows[i];
    s.backward_rows[i] = m.backward_rows[i];
  }
  for (int i = threadIdx.x; i <= m.n; i += Threads) {
    s.factor_rows[i] = m.factor_rows[i];
  }
  for (int i = threadIdx.x; i <= m.forward_levels; i += Threads)
    s.forward_offsets[i] = m.forward_offsets[i];
  for (int i = threadIdx.x; i <= m.backward_levels; i += Threads)
    s.backward_offsets[i] = m.backward_offsets[i];
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor_columns[i] = m.factor_columns[i];
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor_order[i] = m.factor_order[i];
  for (int i = threadIdx.x; i <= m.factor_levels; i += Threads)
    s.factor_level_offsets[i] = m.factor_level_offsets[i];
  __syncthreads();
  if (m.shared_factor_metadata) {
    for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
      const_cast<FactorEntry *>(s.factor_entries)[i] = m.factor_entries[i];
    for (int i = threadIdx.x; i < m.factor_terms; i += Threads)
      const_cast<FactorTerm *>(s.factor_term)[i] = m.factor_term[i];
  }
  if (m.shared_expression_metadata)
    for (int index = threadIdx.x; index < m.expression_node_count;
         index += Threads)
      const_cast<PackedExpressionNode *>(s.expression_nodes)[index] =
          m.parallel_nodes[index];
  __syncthreads();
  const auto dense_before = w.progress.dense_factors;
  for (int chunk = 0;
       (chunk < Chunk) && (w.progress.time < m.stop) && !w.progress.error &&
       (!m.refresh_enabled || w.progress.dense_factors == dense_before);
       ++chunk) {
    if (threadIdx.x == 0) {
      if (w.progress.attempts >= m.maximum_attempts)
        s.error = Budget;
      ++w.progress.attempts;
      while (w.progress.hard_index < m.hard_count &&
             m.hard_points[w.progress.hard_index] <= w.progress.time)
        ++w.progress.hard_index;
      if (w.progress.hard_index >= m.hard_count)
        s.error = Invalid;
      if (!s.error) {
        const double hard = m.hard_points[w.progress.hard_index];
        const double distance = hard - w.progress.time;
        double step = fmin(w.progress.proposed_step, m.maximum_step);
        double next = w.progress.time + step;
        s.landing = step >= distance || next >= hard;
        if (s.landing)
          next = hard;
        else if (hard - next < m.minimum_step)
          next = w.progress.time + distance * .5;
        step = next - w.progress.time;
        if (step > m.maximum_step) {
          next = nextafter(next, w.progress.time);
          step = next - w.progress.time;
          s.landing = next == hard;
        }
        s.next_time = next;
        s.step = step;
        if (!(step > 0) || (step < m.minimum_step && !s.landing) ||
            step > m.maximum_step)
          s.error = Invalid;
        s.waveform = s.landing && m.hard_wave[w.progress.hard_index];
        s.backward =
            w.progress.accepted == 0 || w.progress.recovery || s.waveform;
      }
    }
    __syncthreads();
    if (!s.error)
      Integrate(m, w, s);
    if (s.error) {
      if (threadIdx.x == 0) {
        if (s.error == Nonconvergence && s.step * .5 >= m.minimum_step) {
          ++w.progress.rejected;
          ++w.progress.nonlinear;
          w.progress.proposed_step = s.step * .5;
          w.progress.recovery = true;
          w.progress.agreements = 0;
          s.error = 0;
        } else
          w.progress.error = s.error;
      }
      __syncthreads();
      continue;
    }
    const double error = s.normalized_error;
    const double ratio =
        error == 0
            ? 2.0
            : fmin(2.0,
                   PositiveMaximum(.5, .9 * (s.backward ? sqrt(1.0 / error)
                                                        : cbrt(1.0 / error))));
    const double adapted = s.step * ratio;
    if (threadIdx.x == 0 && s.disable && !w.progress.fallback) {
      w.progress.fallback = true;
      ++w.progress.fallback_entries;
    }
    __syncthreads();
    if (error > 1) {
      if (threadIdx.x == 0) {
        ++w.progress.rejected;
        w.progress.agreements = 0;
        w.progress.proposed_step = adapted;
        w.progress.recovery = s.backward;
        if (adapted < m.minimum_step)
          w.progress.error = Invalid;
      }
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0 && w.progress.accepted >= m.maximum_accepted)
      w.progress.error = Budget;
    __syncthreads();
    if (w.progress.error)
      break;
    const bool reset = s.backward || s.waveform;
    for (int i = threadIdx.x; i < m.reactive; i += Threads) {
      if (!reset)
        w.history_older[i] = w.history_current[i];
      w.history_current[i] = w.history_trial[i];
    }
    for (int i = threadIdx.x; i < m.n; i += Threads) {
      w.state[i] = w.full[i];
      w.output[w.progress.output_count * (m.n + 1) + i + 1] = w.full[i];
    }
    // Every writer must finish the current row before its shared index changes.
    __syncthreads();
    if (threadIdx.x == 0) {
      w.output[w.progress.output_count * (m.n + 1)] = s.next_time;
      ++w.progress.output_count;
      ++w.progress.accepted;
      w.progress.has_older = !reset && w.progress.has_current;
      w.progress.has_current = true;
      w.progress.older_time = w.progress.time;
      w.progress.time = s.next_time;
      w.progress.proposed_step =
          fmin(m.maximum_step, PositiveMaximum(m.minimum_step, adapted));
      w.progress.recovery = s.waveform;
      if (reset) {
        w.progress.first_audit = true;
        w.progress.since_audit = 0;
        w.progress.fallback = false;
        w.progress.agreements = 0;
      } else if (s.audit) {
        w.progress.first_audit = false;
        w.progress.since_audit = 0;
        if (w.progress.fallback && s.agreed) {
          if (++w.progress.agreements == 16) {
            w.progress.fallback = false;
            w.progress.agreements = 0;
            ++w.progress.fallback_recoveries;
          }
        } else
          w.progress.agreements = 0;
      } else if (!w.progress.first_audit) {
        ++w.progress.since_audit;
        w.progress.agreements = 0;
      }
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    workspace->state[i] = w.state[i];
    if (i < m.reactive) {
      workspace->history_current[i] = w.history_current[i];
      workspace->history_older[i] = w.history_older[i];
      workspace->history_trial[i] = w.history_trial[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    w.progress.total_cycles += clock64() - cycles;
    workspace->progress = w.progress;
  }
}
