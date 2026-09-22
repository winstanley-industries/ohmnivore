// Fixed-schedule nonlinear window research; never linked into the simulator.
#include "cuda/emi03_time_probe_common.cuh"
#include <iterator>
namespace ohmnivore {
namespace {
Model Prepare(const MnaSystem &system, const TranAnalysis &analysis,
              const TransientExecutionLimits &limits, Allocations &allocation,
              Workspace &workspace) {
  Model m{};
  if (system.g.rows == 0 || system.g.rows > N ||
      system.node_names.size() > system.g.rows ||
      system.behavioral_descriptors.empty() ||
      system.behavioral_descriptors.size() > 128 ||
      system.capacitor_initial_constraints.size() +
              system.inductor_initial_constraints.size() >
          N ||
      !system.diode_descriptors.empty() || !system.bjt_descriptors.empty() ||
      analysis.use_initial_conditions || limits.retain_output_states ||
      !limits.accepted_state_observer ||
      limits.behavioral_error_estimator !=
          BehavioralErrorEstimator::kDerivativeHistory)
    throw BackendError(
        ErrorCode::kUnsupported,
        "resident transient requires a bounded streamed behavioral job");
  if (!(std::isfinite(analysis.time_step_seconds) &&
        analysis.time_step_seconds > 0 &&
        std::isfinite(analysis.stop_time_seconds) &&
        analysis.stop_time_seconds > 0 &&
        std::isfinite(analysis.start_time_seconds) &&
        analysis.start_time_seconds >= 0 &&
        analysis.start_time_seconds <= analysis.stop_time_seconds &&
        std::isfinite(limits.minimum_step_divisor) &&
        limits.minimum_step_divisor >= 1 && limits.maximum_accepted_steps > 0 &&
        limits.maximum_step_attempts >= limits.maximum_accepted_steps &&
        limits.nonlinear_maximum_iterations > 0 &&
        limits.nonlinear_maximum_iterations <= 100))
    throw BackendError(ErrorCode::kInvalidStructure,
                       "invalid resident execution limits");
  auto valid = ValidateBehavioralTransient(system);
  if (!valid.ok())
    throw BackendError(valid.error().code, valid.error().message);
  auto pattern = FormTransientCompanionMatrix(system.g, system.c,
                                              analysis.time_step_seconds, 1);
  if (!pattern.ok())
    throw BackendError(pattern.error().code, pattern.error().message);
  m.n = static_cast<int>(system.g.rows);
  m.nodes = static_cast<int>(system.node_names.size());
  m.maximum_step = analysis.time_step_seconds;
  m.minimum_step = m.maximum_step / limits.minimum_step_divisor;
  m.stop = analysis.stop_time_seconds;
  m.start = analysis.start_time_seconds;
  m.maximum_attempts = limits.maximum_step_attempts;
  m.maximum_accepted = limits.maximum_accepted_steps;
  m.maximum_newton = static_cast<int>(limits.nonlinear_maximum_iterations);
  if (!(std::isfinite(m.minimum_step) && m.minimum_step > 0))
    throw BackendError(ErrorCode::kInvalidStructure,
                       "unrepresentable resident minimum step");
  std::vector<int> offsets, columns;
  std::vector<double> g, c;
  offsets.push_back(0);
  for (int row = 0; row < m.n; ++row) {
    auto ig = system.g.row_offsets[row], ic = system.c.row_offsets[row];
    for (auto k = pattern.value().row_offsets[row];
         k < pattern.value().row_offsets[row + 1]; ++k) {
      const auto column = pattern.value().column_indices[k];
      columns.push_back(static_cast<int>(column));
      const bool has_g = ig < system.g.row_offsets[row + 1] &&
                         system.g.column_indices[ig] == column;
      const bool has_c = ic < system.c.row_offsets[row + 1] &&
                         system.c.column_indices[ic] == column;
      g.push_back(has_g ? system.g.values[ig++] : 0);
      c.push_back(has_c ? system.c.values[ic++] : 0);
    }
    offsets.push_back(static_cast<int>(columns.size()));
  }
  m.nnz = static_cast<int>(columns.size());
  m.row_offsets = allocation.Upload(offsets);
  m.columns = allocation.Upload(columns);
  m.g = allocation.Upload(g);
  m.c = allocation.Upload(c);
  m.b = allocation.Upload(system.b_dc);
  std::vector<DeviceProgram> programs;
  std::vector<internal::ExportedExpressionNode> nodes;
  std::vector<std::uint32_t> ad, dependencies;
  for (const auto &descriptor : system.behavioral_descriptors) {
    auto exported = internal::ExportExpressionProgram(descriptor.expression);
    if (!exported.ok())
      throw BackendError(exported.error().code, exported.error().message);
    const auto &p = exported.value();
    if (p.nodes.empty() || p.nodes.size() > kMaximumNodes ||
        p.root >= p.nodes.size() ||
        p.state_size != static_cast<unsigned>(m.n) ||
        nodes.size() + p.nodes.size() > kMaximumTotalNodes)
      throw BackendError(ErrorCode::kUnsupportedSize,
                         "resident expression workspace bound");
    programs.push_back(DeviceProgram{
        static_cast<unsigned>(nodes.size()), static_cast<unsigned>(ad.size()),
        static_cast<unsigned>(dependencies.size()),
        static_cast<unsigned>(p.nodes.size()),
        static_cast<unsigned>(p.reverse_ad_indices.size()),
        static_cast<unsigned>(p.dependencies.size()), p.root, p.state_size,
        p.dialect});
    nodes.insert(nodes.end(), p.nodes.begin(), p.nodes.end());
    ad.insert(ad.end(), p.reverse_ad_indices.begin(),
              p.reverse_ad_indices.end());
    dependencies.insert(dependencies.end(), p.dependencies.begin(),
                        p.dependencies.end());
  }
  std::vector<internal::ExportedExpressionNode> parallel = nodes;
  std::vector<Guard> guards(nodes.size());
  std::vector<int> node_levels(nodes.size(), -1);
  for (const auto &program : programs) {
    std::function<int(unsigned, int, bool, int)> schedule;
    schedule = [&](unsigned local, int guard, bool positive, int floor) -> int {
      const auto index = program.node_offset + local;
      if (local >= program.nodes || node_levels[index] != -1)
        throw BackendError(ErrorCode::kInvalidStructure,
                           "resident expression must be a bounded tree");
      node_levels[index] = -2;
      guards[index] = Guard{
          guard, (program.dialect == ExpressionDialect::kBehavioral ? 1U : 0U) |
                     (positive ? 2U : 0U)};
      auto &node = parallel[index];
      int level = floor;
      if (node.op != ExportedExpressionOp::kConstant &&
          node.op != ExportedExpressionOp::kState) {
        const int first_level = schedule(node.first, guard, positive, floor);
        node.first += program.node_offset;
        level = std::max(level, first_level + 1);
        if (node.op == ExportedExpressionOp::kIf) {
          level = std::max(level,
                           schedule(node.second, static_cast<int>(node.first),
                                    true, first_level + 1) +
                               1);
          level =
              std::max(level, schedule(node.third, static_cast<int>(node.first),
                                       false, first_level + 1) +
                                  1);
          node.second += program.node_offset;
          node.third += program.node_offset;
        } else if (node.op != ExportedExpressionOp::kNegate &&
                   node.op != ExportedExpressionOp::kExp) {
          level = std::max(level,
                           schedule(node.second, guard, positive, floor) + 1);
          node.second += program.node_offset;
        }
      }
      node_levels[index] = level;
      return level;
    };
    schedule(program.root, -1, false, 0);
  }
  m.expression_levels =
      *std::max_element(node_levels.begin(), node_levels.end()) + 1;
  std::vector<int> expression_order, expression_starts{0};
  for (int level = 0; level < m.expression_levels; ++level) {
    std::vector<int> members;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
      if (node_levels[i] == level)
        members.push_back(i);
    std::stable_sort(members.begin(), members.end(), [&](int a, int b) {
      return parallel[a].op < parallel[b].op;
    });
    for (std::size_t at = 0; at < members.size(); ++at) {
      if (at && parallel[members[at]].op != parallel[members[at - 1]].op)
        while (expression_order.size() % 32)
          expression_order.push_back(-1);
      expression_order.push_back(members[at]);
    }
    while (expression_order.size() % 32)
      expression_order.push_back(-1);
    expression_starts.push_back(static_cast<int>(expression_order.size()));
  }
  std::vector<PackedExpressionNode> compact;
  std::vector<double> literals;
  for (std::size_t index = 0; index < parallel.size(); ++index) {
    const auto &node = parallel[index];
    const auto &guard = guards[index];
    if (node.first >= 16384 || node.second >= 16384 || node.third >= 16384 ||
        guard.condition < -1 || guard.condition >= 16384 ||
        static_cast<unsigned>(node.op) > 11 || node.constant > 1)
      throw BackendError(ErrorCode::kInvalidStructure,
                         "resident expression encoding bounds");
    PackedExpressionNode packed{};
    packed.first = node.first;
    packed.second = node.second;
    packed.third = node.third;
    packed.guard = guard.condition + 1;
    packed.op = node.op;
    packed.constant = node.constant;
    packed.behavioral = bool(guard.flags & 1);
    packed.positive = bool(guard.flags & 2);
    compact.push_back(packed);
    literals.push_back(node.value);
  }
  m.parallel_nodes = allocation.Upload(compact);
  m.literal_values = allocation.Upload(literals);
  m.expression_order = allocation.Upload(expression_order);
  m.expression_level_offsets = allocation.Upload(expression_starts);
  m.dependency_count = static_cast<int>(dependencies.size());
  m.programs = static_cast<int>(programs.size());
  m.program = allocation.Upload(programs);
  m.expression_node_count = static_cast<int>(nodes.size());
  std::vector<int> leaf_offsets{0}, leaves;
  for (const auto &program : programs) {
    for (unsigned dependency = 0; dependency < program.dependency_count;
         ++dependency) {
      for (unsigned j = 0; j < program.ad_count; ++j) {
        const auto index = program.node_offset + ad[program.ad_offset + j];
        const auto &node = parallel[index];
        if (node.op == ExportedExpressionOp::kState &&
            node.second == dependency)
          leaves.push_back(static_cast<int>(index));
      }
      leaf_offsets.push_back(static_cast<int>(leaves.size()));
    }
  }
  m.gradient_leaf_offsets = allocation.Upload(leaf_offsets);
  m.gradient_leaves = allocation.Upload(leaves);
  m.dependencies = allocation.Upload(dependencies);
  std::vector<int> expression_offsets{0}, jacobian_slots;
  std::vector<ExpressionStamp> expression_stamps;
  for (int row = 0; row < m.n; ++row) {
    for (std::size_t i = 0; i < system.behavioral_descriptors.size(); ++i) {
      for (const auto &stamp : system.behavioral_descriptors[i].rows) {
        if (stamp.row != static_cast<std::size_t>(row))
          continue;
        expression_stamps.push_back(
            ExpressionStamp{static_cast<int>(i), stamp.coefficient,
                            static_cast<int>(jacobian_slots.size())});
        for (unsigned j = 0; j < programs[i].dependency_count; ++j) {
          const auto column = dependencies[programs[i].dependency_offset + j];
          const auto found = std::lower_bound(
              columns.begin() + offsets[row],
              columns.begin() + offsets[row + 1], static_cast<int>(column));
          if (found == columns.begin() + offsets[row + 1] ||
              *found != static_cast<int>(column))
            throw BackendError(
                ErrorCode::kInvalidStructure,
                "resident Jacobian dependency absent from union");
          jacobian_slots.push_back(static_cast<int>(found - columns.begin()));
        }
      }
    }
    expression_offsets.push_back(static_cast<int>(expression_stamps.size()));
  }
  m.expression_offsets = allocation.Upload(expression_offsets);
  m.expression_stamps = allocation.Upload(expression_stamps);
  m.expression_jacobian_slots = allocation.Upload(jacobian_slots);
  double *persistent = allocation.Allocate<double>(4 * m.n);
  CheckCuda(cudaMemsetAsync(persistent, 0, 4 * m.n * sizeof(double),
                            allocation.stream()),
            "resident state initialization");
  emi03_cuda_internal::Synchronize(allocation.stream());
  workspace.state = persistent;
  workspace.history_current = persistent + m.n;
  workspace.history_older = persistent + 2 * m.n;
  workspace.history_trial = persistent + 3 * m.n;
  workspace.factored_jacobian = allocation.Allocate<double>(m.nnz);
  workspace.lu = allocation.Allocate<double>(m.n * N);
  workspace.permutation = allocation.Allocate<int>(m.n);
  workspace.last_dynamic_jacobian = allocation.Allocate<double>(m.nnz);
  workspace.output = allocation.Allocate<double>(Chunk * (m.n + 1));
  std::vector<Reactive> coordinates;
  for (const auto &cap : system.capacitor_initial_constraints)
    coordinates.push_back(Reactive{
        cap.positive_node_index ? static_cast<int>(*cap.positive_node_index)
                                : -1,
        cap.negative_node_index ? static_cast<int>(*cap.negative_node_index)
                                : -1,
        1e-7});
  for (const auto &ind : system.inductor_initial_constraints)
    coordinates.push_back(
        Reactive{static_cast<int>(ind.branch_index), -1, 1e-9});
  m.reactive = static_cast<int>(coordinates.size());
  m.coordinates = allocation.Upload(coordinates);
  if (m.reactive > m.n)
    throw BackendError(ErrorCode::kUnsupportedSize,
                       "resident reactive coordinate workspace bound");
  std::vector<Source> sources;
  std::vector<Stamp> stamps;
  std::vector<Pair> pairs;
  std::vector<double> waves;
  for (const auto &input : system.transient_sources) {
    Source source{};
    source.dc = input.dc_value;
    source.stamp_offset = static_cast<int>(stamps.size());
    source.stamps = static_cast<int>(input.rhs_stamps.size());
    for (const auto &stamp : input.rhs_stamps)
      stamps.push_back(Stamp{static_cast<int>(stamp.index), stamp.coefficient});
    if (const auto *p = std::get_if<PwlWaveform>(&input.waveform)) {
      source.type = 0;
      source.offset = static_cast<int>(pairs.size());
      source.count = static_cast<int>(p->time_value_pairs.size());
      for (const auto &[time, value] : p->time_value_pairs)
        pairs.push_back(Pair{time, value});
    } else if (const auto *p = std::get_if<PulseWaveform>(&input.waveform)) {
      source.type = 1;
      source.pulse = *p;
    } else
      throw BackendError(ErrorCode::kUnsupported, "resident waveform type");
    sources.push_back(source);
    auto points = CollectTransientWaveformBreakpoints(input.waveform, m.stop);
    if (!points.ok())
      throw BackendError(points.error().code, points.error().message);
    waves.insert(waves.end(), points.value().begin(), points.value().end());
  }
  std::sort(waves.begin(), waves.end());
  waves.erase(std::unique(waves.begin(), waves.end()), waves.end());
  auto hard = waves;
  hard.push_back(m.stop);
  hard.push_back(m.start);
  std::sort(hard.begin(), hard.end());
  hard.erase(std::unique(hard.begin(), hard.end()), hard.end());
  hard.erase(
      std::remove_if(hard.begin(), hard.end(), [](double t) { return t <= 0; }),
      hard.end());
  std::vector<unsigned char> hard_wave;
  for (double time : hard)
    hard_wave.push_back(std::binary_search(waves.begin(), waves.end(), time));
  m.hard_count = static_cast<int>(hard.size());
  m.hard_points = allocation.Upload(hard);
  m.hard_wave = allocation.Upload(hard_wave);
  m.sources = static_cast<int>(sources.size());
  m.source = allocation.Upload(sources);
  std::vector<int> row_source_offsets{0};
  std::vector<RowSource> row_sources;
  for (int row = 0; row < m.n; ++row) {
    // Preserve source/stamp order, including repeated contributions to one row.
    for (std::size_t index = 0; index < sources.size(); ++index) {
      const auto &source = sources[index];
      for (int k = 0; k < source.stamps; ++k) {
        const auto &stamp = stamps[source.stamp_offset + k];
        if (stamp.row == row)
          row_sources.push_back(
              RowSource{static_cast<int>(index), stamp.coefficient});
      }
    }
    row_source_offsets.push_back(static_cast<int>(row_sources.size()));
  }
  m.row_source_offsets = allocation.Upload(row_source_offsets);
  m.row_sources = allocation.Upload(row_sources);
  m.pwl = allocation.Upload(pairs);
  workspace.progress.first_audit = true;
  workspace.progress.proposed_step = m.maximum_step;
  return m;
}
struct NonlinearWindow {
  double *current, *candidate, *metrics, *lu, *factored;
  int *permutations, *error;
  double time;
  int length;
};
__global__ void NonlinearWindowSeed(const Model *models, const TimeData *data,
                                    NonlinearWindow window) {
  int at = blockIdx.x * blockDim.x + threadIdx.x, n = models[0].n;
  if (at < n * window.length)
    window.current[at] = data[0].initial[at % n];
}
__global__ void NonlinearWindowInitialJacobian(const Model *models,
                                               Workspace *all,
                                               const TimeData *data) {
  const auto m = models[0];
  const auto t = data[0];
  __shared__ Shared s;
  extern __shared__ double nonlinear_window_storage[];
  InitializeTime(m, all, s, nonlinear_window_storage);
  auto &w = s.runtime;
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    w.companion_rhs[row] = 0;
    for (int j = m.row_offsets[row]; j < m.row_offsets[row + 1]; ++j)
      w.base[j] = m.g[j] + m.c[j] / t.step;
  }
  __syncthreads();
  Assemble(m, w, s, t.initial, true, true);
  for (int j = threadIdx.x; j < m.nnz; j += Threads)
    all[0].jacobian[j] = w.jacobian[j];
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
}
__global__ void NonlinearWindowAffine(const Model *models, const Workspace *all,
                                      TimeData *data, NonlinearWindow window) {
  const int point = blockIdx.x;
  const auto m = models[0];
  const auto t = data[0];
  const double *x = window.current + point * m.n;
  __shared__ Shared s;
  extern __shared__ double nonlinear_window_storage[];
  InitializeTime(m, all, s, nonlinear_window_storage);
  auto &w = s.runtime;
  Expressions(m, w, s, x, false, true);
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    Sum affine;
    affine.Add(Rhs(m, row, window.time + (point + 1) * t.step));
    for (int j = m.row_offsets[row]; j < m.row_offsets[row + 1]; ++j) {
      affine.Product(all[0].jacobian[j] - (m.g[j] + m.c[j] / t.step),
                     x[m.columns[j]]);
    }
    for (int j = m.expression_offsets[row]; j < m.expression_offsets[row + 1];
         ++j) {
      auto stamp = m.expression_stamps[j];
      affine.Product(-stamp.coefficient, w.expressions[stamp.program].value);
    }
    t.rhs[point * m.n + row] = affine.Value();
    if (!Bounded(affine.Value()))
      Reject(s, Nonfinite);
  }
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
}
__global__ void NonlinearWindowBlend(const Model *models, const TimeData *data,
                                     NonlinearWindow window, double scale) {
  int at = blockIdx.x * blockDim.x + threadIdx.x, n = models[0].n;
  if (at < n * window.length)
    window.candidate[at] = window.current[at] +
                           scale * (data[0].solution[at] - window.current[at]);
}
__global__ void NonlinearWindowCheck(const Model *models, const Workspace *all,
                                     const TimeData *data,
                                     NonlinearWindow window, bool certify) {
  const int point = blockIdx.x;
  const auto m = models[0];
  const auto t = data[0];
  Workspace isolated = all[0];
  isolated.lu = window.lu + point * m.n * N;
  isolated.permutation = window.permutations + point * m.n;
  isolated.factored_jacobian = window.factored + point * m.nnz;
  __shared__ Shared s;
  extern __shared__ double nonlinear_window_storage[];
  InitializeTime(m, &isolated, s, nonlinear_window_storage);
  auto &w = s.runtime;
  const double *x = window.candidate + point * m.n,
               *old = window.current + point * m.n,
               *before =
                   point ? window.candidate + (point - 1) * m.n : t.initial;
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    w.current[row] = old[row];
    w.proposed[row] = x[row];
    Sum effective;
    effective.Add(Rhs(m, row, window.time + (point + 1) * t.step));
    for (int j = m.row_offsets[row]; j < m.row_offsets[row + 1]; ++j) {
      w.base[j] = m.g[j] + m.c[j] / t.step;
      effective.Product(m.c[j] / t.step, before[m.columns[j]]);
    }
    w.companion_rhs[row] = effective.Value();
    if (!Bounded(x[row]) || !Bounded(w.companion_rhs[row]))
      Reject(s, Nonfinite);
  }
  __syncthreads();
  Assemble(m, w, s, x, certify, true);
  const double residual = ResidualNorm(m, w, s), update = UpdateNorm(m, w, s);
  if (certify && !s.error) {
    if (threadIdx.x == 0)
      s.active_scale = 1 / t.step;
    Factor(m, w, s);
  }
  if (threadIdx.x == 0) {
    window.metrics[2 * point] = residual;
    window.metrics[2 * point + 1] = update;
    if (s.error)
      atomicCAS(window.error, 0, s.error);
    if (certify && (residual > 1 || update > 1))
      atomicCAS(window.error, 0, Invalid);
  }
}

struct WindowInput {
  MnaSystem system;
  std::vector<double> initial, oracle;
  double time, step;
  int length;
};
WindowInput ReadNonlinearWindow(const char *deck, const char *fixture,
                                int length) {
  WindowInput h{};
  std::ifstream d(deck);
  if (!d)
    throw std::runtime_error("missing nonlinear deck");
  const std::string text{std::istreambuf_iterator<char>(d), {}};
  h.system = ProbeChecked(
      CompileBehavioralMna(ProbeChecked(ParseBehavioralNetlist(text))));
  std::ifstream in(fixture);
  std::string magic;
  int n = 0, available = 0;
  in >> magic >> n >> available >> h.time >> h.step;
  if (!in || magic != "EMI03_NONLINEAR_TIME_1" ||
      n != static_cast<int>(h.system.g.rows) || n != 185 || available != 512 ||
      length < 1 || length > available || !std::isfinite(h.time) ||
      !std::isfinite(h.step) || h.step <= 0)
    throw std::runtime_error("invalid nonlinear time fixture");
  h.length = length;
  h.initial.resize(n);
  h.oracle.resize(available * n);
  for (auto *v : {&h.initial, &h.oracle})
    for (double &x : *v) {
      in >> x;
      if (!in || !std::isfinite(x))
        throw std::runtime_error("invalid nonlinear oracle state");
    }
  std::string extra;
  if (in >> extra)
    throw std::runtime_error("nonlinear fixture trailing data");
  return h;
}
std::size_t PrepareNonlinearWindow(const WindowInput &h,
                                   Allocations &allocation, Model &m,
                                   Workspace &w) {
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  limits.accepted_state_observer = [](double, const std::vector<double> &) {
    return Result<bool>::Ok(true);
  };
  limits.nonlinear_maximum_iterations = 100;
  TranAnalysis analysis{h.step, h.time + 512 * h.step, 0, false};
  m = Prepare(h.system, analysis, limits, allocation, w);
  auto working = h.system;
  working.g = ProbeChecked(
      FormTransientCompanionMatrix(h.system.g, h.system.c, h.step, 1));
  ProbeChecked(RemapBehavioralDescriptors(&working));
  auto linear = ProbeChecked(BuildNonlinearDcLinearization(working, h.initial));
  auto factor_system = working;
  factor_system.g = linear.jacobian;
  PrepareFactorPlan(factor_system, analysis, h.initial, m, allocation);
  w.jacobian = allocation.Allocate<double>(m.nnz);
  allocation.Copy(w.state, h.initial.data(), m.n * sizeof(double),
                  cudaMemcpyHostToDevice);
  std::size_t bytes =
      ((m.factor_nonzeros + 17 * m.n + 3 * m.reactive + 2 * m.nnz +
        m.dependency_count + 2 * m.expression_node_count) *
           sizeof(double) +
       m.programs * sizeof(DeviceResult) + m.expression_node_count + 7) /
          8 * 8 +
      (2 * m.factor_nonzeros + m.factor_levels + 1) * sizeof(int);
  bytes = (bytes + 7) / 8 * 8;
  // Keep complete expression and structure metadata in the original immutable
  // device arrays.
  m.shared_factor_metadata = false;
  m.shared_expression_metadata = false;
  m.shared_structure = false;
  return bytes;
}
void CheckNonlinearOracle(const WindowInput &h, const NonlinearWindow &window,
                          Allocations &allocation) {
  const int n = h.system.g.rows;
  std::vector<double> x(h.length * n);
  allocation.Copy(x.data(), window.candidate, x.size() * sizeof(double),
                  cudaMemcpyDeviceToHost);
  auto working = h.system;
  working.g = ProbeChecked(
      FormTransientCompanionMatrix(h.system.g, h.system.c, h.step, 1));
  ProbeChecked(RemapBehavioralDescriptors(&working));
  auto factor = ProbeChecked(SparseRealFactorization::Analyze(working.g));
  double maximum_residual = 0, maximum_difference = 0;
  std::vector<double> previous = h.initial;
  for (int point = 0; point < h.length; ++point) {
    std::vector<double> state(x.begin() + point * n,
                              x.begin() + (point + 1) * n);
    auto source = ProbeChecked(
        BuildTransientRhs(h.system, h.time + (point + 1) * h.step));
    working.b_dc = ProbeChecked(
        BuildBackwardEulerRhs(h.system.c, previous, source, h.step));
    maximum_residual =
        std::max(maximum_residual,
                 ProbeChecked(ValidateNonlinearResidual(working, state)));
    auto jac = ProbeChecked(BuildNonlinearDcLinearization(working, state));
    ProbeChecked(
        factor->FactorAndSolveRefined(jac.jacobian, std::vector<double>(n)));
    for (int row = 0; row < n; ++row) {
      double reference = h.oracle[point * n + row];
      double absolute =
          row < static_cast<int>(h.system.node_names.size()) ? 1e-7 : 1e-9;
      maximum_difference =
          std::max(maximum_difference,
                   std::abs(state[row] - reference) /
                       (absolute + 1e-5 * std::max(std::abs(state[row]),
                                                   std::abs(reference))));
    }
    previous = std::move(state);
  }
  if (maximum_residual > 1 || maximum_difference > 1)
    throw std::runtime_error("nonlinear window CPU differential failure");
  std::cout << "{\"kind\":\"nonlinear_validation\",\"length\":" << h.length
            << ",\"original_systems\":" << h.length
            << ",\"maximum_nonlinear_residual\":" << maximum_residual
            << ",\"maximum_scaled_cpu_difference\":" << maximum_difference
            << ",\"actual_jacobian_rank_checks\":" << h.length << "}\n";
}
__global__ void SequentialNonlinearWindow(const Model *models,
                                          const Workspace *all,
                                          const TimeData *data,
                                          NonlinearWindow window) {
  const auto m = models[0];
  const auto t = data[0];
  Workspace isolated = all[0];
  isolated.lu = window.lu;
  isolated.permutation = window.permutations;
  isolated.factored_jacobian = window.factored;
  __shared__ Shared s;
  extern __shared__ double nonlinear_sequential_storage[];
  InitializeTime(m, &isolated, s, nonlinear_sequential_storage);
  auto &w = s.runtime;
  for (int row = threadIdx.x; row < m.n; row += Threads)
    w.state[row] = t.initial[row];
  __syncthreads();
  for (int point = 0; point < window.length && !s.error; ++point) {
    Step(m, w, s, w.state, window.time + point * t.step,
         window.time + (point + 1) * t.step, t.step, true, false, w.full);
    for (int row = threadIdx.x; row < m.n; row += Threads) {
      window.candidate[point * m.n + row] = w.full[row];
      w.state[row] = w.full[row];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && s.error)
    atomicCAS(window.error, 0, s.error);
}

struct WindowGraphState {
  int iterations, refinements, backtrack, error, converged, accepted;
  unsigned long long total_refinements, total_backtracks;
  double previous_residual, residual, update;
  double trace[100][4];
};
__global__ void WindowGraphReset(WindowGraphState *state, TimeData *data,
                                 NonlinearWindow window) {
  if (threadIdx.x == 0) {
    *state = {};
    state->previous_residual = INFINITY;
    *data[0].error = 0;
    *data[0].invalid = 0;
    *data[0].nonzero = 0;
    *window.error = 0;
  }
}
__global__ void WindowGraphBeginLinear(WindowGraphState *s, TimeData *data,
                                       cudaGraphConditionalHandle loop) {
  if (threadIdx.x == 0) {
    s->refinements = 0;
    *data[0].invalid = 0;
    *data[0].nonzero = 0;
    cudaGraphSetConditional(loop, 1);
  }
}
__global__ void WindowGraphClearResidual(TimeData *data) {
  if (threadIdx.x == 0) {
    *data[0].invalid = 0;
    *data[0].nonzero = 0;
  }
}
__global__ void WindowGraphLinearDecision(WindowGraphState *s, TimeData *data,
                                          cudaGraphConditionalHandle loop,
                                          cudaGraphConditionalHandle correct) {
  if (threadIdx.x == 0) {
    bool again = false;
    if (*data[0].error)
      s->error = *data[0].error;
    else if (*data[0].invalid || (s->refinements == 0 && *data[0].nonzero)) {
      if (s->refinements < 4) {
        again = true;
        ++s->refinements;
        ++s->total_refinements;
      } else
        s->error = Invalid;
    }
    cudaGraphSetConditional(loop, again);
    cudaGraphSetConditional(correct, again);
  }
}
__global__ void WindowGraphBeginBacktrack(WindowGraphState *s,
                                          cudaGraphConditionalHandle loop) {
  if (threadIdx.x == 0) {
    s->backtrack = 0;
    s->accepted = 0;
    cudaGraphSetConditional(loop, s->error ? 0 : 1);
  }
}
__global__ void WindowGraphTrial(const Model *models, const TimeData *data,
                                 NonlinearWindow window, WindowGraphState *s) {
  const int at = blockIdx.x * blockDim.x + threadIdx.x, n = models[0].n;
  if (at == 0)
    *window.error = 0;
  if (at < n * window.length)
    window.candidate[at] =
        window.current[at] +
        ldexp(1., -s->backtrack) * (data[0].solution[at] - window.current[at]);
}
__global__ void WindowGraphBacktrackDecision(NonlinearWindow window,
                                             WindowGraphState *s,
                                             cudaGraphConditionalHandle loop) {
  if (threadIdx.x != 0)
    return;
  const int error = *window.error;
  double residual = 0, update = 0;
  if (!error)
    for (int i = 0; i < window.length; ++i) {
      residual = fmax(residual, window.metrics[2 * i]);
      update = fmax(update, window.metrics[2 * i + 1]);
    }
  if (error && error != Nonfinite)
    s->error = error;
  if (!error && (residual < s->previous_residual || residual <= 1)) {
    s->accepted = 1;
    s->residual = residual;
    s->update = update;
    cudaGraphSetConditional(loop, 0);
    return;
  }
  if (s->error || s->backtrack == 16) {
    if (!s->error)
      s->error = Nonconvergence;
    cudaGraphSetConditional(loop, 0);
    return;
  }
  ++s->backtrack;
  ++s->total_backtracks;
  cudaGraphSetConditional(loop, 1);
}
__global__ void WindowGraphNext(WindowGraphState *s,
                                cudaGraphConditionalHandle loop,
                                cudaGraphConditionalHandle certify) {
  if (threadIdx.x != 0)
    return;
  if (s->error || !s->accepted) {
    if (!s->error)
      s->error = Nonconvergence;
    cudaGraphSetConditional(loop, 0);
    cudaGraphSetConditional(certify, 0);
    return;
  }
  const int at = s->iterations++;
  s->trace[at][0] = s->residual;
  s->trace[at][1] = s->update;
  s->trace[at][2] = s->backtrack;
  s->trace[at][3] = s->refinements;
  s->converged = s->residual <= 1 && s->update <= 1;
  s->previous_residual = s->residual;
  if (!s->converged && s->iterations >= 100)
    s->error = Nonconvergence;
  cudaGraphSetConditional(loop, !s->error && !s->converged);
  cudaGraphSetConditional(certify, s->converged);
}
__global__ void WindowGraphCommit(const Model *models, NonlinearWindow window,
                                  const WindowGraphState *s) {
  const int at = blockIdx.x * blockDim.x + threadIdx.x;
  if (!s->error && !s->converged && at < window.length * models[0].n)
    window.current[at] = window.candidate[at];
}
__global__ void WindowGraphFinish(WindowGraphState *s, NonlinearWindow window) {
  if (threadIdx.x == 0 && *window.error) {
    s->error = *window.error;
    s->converged = 0;
  }
}
class WindowGraph {
public:
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  WindowGraph() {
    CheckCuda(cudaGraphCreate(&graph, 0), "window graph create");
  }
  ~WindowGraph() {
    if (executable && cudaGraphExecDestroy(executable) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    if (graph && cudaGraphDestroy(graph) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
  }
  template <class F, class... Args>
  static void Kernel(cudaGraph_t graph, cudaGraphNode_t &tail, dim3 grid,
                     dim3 block, std::size_t bytes, F function, Args... args) {
    void *arguments[]{static_cast<void *>(&args)...};
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void *>(function);
    p.gridDim = grid;
    p.blockDim = block;
    p.sharedMemBytes = bytes;
    p.kernelParams = arguments;
    cudaGraphNode_t next = nullptr;
    CheckCuda(cudaGraphAddKernelNode(&next, graph, tail ? &tail : nullptr,
                                     tail ? 1 : 0, &p),
              "window graph kernel");
    tail = next;
  }
  static cudaGraphConditionalHandle Handle(cudaGraph_t graph,
                                           unsigned int initial = 0) {
    cudaGraphConditionalHandle h{};
    CheckCuda(cudaGraphConditionalHandleCreate(&h, graph, initial,
                                               cudaGraphCondAssignDefault),
              "window conditional handle");
    return h;
  }
  static cudaGraph_t Conditional(cudaGraph_t graph, cudaGraphNode_t &tail,
                                 cudaGraphConditionalHandle handle,
                                 cudaGraphConditionalNodeType type) {
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = type;
    p.conditional.size = 1;
    cudaGraphNode_t next = nullptr;
    CheckCuda(cudaGraphAddNode(&next, graph, tail ? &tail : nullptr, nullptr,
                               tail ? 1 : 0, &p),
              "window conditional node");
    tail = next;
    return p.conditional.phGraph_out[0];
  }
  static void Pass(cudaGraph_t graph, cudaGraphNode_t &tail,
                   const Model *models, const Workspace *workspace,
                   TimeData *data, int length, bool correction) {
    Kernel(graph, tail, dim3((length + 7) / 8), dim3(Threads),
           8 * 3 * 185 * sizeof(double), TimeWarpIndependent, models, workspace,
           data, length, correction);
    bool input_a = true;
    for (int level = 0; (1 << level) < length; ++level) {
      Kernel(graph, tail, dim3(length), dim3(Threads), 0, TimeScan, data,
             length, level, input_a);
      input_a = !input_a;
    }
    Kernel(graph, tail, dim3(length), dim3(Threads), 0, TimeRecover, models,
           data, length, input_a, correction);
  }
  WindowGraphState Run(const Model *models, const Workspace *workspace,
                       TimeData *data, NonlinearWindow window,
                       std::size_t bytes, Allocations &allocation,
                       bool comparison, const WindowInput &input) {
    auto *state = allocation.Allocate<WindowGraphState>(1);
    const int length = window.length;
    cudaGraphNode_t root_tail = nullptr;
    const auto loop = Handle(graph, 1), certify = Handle(graph);
    Kernel(graph, root_tail, 1, 1, 0, WindowGraphReset, state, data, window);
    Kernel(graph, root_tail, dim3((length * 185 + Threads - 1) / Threads),
           dim3(Threads), 0, NonlinearWindowSeed, models, data, window);
    auto body = Conditional(graph, root_tail, loop, cudaGraphCondTypeWhile);
    cudaGraphNode_t tail = nullptr;
    Kernel(body, tail, dim3(length), dim3(Threads), bytes,
           NonlinearWindowAffine, models, workspace, data, window);
    Pass(body, tail, models, workspace, data, length, false);
    const auto linear_loop = Handle(body);
    Kernel(body, tail, 1, 1, 0, WindowGraphBeginLinear, state, data,
           linear_loop);
    auto linear_body =
        Conditional(body, tail, linear_loop, cudaGraphCondTypeWhile);
    cudaGraphNode_t linear_tail = nullptr;
    const auto correct = Handle(linear_body);
    Kernel(linear_body, linear_tail, 1, 1, 0, WindowGraphClearResidual, data);
    Kernel(linear_body, linear_tail, dim3(length), dim3(Threads), 0,
           TimeResidual, models, workspace, data, length);
    Kernel(linear_body, linear_tail, 1, 1, 0, WindowGraphLinearDecision, state,
           data, linear_loop, correct);
    auto correct_body =
        Conditional(linear_body, linear_tail, correct, cudaGraphCondTypeIf);
    cudaGraphNode_t correct_tail = nullptr;
    Pass(correct_body, correct_tail, models, workspace, data, length, true);
    const auto back = Handle(body);
    Kernel(body, tail, 1, 1, 0, WindowGraphBeginBacktrack, state, back);
    auto back_body = Conditional(body, tail, back, cudaGraphCondTypeWhile);
    cudaGraphNode_t back_tail = nullptr;
    Kernel(back_body, back_tail, dim3((length * 185 + Threads - 1) / Threads),
           dim3(Threads), 0, WindowGraphTrial, models, data, window, state);
    Kernel(back_body, back_tail, dim3(length), dim3(Threads), bytes,
           NonlinearWindowCheck, models, workspace, data, window, false);
    Kernel(back_body, back_tail, 1, 1, 0, WindowGraphBacktrackDecision, window,
           state, back);
    Kernel(body, tail, 1, 1, 0, WindowGraphNext, state, loop, certify);
    Kernel(body, tail, dim3((length * 185 + Threads - 1) / Threads),
           dim3(Threads), 0, WindowGraphCommit, models, window, state);
    auto certify_body =
        Conditional(graph, root_tail, certify, cudaGraphCondTypeIf);
    cudaGraphNode_t certify_tail = nullptr;
    Kernel(certify_body, certify_tail, dim3(length), dim3(Threads), bytes,
           NonlinearWindowCheck, models, workspace, data, window, true);
    Kernel(graph, root_tail, 1, 1, 0, WindowGraphFinish, state, window);
    CheckCuda(cudaGraphInstantiate(&executable, graph, 0),
              "window graph instantiate");
    WindowGraphState result{};
    CheckCuda(cudaFuncSetAttribute(SequentialNonlinearWindow,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "sequential window shared");
    for (int repetition = -1; repetition < (comparison ? 9 : 0); ++repetition)
      for (int order = 0; order < (comparison ? 2 : 1); ++order) {
        const bool graph_method =
            !comparison || ((repetition + 1 + order) % 2 == 0);
        CheckCuda(
            cudaMemsetAsync(window.error, 0, sizeof(int), allocation.stream()),
            "sequential error clear");
        const auto start = std::chrono::steady_clock::now();
        if (graph_method) {
          CheckCuda(cudaGraphLaunch(executable, allocation.stream()),
                    "window graph launch");
          allocation.Copy(&result, state, sizeof(result),
                          cudaMemcpyDeviceToHost);
        } else {
          SequentialNonlinearWindow<<<1, Threads, bytes, allocation.stream()>>>(
              models, workspace, data, window);
          CheckCuda(cudaGetLastError(), "sequential nonlinear launch");
          int error = 0;
          allocation.Copy(&error, window.error, sizeof(int),
                          cudaMemcpyDeviceToHost);
          if (error)
            throw std::runtime_error("sequential nonlinear failure " +
                                     std::to_string(error));
        }
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        std::cout << "{\"kind\":\"graph_comparison\",\"method\":\""
                  << (graph_method ? "graph" : "sequential")
                  << "\",\"repetition\":" << repetition
                  << ",\"length\":" << length << ",\"wall_s\":" << elapsed
                  << ",\"iterations\":"
                  << (graph_method ? result.iterations : 0)
                  << ",\"linear_refinements\":"
                  << (graph_method ? result.total_refinements : 0)
                  << ",\"backtracks\":"
                  << (graph_method ? result.total_backtracks : 0)
                  << ",\"error\":" << (graph_method ? result.error : 0)
                  << "}\n";
        if (comparison && (graph_method ? result.converged : true))
          CheckNonlinearOracle(input, window, allocation);
      }
    // Leave the graph result in the output for the caller's final
    // certification.
    if (comparison) {
      CheckCuda(cudaGraphLaunch(executable, allocation.stream()),
                "window graph final replay");
      allocation.Copy(&result, state, sizeof(result), cudaMemcpyDeviceToHost);
    }
    return result;
  }
};

struct WindowIterationStatus {
  int linear_error, linear_invalid, trial_error, reserved;
  double residual, update;
};
__global__ void SimpleWindowReset(TimeData *data, NonlinearWindow window) {
  if (threadIdx.x == 0) {
    *data[0].error = 0;
    *data[0].invalid = 0;
    *data[0].nonzero = 0;
    *window.error = 0;
  }
}
__global__ void SimpleWindowStatus(const TimeData *data, NonlinearWindow window,
                                   WindowIterationStatus *out) {
  if (threadIdx.x != 0)
    return;
  WindowIterationStatus value{};
  value.linear_error = *data[0].error;
  value.linear_invalid = *data[0].invalid;
  value.trial_error = *window.error;
  for (int i = 0; i < window.length; ++i) {
    if (!isfinite(window.metrics[2 * i]) ||
        !isfinite(window.metrics[2 * i + 1]))
      value.trial_error = Nonfinite;
    value.residual = fmax(value.residual, window.metrics[2 * i]);
    value.update = fmax(value.update, window.metrics[2 * i + 1]);
  }
  *out = value;
}
class SimpleWindowGraph : public WindowGraph {
  const Model *models_;
  const Workspace *workspace_;
  TimeData *data_;
  NonlinearWindow window_;
  std::size_t bytes_;
  Allocations &allocation_;
  WindowIterationStatus *status_;

public:
  SimpleWindowGraph(const Model *models, const Workspace *workspace,
                    TimeData *data, NonlinearWindow window, std::size_t bytes,
                    Allocations &allocation)
      : models_(models), workspace_(workspace), data_(data), window_(window),
        bytes_(bytes), allocation_(allocation) {
    status_ = allocation.Allocate<WindowIterationStatus>(1);
    const int length = window.length;
    cudaGraphNode_t tail = nullptr;
    Kernel(graph, tail, 1, 1, 0, SimpleWindowReset, data, window);
    Kernel(graph, tail, dim3(length), dim3(Threads), bytes,
           NonlinearWindowAffine, models, workspace, data, window);
    Pass(graph, tail, models, workspace, data, length, false);
    Kernel(graph, tail, dim3(length), dim3(Threads), 0, TimeResidual, models,
           workspace, data, length);
    // One correction is always attempted. The following full residual decides
    // whether another bounded correction is needed; no failed result is
    // accepted.
    Pass(graph, tail, models, workspace, data, length, true);
    Kernel(graph, tail, 1, 1, 0, WindowGraphClearResidual, data);
    Kernel(graph, tail, dim3(length), dim3(Threads), 0, TimeResidual, models,
           workspace, data, length);
    Kernel(graph, tail, dim3((length * 185 + Threads - 1) / Threads),
           dim3(Threads), 0, NonlinearWindowBlend, models, data, window, 1.);
    Kernel(graph, tail, dim3(length), dim3(Threads), bytes,
           NonlinearWindowCheck, models, workspace, data, window, false);
    Kernel(graph, tail, 1, 1, 0, SimpleWindowStatus, data, window, status_);
    CheckCuda(cudaGraphInstantiate(&executable, graph, 0),
              "simple window instantiate");
  }
  WindowIterationStatus ReadStatus() {
    WindowIterationStatus value{};
    allocation_.Copy(&value, status_, sizeof(value), cudaMemcpyDeviceToHost);
    return value;
  }
  WindowIterationStatus Trial(double scale) {
    CheckCuda(
        cudaMemsetAsync(window_.error, 0, sizeof(int), allocation_.stream()),
        "simple trial clear");
    NonlinearWindowBlend<<<(window_.length * 185 + Threads - 1) / Threads,
                           Threads, 0, allocation_.stream()>>>(models_, data_,
                                                               window_, scale);
    CheckCuda(cudaGetLastError(), "simple trial blend");
    NonlinearWindowCheck<<<window_.length, Threads, bytes_,
                           allocation_.stream()>>>(models_, workspace_, data_,
                                                   window_, false);
    CheckCuda(cudaGetLastError(), "simple trial check");
    SimpleWindowStatus<<<1, 1, 0, allocation_.stream()>>>(data_, window_,
                                                          status_);
    CheckCuda(cudaGetLastError(), "simple status");
    return ReadStatus();
  }
  WindowGraphState Solve(const std::vector<TimeData> &host_data) {
    WindowGraphState state{};
    state.previous_residual = INFINITY;
    NonlinearWindowSeed<<<(window_.length * 185 + Threads - 1) / Threads,
                          Threads, 0, allocation_.stream()>>>(models_, data_,
                                                              window_);
    CheckCuda(cudaGetLastError(), "simple window seed");
    for (int iteration = 0; iteration < 100; ++iteration) {
      CheckCuda(cudaGraphLaunch(executable, allocation_.stream()),
                "simple iteration launch");
      auto status = ReadStatus();
      int refinements = 1;
      ++state.total_refinements;
      while (!status.linear_error && status.linear_invalid && refinements < 4) {
        TimePass(models_, workspace_, data_, 1, host_data[0].rank,
                 window_.length, bytes_, allocation_, true);
        ClearTimeFlags(host_data, allocation_, false);
        TimeResidual<<<window_.length, Threads, 0, allocation_.stream()>>>(
            models_, workspace_, data_, window_.length);
        CheckCuda(cudaGetLastError(), "simple extra residual");
        status = Trial(1);
        ++refinements;
        ++state.total_refinements;
      }
      if (status.linear_error || status.linear_invalid) {
        state.error = status.linear_error ? status.linear_error : Invalid;
        break;
      }
      bool accepted = false;
      int backtrack = 0;
      for (; backtrack <= 16; ++backtrack) {
        if (backtrack) {
          status = Trial(std::ldexp(1., -backtrack));
          ++state.total_backtracks;
        }
        if (status.trial_error) {
          if (status.trial_error != Nonfinite) {
            state.error = status.trial_error;
            break;
          }
          continue;
        }
        if (status.residual < state.previous_residual || status.residual <= 1) {
          accepted = true;
          break;
        }
      }
      if (!accepted) {
        if (!state.error)
          state.error = Nonconvergence;
        break;
      }
      const int at = state.iterations++;
      state.trace[at][0] = state.residual = status.residual;
      state.trace[at][1] = state.update = status.update;
      state.trace[at][2] = backtrack;
      state.trace[at][3] = refinements;
      state.previous_residual = status.residual;
      if (status.residual <= 1 && status.update <= 1) {
        NonlinearWindowCheck<<<window_.length, Threads, bytes_,
                               allocation_.stream()>>>(models_, workspace_,
                                                       data_, window_, true);
        CheckCuda(cudaGetLastError(), "simple final certification");
        int error = 0;
        allocation_.Copy(&error, window_.error, sizeof(int),
                         cudaMemcpyDeviceToHost);
        state.error = error;
        state.converged = !error;
        break;
      }
      CheckCuda(cudaMemcpyAsync(window_.current, window_.candidate,
                                window_.length * 185 * sizeof(double),
                                cudaMemcpyDeviceToDevice, allocation_.stream()),
                "simple iteration commit");
    }
    if (!state.converged && !state.error)
      state.error = Nonconvergence;
    return state;
  }
};
WindowGraphState CompareSimpleWindow(const Model *models,
                                     const Workspace *workspace, TimeData *data,
                                     NonlinearWindow window, std::size_t bytes,
                                     Allocations &allocation,
                                     const std::vector<TimeData> &host_data,
                                     const WindowInput &input,
                                     bool comparison) {
  SimpleWindowGraph graph(models, workspace, data, window, bytes, allocation);
  WindowGraphState result{};
  CheckCuda(cudaFuncSetAttribute(SequentialNonlinearWindow,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes),
            "simple sequential shared");
  for (int repetition = -1; repetition < (comparison ? 9 : 0); ++repetition)
    for (int order = 0; order < (comparison ? 2 : 1); ++order) {
      const bool parallel = !comparison || ((repetition + 1 + order) % 2 == 0);
      CheckCuda(
          cudaMemsetAsync(window.error, 0, sizeof(int), allocation.stream()),
          "simple comparison clear");
      const auto start = std::chrono::steady_clock::now();
      if (parallel)
        result = graph.Solve(host_data);
      else {
        SequentialNonlinearWindow<<<1, Threads, bytes, allocation.stream()>>>(
            models, workspace, data, window);
        CheckCuda(cudaGetLastError(), "simple sequential launch");
        int error = 0;
        allocation.Copy(&error, window.error, sizeof(int),
                        cudaMemcpyDeviceToHost);
        if (error)
          throw std::runtime_error("simple sequential failure " +
                                   std::to_string(error));
      }
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      std::cout << "{\"kind\":\"graph_comparison\",\"method\":\""
                << (parallel ? "simple_graph" : "sequential")
                << "\",\"repetition\":" << repetition
                << ",\"length\":" << window.length << ",\"wall_s\":" << elapsed
                << ",\"iterations\":" << (parallel ? result.iterations : 0)
                << ",\"linear_refinements\":"
                << (parallel ? result.total_refinements : 0)
                << ",\"backtracks\":"
                << (parallel ? result.total_backtracks : 0)
                << ",\"error\":" << (parallel ? result.error : 0) << "}\n";
      if (comparison && (parallel ? result.converged : true))
        CheckNonlinearOracle(input, window, allocation);
    }
  if (comparison)
    result = graph.Solve(host_data);
  return result;
}

int NonlinearTimeProbe(const char *deck, const char *fixture, int length,
                       bool graph_mode, bool comparison, bool simple_mode) {
  auto h = ReadNonlinearWindow(deck, fixture, length);
  ProbeChecked(BeginEmi03CudaJob("nonlinear-time-window-probe"));
  bool converged = false;
  {
    HostStaging staging;
    Allocations allocation(staging);
    Model m{};
    Workspace w{};
    const auto bytes = PrepareNonlinearWindow(h, allocation, m, w);
    HostTime host{};
    host.sample.a.rows = m.n;
    host.mass = h.system.c;
    host.step = h.step;
    host.truth = h.initial;
    host.rhs.resize(512 * m.n);
    std::vector<int> active;
    for (std::size_t j = 0; j < host.mass.values.size(); ++j)
      if (host.mass.values[j] != 0)
        active.push_back(host.mass.column_indices[j]);
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    int *flags = allocation.Upload(std::vector<int>(3));
    std::vector<TimeData> data{
        PrepareTimeData(host, active, m, allocation, flags)};
    auto *dm = allocation.Upload(std::vector<Model>{m});
    auto *dw = allocation.Upload(std::vector<Workspace>{w});
    auto *dt = allocation.Upload(data);
    NonlinearWindow window{};
    window.time = h.time;
    window.length = length;
    window.current = allocation.Allocate<double>(length * m.n);
    window.candidate = allocation.Allocate<double>(length * m.n);
    window.metrics = allocation.Allocate<double>(2 * length);
    window.error = allocation.Upload(std::vector<int>(1));
    window.lu = allocation.Allocate<double>(length * m.n * N);
    window.factored = allocation.Allocate<double>(length * m.nnz);
    window.permutations = allocation.Allocate<int>(length * m.n);
    for (const auto kernel : {NonlinearWindowInitialJacobian})
      CheckCuda(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
                "window initial shared");
    CheckCuda(cudaFuncSetAttribute(NonlinearWindowAffine,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "window affine shared");
    CheckCuda(cudaFuncSetAttribute(NonlinearWindowCheck,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "window check shared");
    CheckCuda(cudaFuncSetAttribute(CacheTimeFactors,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "window factor shared");
    CheckCuda(cudaFuncSetAttribute(MakeTimeTransfer,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "window transfer shared");
    NonlinearWindowInitialJacobian<<<1, Threads, bytes, allocation.stream()>>>(
        dm, dw, dt);
    CheckCuda(cudaGetLastError(), "window initial Jacobian");
    CheckTimeErrors("window initial Jacobian", data, allocation);
    CacheTimeFactors<<<1, Threads, bytes, allocation.stream()>>>(dm, dw, dt);
    CheckCuda(cudaGetLastError(), "window factors");
    CheckTimeErrors("window factors", data, allocation);
    MakeTimeTransfer<<<active.size(), Threads, bytes, allocation.stream()>>>(
        dm, dw, dt, active.size());
    CheckCuda(cudaGetLastError(), "window transfer");
    CheckTimeErrors("window transfer", data, allocation);
    for (int level = 0; level < 9; ++level) {
      TimePower<<<dim3((active.size() * active.size() + Threads - 1) / Threads,
                       1),
                  Threads, 0, allocation.stream()>>>(dt, active.size(), level);
      CheckCuda(cudaGetLastError(), "window powers");
    }
    CheckTimeErrors("window powers", data, allocation);
    NonlinearWindowSeed<<<(length * m.n + Threads - 1) / Threads, Threads, 0,
                          allocation.stream()>>>(dm, dt, window);
    CheckCuda(cudaGetLastError(), "window seed");
    const auto start = std::chrono::steady_clock::now();
    double old_residual = std::numeric_limits<double>::infinity();
    int accepted_iterations = 0;
    if (graph_mode) {
      WindowGraph graph;
      auto result =
          simple_mode
              ? CompareSimpleWindow(dm, dw, dt, window, bytes, allocation, data,
                                    h, comparison)
              : graph.Run(dm, dw, dt, window, bytes, allocation, comparison, h);
      converged = result.converged;
      accepted_iterations = result.iterations;
      for (int i = 0; i < result.iterations; ++i)
        std::cout << "{\"kind\":\"graph_iteration\",\"iteration\":" << i
                  << ",\"residual\":" << result.trace[i][0]
                  << ",\"update\":" << result.trace[i][1]
                  << ",\"backtracks\":" << result.trace[i][2]
                  << ",\"linear_refinements\":" << result.trace[i][3] << "}\n";
      if (result.error && result.error != Nonconvergence)
        throw std::runtime_error("graph window failure " +
                                 std::to_string(result.error));
    } else {
      for (int iteration = 0; iteration < 100; ++iteration) {
        NonlinearWindowAffine<<<length, Threads, bytes, allocation.stream()>>>(
            dm, dw, dt, window);
        CheckCuda(cudaGetLastError(), "window affine");
        CheckTimeErrors("window affine", data, allocation);
        int refinements =
            SolveWindow(dm, dw, dt, data, length, bytes, allocation);
        bool accepted = false;
        double residual = 0, update = 0;
        int backtrack = 0;
        for (; backtrack <= 16; ++backtrack) {
          CheckCuda(cudaMemsetAsync(window.error, 0, sizeof(int),
                                    allocation.stream()),
                    "window trial error clear");
          NonlinearWindowBlend<<<(length * m.n + Threads - 1) / Threads,
                                 Threads, 0, allocation.stream()>>>(
              dm, dt, window, std::ldexp(1., -backtrack));
          CheckCuda(cudaGetLastError(), "window blend");
          NonlinearWindowCheck<<<length, Threads, bytes, allocation.stream()>>>(
              dm, dw, dt, window, false);
          CheckCuda(cudaGetLastError(), "window check");
          int error = 0;
          allocation.Copy(&error, window.error, sizeof(int),
                          cudaMemcpyDeviceToHost);
          if (error) {
            if (error != Nonfinite)
              throw std::runtime_error("nonlinear trial failure " +
                                       std::to_string(error));
            continue;
          }
          std::vector<double> metrics(2 * length);
          allocation.Copy(metrics.data(), window.metrics,
                          metrics.size() * sizeof(double),
                          cudaMemcpyDeviceToHost);
          residual = update = 0;
          for (int k = 0; k < length; ++k) {
            residual = std::max(residual, metrics[2 * k]);
            update = std::max(update, metrics[2 * k + 1]);
          }
          if (residual < old_residual || residual <= 1) {
            accepted = true;
            break;
          }
        }
        std::cout << "{\"kind\":\"nonlinear_iteration\",\"iteration\":"
                  << iteration << ",\"backtracks\":" << backtrack
                  << ",\"residual\":" << residual << ",\"update\":" << update
                  << ",\"linear_refinements\":" << refinements << "}\n";
        if (!accepted)
          break;
        ++accepted_iterations;
        if (residual <= 1 && update <= 1) {
          NonlinearWindowCheck<<<length, Threads, bytes, allocation.stream()>>>(
              dm, dw, dt, window, true);
          CheckCuda(cudaGetLastError(), "window final certification");
          int error = 0;
          allocation.Copy(&error, window.error, sizeof(int),
                          cudaMemcpyDeviceToHost);
          if (error)
            throw std::runtime_error("nonlinear final certification failure " +
                                     std::to_string(error));
          converged = true;
          break;
        }
        CheckCuda(cudaMemcpyAsync(window.current, window.candidate,
                                  length * m.n * sizeof(double),
                                  cudaMemcpyDeviceToDevice,
                                  allocation.stream()),
                  "window accepted iteration");
        old_residual = residual;
      }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cout << "{\"kind\":\"nonlinear_window\",\"length\":" << length
              << ",\"converged\":" << (converged ? "true" : "false")
              << ",\"iterations\":" << accepted_iterations
              << ",\"solve_wall_s\":" << seconds << "}\n";
    if (converged)
      CheckNonlinearOracle(h, window, allocation);
  }
  const auto ended = ProbeChecked(EndEmi03CudaJob());
  if (ended.outstanding_device_bytes || ended.cleanup_failures)
    throw std::runtime_error("nonlinear window cleanup failure");
  return converged ? 0 : 3;
}

} // namespace
} // namespace ohmnivore
int main(int argc, char **argv) {
  try {
    if ((argc != 4 && argc != 5) ||
        (argc == 5 && std::string(argv[4]) != "--graph" &&
         std::string(argv[4]) != "--compare" &&
         std::string(argv[4]) != "--simple" &&
         std::string(argv[4]) != "--simple-compare"))
      return 2;
    std::cout << std::setprecision(17);
    return ohmnivore::NonlinearTimeProbe(
        argv[1], argv[2], std::stoi(argv[3]), argc == 5,
        argc == 5 && (std::string(argv[4]) == "--compare" ||
                      std::string(argv[4]) == "--simple-compare"),
        argc == 5 && (std::string(argv[4]) == "--simple" ||
                      std::string(argv[4]) == "--simple-compare"));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
