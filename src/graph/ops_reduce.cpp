// =============================================================================
//  src/graph/ops_reduce.cpp  -- ReduceOp: row-wise sum / max / mean.
//
//  Same enum-bridge discipline as ops_bias_act.cpp: switches with no `default:`
//  so -Wswitch catches a grown enum, plus static_asserts so a REORDERING (which
//  -Wswitch cannot see) fails to compile.
//
//  This is also the only op with a workspace, which is why Op::workspace_bytes()
//  exists at all -- see the note there.
// =============================================================================
#include "mcke/graph/op.hpp"

#include <string>

namespace mcke {
namespace {

static_assert(static_cast<int>(ReduceParams::Kind::kSum) ==
              static_cast<int>(kernels::ReduceKind::kSum));
static_assert(static_cast<int>(ReduceParams::Kind::kMax) ==
              static_cast<int>(kernels::ReduceKind::kMax));
static_assert(static_cast<int>(ReduceParams::Kind::kMean) ==
              static_cast<int>(kernels::ReduceKind::kMean));
static_assert(static_cast<int>(ReduceParams::Variant::kSmemTree) ==
              static_cast<int>(kernels::ReduceVariant::kSmemTree));
static_assert(static_cast<int>(ReduceParams::Variant::kWarpShuffle) ==
              static_cast<int>(kernels::ReduceVariant::kWarpShuffle));
static_assert(static_cast<int>(ReduceParams::Variant::kTwoPass) ==
              static_cast<int>(kernels::ReduceVariant::kTwoPass));

[[maybe_unused]] kernels::ReduceKind to_kernel(ReduceParams::Kind k) {
  switch (k) {   // no default
    case ReduceParams::Kind::kSum:  return kernels::ReduceKind::kSum;
    case ReduceParams::Kind::kMax:  return kernels::ReduceKind::kMax;
    case ReduceParams::Kind::kMean: return kernels::ReduceKind::kMean;
  }
  return kernels::ReduceKind::kSum;
}

[[maybe_unused]] kernels::ReduceVariant to_kernel(ReduceParams::Variant v) {
  switch (v) {   // no default
    case ReduceParams::Variant::kSmemTree:    return kernels::ReduceVariant::kSmemTree;
    case ReduceParams::Variant::kWarpShuffle: return kernels::ReduceVariant::kWarpShuffle;
    case ReduceParams::Variant::kTwoPass:     return kernels::ReduceVariant::kTwoPass;
  }
  return kernels::ReduceVariant::kWarpShuffle;
}

// The axis must be the last one: every reduce kernel in Phase 3b is row-wise.
bool axis_is_last(int axis, int rank) { return axis == -1 || axis == rank - 1; }

}  // namespace

StatusOr<std::vector<Shape>> ReduceOp::infer_shapes(const std::vector<Shape>& in) const {
  if (in.size() != 1)
    return InvalidArgumentError("ReduceOp: expects exactly 1 input, got " +
                                std::to_string(in.size()));
  if (!axis_is_last(p_.axis, in[0].rank()))
    return InvalidArgumentError(
        "ReduceOp: axis " + std::to_string(p_.axis) + " is not the last axis of a rank-" +
        std::to_string(in[0].rank()) + " tensor. Phase 3b's kernels are row-wise only; "
        "reducing an interior axis needs a transpose or a different kernel.");

  // Drop the last dimension. A rank-1 input reduces to a single value, which we
  // represent as rank 1 with extent 1 rather than rank 0 -- Shape's minimum rank
  // is 1, and a "scalar" that silently became rank 0 would fail validate_shape.
  //
  // This is exactly why Shape needed a runtime-rank constructor: the output rank
  // is a runtime value, and an initializer_list cannot express that without a
  // hand-written switch over every rank up to kMaxRank.
  const int r = in[0].rank();
  if (r <= 1) return std::vector<Shape>{Shape{1}};
  dim_t dims[kMaxRank]{};
  for (int i = 0; i < r - 1; ++i) dims[i] = in[0].dim(i);
  return std::vector<Shape>{Shape(dims, r - 1)};
}

OpCost ReduceOp::cost(const std::vector<Shape>& in) const {
  OpCost c;
  if (in.empty()) return c;
  const std::uint64_t rows = static_cast<std::uint64_t>(in[0].rows());
  const std::uint64_t cols = static_cast<std::uint64_t>(in[0].cols());
  // Matches bench/reduce_bench.cpp exactly: rows*(cols-1) additions, and
  // compulsory traffic of the whole input plus one output per row.
  //
  // kMax's "flops" are COMPARISONS, not floating-point operations in the FMA
  // sense that peak_tflops measures. Counting them would inflate the arithmetic
  // intensity of an op that is already hopelessly memory-bound and make the
  // roofline lie about which roof binds it. Reported as 0, deliberately -- and
  // op.hpp already says reporting TFLOP/s for a reduction is a red flag.
  c.flops = (p_.kind == ReduceParams::Kind::kMax) ? 0ull : rows * (cols > 0 ? cols - 1 : 0);
  c.bytes = (rows * cols + rows) * sizeof(float);
  return c;
}

std::size_t ReduceOp::workspace_bytes(const std::vector<Shape>& in) const {
#if MCKE_WITH_CUDA
  if (in.empty()) return 0;
  return kernels::row_reduce_workspace_bytes(in[0].rows(), in[0].cols(),
                                             to_kernel(p_.variant));
#else
  // The host build has no kernel library to ask, so it cannot know the two-pass
  // split. Returning 0 is honest for a build that cannot launch anything anyway;
  // the planner only ever needs a real answer on a machine that will run it.
  (void)in;
  return 0;
#endif
}

Status ReduceOp::launch(const OpContext& ctx, const std::vector<Tensor>& inputs,
                        const std::vector<Tensor>& outputs) {
#if MCKE_WITH_CUDA
  if (inputs.size() != 1 || outputs.size() != 1)
    return InvalidArgumentError("ReduceOp::launch: expects 1 input and 1 output");
  const float* x = inputs[0].data_as<float>();
  float* out     = outputs[0].data_as<float>();
  if (!x || !out)
    return InvalidArgumentError("ReduceOp::launch: a tensor is undefined or not f32");
  const dim_t rows = inputs[0].shape().rows();
  const dim_t cols = inputs[0].shape().cols();
  // The workspace comes from the PLANNER as a plain pointer, never from an
  // allocator call here -- Phase 2 measured cudaMalloc at up to 720 us with a
  // device-synchronising free, and one of those inside run_async() would convert
  // the async runtime to a synchronous one.
  return kernels::launch_row_reduce_f32(x, out, rows, cols, to_kernel(p_.kind),
                                        to_kernel(p_.variant),
                                        static_cast<float*>(ctx.workspace),
                                        ctx.workspace_bytes, ctx.stream);
#else
  (void)ctx; (void)inputs; (void)outputs;
  return UnimplementedError("ReduceOp::launch: built with MCKE_WITH_CUDA=0");
#endif
}

}  // namespace mcke
