// =============================================================================
//  src/graph/ops_bias_act.cpp  -- BiasActOp: y = act(x + bias).
//
//  THE ENUM BRIDGE. BiasActParams::Act and kernels::Activation describe the same
//  four activations with the same spellings in the same order, and op.hpp's own
//  comment warns that "letting the same activation carry two different names
//  across that boundary is how a translation quietly maps the wrong case."
//
//  Two mechanisms make that warning enforceable rather than aspirational:
//    * the switch below has NO `default:` label, so -Wswitch fires the moment
//      either enum grows an enumerator the other lacks;
//    * static_asserts pin the numeric values, so a REORDERING (which -Wswitch
//      cannot see) fails to compile.
//  Both are compile-time, so both are checked on the MacBook.
//
//  This is not hypothetical care: GemmParams::Variant DID drift from five
//  enumerators to kernels::GemmVariant's eight during Phase 3, silently, with no
//  diagnostic, and was only found by design review. That enum is now an alias.
//  These three still exist separately because they are the graph's user-facing
//  vocabulary -- so they get the guards instead.
// =============================================================================
#include "mcke/graph/op.hpp"

#include <algorithm>
#include <string>

namespace mcke {
namespace {

static_assert(static_cast<int>(BiasActParams::Act::kNone) ==
              static_cast<int>(kernels::Activation::kNone));
static_assert(static_cast<int>(BiasActParams::Act::kRelu) ==
              static_cast<int>(kernels::Activation::kRelu));
static_assert(static_cast<int>(BiasActParams::Act::kGeluErf) ==
              static_cast<int>(kernels::Activation::kGeluErf));
static_assert(static_cast<int>(BiasActParams::Act::kGeluTanh) ==
              static_cast<int>(kernels::Activation::kGeluTanh));

[[maybe_unused]] kernels::Activation to_kernel(BiasActParams::Act a) {
  switch (a) {   // no default: -Wswitch must fire if either enum grows
    case BiasActParams::Act::kNone:     return kernels::Activation::kNone;
    case BiasActParams::Act::kRelu:     return kernels::Activation::kRelu;
    case BiasActParams::Act::kGeluErf:  return kernels::Activation::kGeluErf;
    case BiasActParams::Act::kGeluTanh: return kernels::Activation::kGeluTanh;
  }
  return kernels::Activation::kNone;
}

// FLOPs per element, matching bench/bias_act_bench.cpp exactly so a graph's cost
// cross-checks against the published RESULTS.md sec 3a numbers.
std::uint64_t act_flops_per_elem(BiasActParams::Act a) {
  switch (a) {
    case BiasActParams::Act::kNone:     return 1;    // the bias add
    case BiasActParams::Act::kRelu:     return 2;    // add + max
    case BiasActParams::Act::kGeluTanh: return 10;
    case BiasActParams::Act::kGeluErf:  return 22;
  }
  return 0;
}

}  // namespace

StatusOr<std::vector<Shape>> BiasActOp::infer_shapes(const std::vector<Shape>& in) const {
  if (in.size() != 2)
    return InvalidArgumentError("BiasActOp: expects 2 inputs (x, bias), got " +
                                std::to_string(in.size()));
  if (in[1].rank() != 1)
    return InvalidArgumentError("BiasActOp: bias must be rank 1, got rank " +
                                std::to_string(in[1].rank()));
  if (in[1].dim(0) != in[0].cols())
    return InvalidArgumentError("BiasActOp: bias length " + std::to_string(in[1].dim(0)) +
                                " does not match x's last dimension " +
                                std::to_string(in[0].cols()) +
                                " (bias broadcasts over the last axis)");
  if (p_.vector_width != 0 && p_.vector_width != 1 && p_.vector_width != 2 &&
      p_.vector_width != 4)
    return InvalidArgumentError("BiasActOp: vector_width must be 0 (auto), 1, 2 or 4");
  // The precondition is on COLS, not on the element count: row r starts at
  // element r*cols, so a float4 load needs every row start 16 B aligned.
  // rows=4, cols=3 has numel 12 divisible by 4 while every odd row is misaligned.
  if (p_.vector_width > 1 && in[0].cols() % p_.vector_width != 0)
    return InvalidArgumentError("BiasActOp: cols " + std::to_string(in[0].cols()) +
                                " is not a multiple of vector_width " +
                                std::to_string(p_.vector_width));
  return std::vector<Shape>{in[0]};
}

OpCost BiasActOp::cost(const std::vector<Shape>& in) const {
  OpCost c;
  if (in.empty()) return c;
  const std::uint64_t n    = static_cast<std::uint64_t>(in[0].numel());
  const std::uint64_t cols = static_cast<std::uint64_t>(in[0].cols());
  c.flops = n * act_flops_per_elem(p_.act);
  // FUSED traffic: read x once, write y once, read the bias vector once. The
  // unfused pair would be (4n + cols) -- twice the traffic for the same result,
  // which is the entire argument for fusion and the reason Phase 4 can express
  // both as graphs and measure the difference.
  c.bytes = (2ull * n + cols) * sizeof(float);
  return c;
}

Status BiasActOp::launch(const OpContext& ctx, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& outputs) {
#if MCKE_WITH_CUDA
  if (inputs.size() != 2 || outputs.size() != 1)
    return InvalidArgumentError("BiasActOp::launch: expects 2 inputs and 1 output");
  const float* x    = inputs[0].data_as<float>();
  const float* bias = inputs[1].data_as<float>();
  float* y          = outputs[0].data_as<float>();
  if (!x || !bias || !y)
    return InvalidArgumentError("BiasActOp::launch: a tensor is undefined or not f32");
  const dim_t rows = inputs[0].shape().rows();
  const dim_t cols = inputs[0].shape().cols();
  // The MIN over both pointers, not either one: a float4 store to y is illegal
  // if y is only 8-byte aligned even when x is 16-byte aligned, and the launcher
  // rejects an illegal width rather than downgrading. Taking one pointer's answer
  // and applying it to both is the kind of thing that works on every allocation
  // a 256-byte-aligned pool hands out and then fails on a sliced sub-buffer --
  // which is exactly what the Phase 4 memory planner produces.
  const int vw = p_.vector_width != 0
                     ? p_.vector_width
                     : std::min(kernels::max_vector_width_f32(x, cols),
                                kernels::max_vector_width_f32(y, cols));
  return kernels::launch_bias_act_f32(x, bias, y, rows, cols, to_kernel(p_.act), vw,
                                      ctx.stream, p_.max_row_blocks);
#else
  (void)ctx; (void)inputs; (void)outputs;
  return UnimplementedError("BiasActOp::launch: built with MCKE_WITH_CUDA=0");
#endif
}

}  // namespace mcke
