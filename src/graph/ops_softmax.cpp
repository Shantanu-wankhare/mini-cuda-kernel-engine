// =============================================================================
//  src/graph/ops_softmax.cpp  -- SoftmaxOp: row-wise softmax.
// =============================================================================
#include "mcke/graph/op.hpp"

#include <string>

namespace mcke {
namespace {

static_assert(static_cast<int>(SoftmaxParams::Variant::kThreePass) ==
              static_cast<int>(kernels::SoftmaxVariant::kThreePass));
static_assert(static_cast<int>(SoftmaxParams::Variant::kOnlineOnePass) ==
              static_cast<int>(kernels::SoftmaxVariant::kOnlineOnePass));

[[maybe_unused]] kernels::SoftmaxVariant to_kernel(SoftmaxParams::Variant v) {
  switch (v) {   // no default
    case SoftmaxParams::Variant::kThreePass:    return kernels::SoftmaxVariant::kThreePass;
    case SoftmaxParams::Variant::kOnlineOnePass:
      return kernels::SoftmaxVariant::kOnlineOnePass;
  }
  return kernels::SoftmaxVariant::kOnlineOnePass;
}

}  // namespace

StatusOr<std::vector<Shape>> SoftmaxOp::infer_shapes(const std::vector<Shape>& in) const {
  if (in.size() != 1)
    return InvalidArgumentError("SoftmaxOp: expects exactly 1 input, got " +
                                std::to_string(in.size()));
  if (!(p_.axis == -1 || p_.axis == in[0].rank() - 1))
    return InvalidArgumentError("SoftmaxOp: axis " + std::to_string(p_.axis) +
                                " is not the last axis of a rank-" +
                                std::to_string(in[0].rank()) +
                                " tensor; Phase 3c's kernels are row-wise only");
  // Both kernels subtract the row max before exponentiating, so there is no
  // "unstable" configuration to express. numerically_stable == false is not
  // silently ignored -- it is refused, because accepting a flag we do not honour
  // is how a caller ends up believing they measured something they did not.
  if (!p_.numerically_stable)
    return UnimplementedError(
        "SoftmaxOp: numerically_stable = false is not implemented. Both Phase 3c "
        "kernels subtract the row max unconditionally (without it expf overflows "
        "f32 above x = 88.7 and the row becomes NaN), so there is no unstable "
        "path to select.");
  return std::vector<Shape>{in[0]};
}

OpCost SoftmaxOp::cost(const std::vector<Shape>& in) const {
  OpCost c;
  if (in.empty()) return c;
  const std::uint64_t n = static_cast<std::uint64_t>(in[0].numel());
  // Matches bench/softmax_bench.cpp: ~5 flops/element (max, subtract, exp,
  // accumulate, divide) and compulsory traffic of one read plus one write.
  //
  // "Compulsory" is the operative word: kThreePass actually READS x three times
  // and kOnlineOnePass twice, but bytes() is contractually the IDEAL traffic
  // (op.hpp), because the ratio of ideal to Nsight's dram__bytes IS the
  // cache-efficiency measurement. Putting real traffic here would destroy that
  // signal -- and would also make the two variants incomparable, since they
  // would be divided by different denominators.
  c.flops = n * 5ull;
  c.bytes = 2ull * n * sizeof(float);
  return c;
}

Status SoftmaxOp::launch(const OpContext& ctx, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& outputs) {
#if MCKE_WITH_CUDA
  if (inputs.size() != 1 || outputs.size() != 1)
    return InvalidArgumentError("SoftmaxOp::launch: expects 1 input and 1 output");
  const float* x = inputs[0].data_as<float>();
  float* y       = outputs[0].data_as<float>();
  if (!x || !y)
    return InvalidArgumentError("SoftmaxOp::launch: a tensor is undefined or not f32");
  return kernels::launch_row_softmax_f32(x, y, inputs[0].shape().rows(),
                                         inputs[0].shape().cols(),
                                         to_kernel(p_.variant), ctx.stream);
#else
  (void)ctx; (void)inputs; (void)outputs;
  return UnimplementedError("SoftmaxOp::launch: built with MCKE_WITH_CUDA=0");
#endif
}

}  // namespace mcke
