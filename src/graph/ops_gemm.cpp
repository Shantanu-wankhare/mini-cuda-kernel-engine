// =============================================================================
//  src/graph/ops_gemm.cpp  -- GemmOp: shapes, cost, launch.
//
//  WHY .cpp AND NOT .cu: this file contains no <<<>>> and no __global__. It
//  CALLS launch_gemm_f32, whose declaration is plain C++ in kernels.hpp. That is
//  precisely the boundary kernels.hpp exists to define, and it means changing
//  the graph layer recompiles no CUDA at all.
// =============================================================================
#include "mcke/graph/op.hpp"

#include <string>

namespace mcke {

StatusOr<std::vector<Shape>> GemmOp::infer_shapes(const std::vector<Shape>& in) const {
  if (in.size() != 2)
    return InvalidArgumentError("GemmOp: expects exactly 2 inputs (A, B), got " +
                                std::to_string(in.size()));
  if (in[0].rank() != 2 || in[1].rank() != 2)
    return InvalidArgumentError("GemmOp: both inputs must be rank 2, got ranks " +
                                std::to_string(in[0].rank()) + " and " +
                                std::to_string(in[1].rank()));
  const dim_t m = in[0].dim(0), k = in[0].dim(1);
  const dim_t k2 = in[1].dim(0), n = in[1].dim(1);
  if (k != k2)
    return InvalidArgumentError("GemmOp: inner dimensions disagree -- A is [" +
                                std::to_string(m) + "," + std::to_string(k) +
                                "] and B is [" + std::to_string(k2) + "," +
                                std::to_string(n) + "]");

  // beta != 0 makes C an INPUT the graph does not model: the kernel reads C
  // before writing it, so under any buffer-reuse policy C is recycled bytes
  // whose contents depend on the schedule. That breaks the SSA single-producer
  // rule the whole graph is built on, and it would fail the numerics gate in a
  // way that looks exactly like a race and is not one. Refuse it here, at build
  // time, rather than let it become a confusing Colab afternoon.
  if (p_.beta != 0.0f)
    return UnimplementedError(
        "GemmOp: beta != 0 is not supported in Phase 4. A beta != 0 GEMM reads C, "
        "so C is a third input the graph does not model; modelling it properly "
        "means taking C as a declared input that produces a NEW output tensor.");

  // The (variant, tile) pair, checked here rather than inside launch_gemm_f32.
  // GemmTile's defaults are (128,128,8,8,8) while kTiledSmem is only
  // instantiated for (32,32,32,1,1), so the natural-looking
  // GemmParams{.variant = kTiledSmem} is otherwise an error discovered on a GPU.
  if (!kernels::gemm_tile_is_supported(p_.variant, p_.tile))
    return InvalidArgumentError(
        "GemmOp: tile (" + std::to_string(p_.tile.bm) + "," + std::to_string(p_.tile.bn) +
        "," + std::to_string(p_.tile.bk) + "," + std::to_string(p_.tile.tm) + "," +
        std::to_string(p_.tile.tn) + ") is not an instantiated configuration for this "
        "variant. kTiledSmem requires (32,32,32,1,1); the register-blocked and "
        "warp-tiled variants require (128,128,8,8,8); naive and cuBLAS ignore the tile.");

  return std::vector<Shape>{Shape{m, n}};
}

OpCost GemmOp::cost(const std::vector<Shape>& in) const {
  OpCost c;
  if (in.size() != 2 || in[0].rank() != 2 || in[1].rank() != 2) return c;
  const std::uint64_t m = static_cast<std::uint64_t>(in[0].dim(0));
  const std::uint64_t k = static_cast<std::uint64_t>(in[0].dim(1));
  const std::uint64_t n = static_cast<std::uint64_t>(in[1].dim(1));
  // Identical to bench/gemm_bench.cpp, deliberately: 2*M*N*K (the standard
  // convention, excluding the alpha/beta epilogue) and compulsory traffic at
  // beta == 0 (read A and B once, write C once, never read C). Matching the
  // bench means a graph's reported cost cross-checks against RESULTS.md sec 3d
  // rather than being a second, independently-wrong model.
  c.flops = 2ull * m * n * k;
  c.bytes = (m * k + k * n + m * n) * sizeof(float);
  return c;
}

Status GemmOp::launch(const OpContext& ctx, const std::vector<Tensor>& inputs,
                      const std::vector<Tensor>& outputs) {
#if MCKE_WITH_CUDA
  if (inputs.size() != 2 || outputs.size() != 1)
    return InvalidArgumentError("GemmOp::launch: expects 2 inputs and 1 output");
  const float* a = inputs[0].data_as<float>();
  const float* b = inputs[1].data_as<float>();
  float* c = outputs[0].data_as<float>();
  if (!a || !b || !c)
    return InvalidArgumentError("GemmOp::launch: a tensor is undefined or not f32");
  const dim_t m = inputs[0].shape().dim(0);
  const dim_t k = inputs[0].shape().dim(1);
  const dim_t n = inputs[1].shape().dim(1);
  // NOTE: no synchronisation of any kind here, and that is the invariant the
  // whole phase's overlap depends on (op.hpp's Op::launch contract).
  return kernels::launch_gemm_f32(a, b, c, m, n, k, p_.alpha, p_.beta, p_.variant,
                                  p_.tile, ctx.stream);
#else
  (void)ctx; (void)inputs; (void)outputs;
  return UnimplementedError("GemmOp::launch: built with MCKE_WITH_CUDA=0; "
                            "shapes and costs are available, kernels are not");
#endif
}

}  // namespace mcke
