// =============================================================================
//  mcke/kernels/kernels.hpp
//
//  WHAT: Host-callable *launcher* declarations for every kernel, plus the tile
//        configuration types.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — launchers in .cu, declarations in a plain .hpp.
//
//  The `<<<grid, block, smem, stream>>>` syntax is nvcc-only, so every launch
//  site must live in a .cu file. But we do NOT want the whole runtime to become
//  .cu files: nvcc is slow, compiles once per target architecture, and its host
//  compiler support lags (concepts, ranges, and newer standard-library headers
//  are the usual casualties).
//
//  So the boundary is:
//     kernels/*.cu   : __global__ kernels + one thin `launch_xxx(...)` function
//                      per kernel, containing the <<<>>> call. Compiled by nvcc.
//     kernels.hpp    : ordinary C++ declarations of those launchers. Compiled by
//                      *any* compiler. Includes no CUDA headers except through
//                      stream.hpp's guarded ones.
//     src/graph/ops_*.cpp : Op::launch() implementations, compiled by the host
//                      compiler, calling the launchers.
//
//  Result: adding an op recompiles one .cu; changing the graph/scheduler
//  recompiles no CUDA at all. On Colab, where you rebuild constantly, this is
//  the difference between a 15-second and a 3-minute edit cycle.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — tile shape as a compile-time template parameter.
//
//  Tile sizes MUST be compile-time constants, not runtime arguments:
//    * shared-memory array sizes need constants (dynamic smem is possible but
//      loses the compiler's alias/offset knowledge),
//    * `#pragma unroll` on the inner K-loop needs a constant trip count, and
//      unrolling that loop is where register blocking gets its speed,
//    * the register-blocking accumulator array `float acc[TM][TN]` must be a
//      compile-time size to live in registers at all — a runtime size would
//      spill to local memory, which is DRAM, which defeats the entire point.
//  Hence `template <int BM, int BN, int BK, int TM, int TN>` and an explicit
//  instantiation list in the .cu. The cost is compile time and code size; the
//  benefit is that this is the only way the fast version exists.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#include "mcke/core/status.hpp"
#include "mcke/runtime/stream.hpp"

namespace mcke::kernels {

// ---------------------------------------------------------------------------
// Phase 0: the smoke test. Proves the toolchain, the launch path, and the
// event-timing path all work on a new machine before we build anything real.
// ---------------------------------------------------------------------------
[[nodiscard]] Status launch_vector_add_f32(const float* a, const float* b, float* out,
                                          std::size_t n, rt::StreamHandle stream);

// ---------------------------------------------------------------------------
// Phase 3a: fused bias + activation.  y[i] = act(x[i] + bias[i % cols])
// ---------------------------------------------------------------------------
enum class Activation : std::uint8_t { kNone, kRelu, kGeluErf, kGeluTanh };

// `vector_width` selects float / float2 / float4 loads. Wider loads issue fewer
// memory instructions for the same bytes (one 128-bit LDG instead of four
// 32-bit ones), which matters on a bandwidth-bound kernel because instruction
// issue, not bandwidth, can become the limiter. Requires the pointer AND the
// element count to be suitably aligned — our 256 B allocator alignment
// guarantees the pointer; the launcher checks n % 4 and falls back.
[[nodiscard]] Status launch_bias_act_f32(const float* x, const float* bias, float* y,
                                        std::int64_t rows, std::int64_t cols,
                                        Activation act, int vector_width,
                                        rt::StreamHandle stream);

// ---------------------------------------------------------------------------
// Phase 3b: reductions.
// ---------------------------------------------------------------------------
enum class ReduceKind : std::uint8_t { kSum, kMax, kMean };

// Row-wise reduce of a [rows, cols] matrix into [rows].
//   kSmemTree    : classic shared-memory tree, log2(blockDim) __syncthreads.
//   kWarpShuffle : __shfl_down_sync within a warp (no smem, no barrier), then
//                  one smem stage across warps. Fewer barriers, no bank
//                  conflicts, less smem => higher occupancy.
//   kTwoPass     : for very long rows; partials in global memory.
enum class ReduceVariant : std::uint8_t { kSmemTree, kWarpShuffle, kTwoPass };

[[nodiscard]] Status launch_row_reduce_f32(const float* x, float* out,
                                          std::int64_t rows, std::int64_t cols,
                                          ReduceKind kind, ReduceVariant variant,
                                          rt::StreamHandle stream);

// ---------------------------------------------------------------------------
// Phase 3c: row-wise softmax.
// ---------------------------------------------------------------------------
enum class SoftmaxVariant : std::uint8_t { kThreePass, kOnlineOnePass };

[[nodiscard]] Status launch_row_softmax_f32(const float* x, float* y,
                                           std::int64_t rows, std::int64_t cols,
                                           SoftmaxVariant variant,
                                           rt::StreamHandle stream);

// ---------------------------------------------------------------------------
// Phase 3d: GEMM.  C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
//
// Variants, in the order we build them — each one exists to isolate ONE idea,
// so that the benchmark table has a clean attribution for every speedup:
//   kNaive         : one thread per output element, reads A row + B col from
//                    global memory. AI = 2*K flops / (2*K*4 bytes) = 0.25.
//                    Hopelessly memory-bound; expect ~2-4% of peak.
//   kTiledSmem     : BMxBN output tile per block, staged through shared memory.
//                    Each element of A is now read N/BN times instead of N.
//                    AI rises to ~BK*... ; expect 15-25% of peak.
//   kTiledRegBlock : each thread computes a TMxTN micro-tile in registers.
//                    This is the big one: it raises reuse *within* a thread, so
//                    the ratio of FFMA to LDS instructions goes up. Expect
//                    45-65% of peak.
//   kWarpTile      : adds a warp-level tile layer + double-buffered smem
//                    (prefetch tile k+1 while computing tile k), removing the
//                    __syncthreads bubble. Expect 60-80%.
//   kCublasRef     : cuBLAS, as the ceiling. Not cheating — knowing the gap is
//                    the only way to know whether 65% is good.
// ---------------------------------------------------------------------------
enum class GemmVariant : std::uint8_t { kNaive, kTiledSmem, kTiledRegBlock, kWarpTile, kCublasRef };

struct GemmTile {
  int bm = 128, bn = 128, bk = 8;   // block tile
  int tm = 8,   tn = 8;             // per-thread (register) micro-tile
  // threads per block = (bm/tm) * (bn/tn); with the defaults, 16*16 = 256.
  // Shared memory per block = (bm*bk + bk*bn) * 4 B * (2 if double-buffered).
  // With the defaults: (128*8 + 8*128)*4 = 8 KiB, x2 = 16 KiB — leaves room for
  // 3 concurrent blocks per SM inside a 64 KiB smem budget. These two formulas
  // are the entire occupancy calculation, and they belong next to the numbers.
};

[[nodiscard]] Status launch_gemm_f32(const float* a, const float* b, float* c,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, float beta,
                                    GemmVariant variant, GemmTile tile,
                                    rt::StreamHandle stream);

}  // namespace mcke::kernels
