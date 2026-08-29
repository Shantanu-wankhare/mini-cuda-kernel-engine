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
// 32-bit ones). The usual claim is that this matters on a bandwidth-bound kernel
// because instruction issue becomes the limiter before bandwidth does -- but on
// a T4 at 100% occupancy that is NOT true and the sweep is predicted to be
// flat-to-10%: the chip absorbs ~3.7 B/cycle/SM while an SM can issue ~512 B of
// requests per cycle, 138x more than DRAM can take. Vector width buys
// memory-level parallelism, and MLP is only scarce when OCCUPANCY is scarce --
// so the informative experiment runs the sweep twice, once at full occupancy
// (predict flat) and once deliberately starved (predict a large win).
//
// PRECONDITION, and an earlier version of this comment got it wrong: the check
// is `cols % vector_width == 0`, NOT `n % 4`. Row r starts at element r*cols, so
// a float4 load needs every row start 16 B aligned, which requires cols (not the
// total element count) to be a multiple of the width. rows=4, cols=3 has n=12
// divisible by 4 while every odd row start is misaligned. Our 256 B allocator
// alignment guarantees the base pointer; the launcher REJECTS a bad width rather
// than silently downgrading, because the frozen signature has no channel to
// report "I ran at width 1 even though you asked for 4" and a benchmark that
// cannot tell you what it measured is worse than one that errors.
// `max_row_blocks` caps gridDim.y; 0 means "auto" (one block-row per matrix row,
// the normal full-occupancy launch). It exists for ONE experiment, and that
// experiment is what makes the vector_width result mean anything:
//
// At full occupancy the sweep is predicted to be flat, because instruction issue
// is 138x away from being the limiter. A flat result on its own reads as
// "vectorisation is pointless", which is the wrong lesson. Re-running the same
// sweep with the grid deliberately starved (~40 blocks, one per SM, ~6%
// occupancy) collapses the in-flight byte budget so that bytes-per-thread is the
// only lever left -- and a large win there establishes the actual rule:
// vectorised loads buy memory-level parallelism, and MLP is only scarce when
// occupancy is scarce. Two rows, one rule; neither row alone says it.
//
// The kernel already grid-strides over rows, so capping the grid changes nothing
// but the launch geometry.
[[nodiscard]] Status launch_bias_act_f32(const float* x, const float* bias, float* y,
                                        std::int64_t rows, std::int64_t cols,
                                        Activation act, int vector_width,
                                        rt::StreamHandle stream,
                                        int max_row_blocks = 0);

// --- The UNFUSED baseline, so that "fusion is ~2x" has an honest denominator ---
//
// These are not dead code and they are not a strawman. Two alternatives were
// rejected:
//   * Reusing launch_vector_add_f32 would need a materialised [rows, cols]
//     broadcast of the [cols] bias first -- a third full-size array plus a
//     broadcast kernel -- so the "unfused" path would move 5 units of traffic
//     instead of 4 and the 2x claim would be measured against something worse
//     than what a real unfused implementation does. Rejected on FAIRNESS.
//   * Defining them inside bench/ would put them outside mcke_kernels, where
//     -Xptxas=-v does not reach -- so the baseline's regs/smem would be
//     invisible while the fused kernel's were visible, comparing an instrumented
//     kernel against an uninstrumented one. And Phase 4 needs an unfused
//     two-node graph to show what a fusion pass buys; a kernel in bench/ is
//     unreachable from src/graph/.
//
// Fairness controls held constant across fused and unfused: same block size,
// same grid heuristic, same vector_width, same activation, same input buffer,
// same stream, same warmup/iters. The unfused timing brackets BOTH launches, so
// the second kernel's launch overhead is counted -- that is a genuine cost of
// not fusing and must not be excluded.
[[nodiscard]] Status launch_bias_add_f32(const float* x, const float* bias, float* y,
                                        std::int64_t rows, std::int64_t cols,
                                        int vector_width, rt::StreamHandle stream);

[[nodiscard]] Status launch_activation_f32(const float* x, float* y, std::int64_t n,
                                          Activation act, int vector_width,
                                          rt::StreamHandle stream);

// Largest legal width in {4,2,1} for this pointer and column count.
//
// The LAUNCHER rejects an illegal width rather than silently downgrading (see
// above); this is where the *policy* "pick the widest that works" lives, so the
// two concerns stay separate. Callers that want a fallback call this first.
// Pure host arithmetic, hence inline here -- usable from a host-only build.
[[nodiscard]] inline int max_vector_width_f32(const void* p, std::int64_t cols) noexcept {
  const auto addr = reinterpret_cast<std::uintptr_t>(p);
  for (int w = 4; w > 1; w >>= 1)
    if (cols % w == 0 && addr % (sizeof(float) * static_cast<std::size_t>(w)) == 0) return w;
  return 1;
}

// ---------------------------------------------------------------------------
// Phase 3b: reductions.
// ---------------------------------------------------------------------------
enum class ReduceKind : std::uint8_t { kSum, kMax, kMean };

// Row-wise reduce of a [rows, cols] matrix into [rows].
//   kSmemTree    : classic shared-memory tree. 1 + log2(blockDim) __syncthreads
//                  -- 9 at 256 threads, NOT 8. The load-into-smem barrier before
//                  the tree starts is a real barrier and must be counted; an
//                  earlier version of this comment said log2(blockDim) and was
//                  off by one. RESULTS.md has a column for this number, so it
//                  needs to be right.
//   kWarpShuffle : __shfl_down_sync within a warp (no smem, no barrier), then
//                  one smem stage across warps. Exactly 1 __syncthreads. Fewer
//                  barriers, no bank conflicts, less smem => higher occupancy --
//                  though note the third reason is worth ~nothing at 256 threads
//                  on a T4, where the 1024-threads-per-SM cap binds before smem
//                  does. It becomes real with bigger blocks or when a reduction
//                  is fused into a kernel that already uses smem.
//   kTwoPass     : partials staged through global memory.
//
//                  NOT "for very long rows" -- that was the original rationale
//                  here and it is wrong. A grid-stride loop over columns handles
//                  a row of ANY length in one block; length is never the problem.
//                  The real trigger is too FEW ROWS to fill the machine: with one
//                  block per row, 8192 rows on 40 SMs is ~51 waves (saturated,
//                  kTwoPass buys nothing and costs ~1-3%), but 64 rows is 0.4
//                  waves with 24 SMs idle, and splitting each row across blocks
//                  is the only way to fill them. Benchmark BOTH shapes or the
//                  variant looks like it only ever loses.
enum class ReduceVariant : std::uint8_t { kSmemTree, kWarpShuffle, kTwoPass };

// How much global scratch `variant` needs for this shape. Zero for kSmemTree and
// kWarpShuffle; kTwoPass needs rows * split * sizeof(float) to stage its partial
// results between the two passes.
//
// WHY A QUERY + AN EXPLICIT BUFFER, rather than letting the launcher allocate:
// a cudaMalloc inside the launcher would sit inside the benchmark's timed loop,
// and Phase 2 measured raw cudaMalloc on this hardware at a 720 us maximum with
// a cudaFree that SYNCHRONISES THE DEVICE (see RESULTS.md section 2a and the
// banner of memory/allocator.hpp). Putting that inside a ~570 us kernel would
// make the kTwoPass row a measurement of the driver, and the synchronising free
// would wreck the async model Phase 4 is built on. Hidden static scratch was the
// other option and is worse: silent global state in a runtime whose entire
// thesis is explicit memory management.
//
// This mirrors Op::workspace_bytes() in graph/op.hpp, which already anticipated
// exactly this pattern -- the graph layer had it and the kernel layer did not.
[[nodiscard]] std::size_t row_reduce_workspace_bytes(std::int64_t rows, std::int64_t cols,
                                                    ReduceVariant variant) noexcept;

// The frozen 7-argument form. Kept verbatim so nothing that already calls it
// breaks; it delegates to the overload below with a null workspace, and returns
// UnimplementedError if kTwoPass is requested without one.
[[nodiscard]] Status launch_row_reduce_f32(const float* x, float* out,
                                          std::int64_t rows, std::int64_t cols,
                                          ReduceKind kind, ReduceVariant variant,
                                          rt::StreamHandle stream);

[[nodiscard]] Status launch_row_reduce_f32(const float* x, float* out,
                                          std::int64_t rows, std::int64_t cols,
                                          ReduceKind kind, ReduceVariant variant,
                                          float* workspace, std::size_t workspace_bytes,
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
// kWarpTileNoDbuf is APPENDED, after kCublasRef, deliberately: appending leaves
// every existing enumerator's std::uint8_t value unchanged, so nothing that has
// already been recorded or serialised shifts meaning.
//
// It exists because kWarpTile bundles TWO independent changes -- a warp-level
// tile layer AND double-buffered shared memory -- which would give its results
// row two attributable causes and break the one-variable-per-step rule the whole
// table depends on. The two effects are separable and predicted to be comparable
// in size: warp tiling removes a bank conflict on the B-fragment load, double
// buffering halves the barriers and hides DRAM latency behind the next tile's
// load. Measuring their sum and calling it one number would hide which one
// mattered.
enum class GemmVariant : std::uint8_t {
  kNaive, kTiledSmem, kTiledRegBlock, kWarpTile, kCublasRef,
  kWarpTileNoDbuf,      // warp tiling WITHOUT double buffering -- the isolator
};

struct GemmTile {
  int bm = 128, bn = 128, bk = 8;   // block tile
  int tm = 8,   tn = 8;             // per-thread (register) micro-tile
  // threads per block = (bm/tm) * (bn/tn); with the defaults, 16*16 = 256.
  // Shared memory per block = (bm*bk + bk*bn) * 4 B * (2 if double-buffered).
  // With the defaults: (128*8 + 8*128)*4 = 8 KiB, x2 = 16 KiB.
  //
  // An earlier version of this comment then said "leaves room for 3 concurrent
  // blocks per SM" and called these "the entire occupancy calculation". Both
  // claims are wrong, and the second one is wrong in the way this whole phase
  // exists to correct:
  //   * 64 KiB / 16 KiB = 4 blocks, not 3.
  //   * Occupancy has FOUR limiters, not one: shared memory, registers, the
  //     1024-threads-per-SM cap (=> 4 blocks at 256 threads), and the 32-resident
  //     -block cap. The answer is the MINIMUM of all four.
  //   * On this kernel shared memory is NOT the binding one. A register-blocked
  //     GEMM holding float acc[8][8] needs ~128+ registers/thread; the 65536
  //     registers per SM then cap it at 2 blocks -- half what smem allows.
  // Identifying the obvious constraint and finding it is not the binding one is
  // the actual lesson; see LEARNING_LOG.md.
};

[[nodiscard]] Status launch_gemm_f32(const float* a, const float* b, float* c,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, float beta,
                                    GemmVariant variant, GemmTile tile,
                                    rt::StreamHandle stream);

}  // namespace mcke::kernels
