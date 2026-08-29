// =============================================================================
//  kernels/reduce_ops.cuh
//
//  WHAT: The reduction operators and the warp/block combine primitives shared by
//        kernels/reduce.cu and kernels/softmax.cu.
//
//  WHY .cuh: it contains __device__ code, so only nvcc can compile it
//  (CLAUDE.md section 5). It therefore lives in kernels/ and NOT in include/mcke/,
//  because everything under include/mcke/ must compile with plain clang++ on a
//  machine with no CUDA. This header is never installed and never reachable from
//  a .hpp.
//
//  WHY A HEADER AT ALL, rather than a second translation unit:
//  CMAKE_CUDA_SEPARABLE_COMPILATION is OFF, deliberately -- the build note says
//  "we keep every kernel + its device helpers in one .cu translation unit
//  instead", because separable compilation forces a device-link step and gives
//  up cross-TU inlining of exactly this kind of helper. So a shared __device__
//  function must be TEXTUALLY included. Each including .cu gets its own inlined
//  copy; with internal linkage there is no ODR problem and no device link, which
//  is precisely what that build setting buys in exchange for the duplication.
// =============================================================================
#pragma once

#include <cfloat>

namespace mcke::kernels {
namespace {

// -----------------------------------------------------------------------------
// Operators.
//
// Two functors, not three. `kMean` is `kSum` with a single multiply applied by
// the one thread that stores the row's result -- one extra multiply per ROW
// (8192 of them), not per ELEMENT (33.5 million of them). So the three public
// ReduceKinds cost two instantiations, and "how do the kinds share code" has the
// slightly surprising answer: two of them are literally the same kernel.
// -----------------------------------------------------------------------------
struct SumOp {
  static __device__ __forceinline__ float identity() { return 0.0f; }
  static __device__ __forceinline__ float combine(float a, float b) { return a + b; }
};

struct MaxOp {
  // -FLT_MAX, NOT -INFINITY, and this is load-bearing rather than stylistic.
  //
  // For max alone, -INFINITY would be fine: fmaxf(-inf, x) == x. But the SAME
  // identity convention is used by the online-softmax combine in softmax.cu,
  // which computes d_a * expf(m_a - m). When two empty partials meet, that is
  //     0 * expf(-inf - (-inf)) = 0 * expf(NaN) = NaN
  // and one NaN poisons the entire block reduction. With -FLT_MAX the same
  // expression underflows cleanly to 0 * 0 = 0.
  //
  // This is only reachable when cols < blockDim.x, i.e. when some threads have
  // no data at all -- which is exactly why the validation shape list contains
  // cols = 1 and cols = 17. One convention, one comment, one place to get right.
  static __device__ __forceinline__ float identity() { return -FLT_MAX; }
  static __device__ __forceinline__ float combine(float a, float b) { return fmaxf(a, b); }
};

// -----------------------------------------------------------------------------
// Intra-warp combine. Zero barriers and zero shared memory: the 32 lanes are
// combined through the register-file crossbar, never through memory.
//
// THE MASK. Before Volta a warp had ONE program counter, so every lane was
// guaranteed to be at the same instruction and __shfl_down needed no mask. Volta
// and later (sm_75 included) have independent thread scheduling: each lane has
// its own PC, diverged lanes may be scheduled at genuinely different times, and
// they may never reconverge on their own. So the _sync intrinsics take an
// explicit PARTICIPATION MASK -- the caller's declaration of which lanes will
// execute this exact instruction. The hardware converges the named lanes, then
// exchanges.
//
// Getting it wrong is silent: a lane named in the mask that never arrives is
// undefined behaviour, and a lane that arrives without being named returns an
// undefined value. No error, just a plausible wrong number under some occupancy.
//
// 0xffffffff is correct HERE specifically because this function is only ever
// called from a point where all 32 lanes of the warp are unconditionally
// executing -- never from inside a divergent branch. That is the invariant; if
// you call this from inside an `if (threadIdx.x < n)`, the mask is a lie.
// -----------------------------------------------------------------------------
template <class Op>
__device__ __forceinline__ float warp_reduce(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = Op::combine(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;   // lane 0 holds the warp's result
}

// -----------------------------------------------------------------------------
// Block-wide combine via one warp-reduce, one shared-memory stage, and a second
// warp-reduce. Exactly ONE __syncthreads.
//
// `smem` must have at least blockDim.x / 32 floats (8 at 256 threads = 32 bytes
// total -- compare the 1024 bytes a full smem tree needs).
//
// Note lanes >= n_warps are padded with Op::identity() rather than masked out of
// the second shuffle. Both work, but padding keeps the mask full and unchanging,
// so there is one fewer thing that has to stay in sync with the warp count --
// and it is the same reason MaxOp::identity() must not be -INFINITY.
// -----------------------------------------------------------------------------
template <class Op>
__device__ __forceinline__ float block_reduce(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  v = warp_reduce<Op>(v);                     // stage A: in-register, no barrier
  if (lane == 0) smem[warp] = v;
  __syncthreads();                            // the ONE barrier

  // Stage B: warp 0 alone combines the per-warp partials. The branch is
  // warp-uniform, so inside it all 32 lanes are active and the full mask holds.
  if (warp == 0) {
    v = (lane < n_warps) ? smem[lane] : Op::identity();
    v = warp_reduce<Op>(v);
  }
  return v;                                   // meaningful in thread 0 only
}

}  // namespace
}  // namespace mcke::kernels
