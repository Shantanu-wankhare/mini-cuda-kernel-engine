// =============================================================================
//  kernels/softmax.cu
//
//  WHAT: Phase 3c. Row-wise softmax, in two variants: the standard
//        three-pass numerically-stable form, and the online one-pass form.
//
//  WHY .cu: __global__ functions and <<<>>> launch sites.
//
//  ---------------------------------------------------------------------------
//  WHY SUBTRACTING THE ROW MAX IS MANDATORY, NOT AN OPTIMISATION
//
//  float maxes out at ~3.4e38 = e^88.7, so expf(x) overflows to +inf for any
//  x > 88.7, and inf/inf = NaN. Attention logits routinely exceed 88 before
//  scaling, so a single large value would NaN an entire row and the NaN then
//  propagates through every downstream layer.
//
//  Subtracting m = max(row) makes every exponent <= 0, so every expf lands in
//  (0, 1] and overflow becomes STRUCTURALLY IMPOSSIBLE. The identity
//  softmax(x) == softmax(x - m) means it costs nothing mathematically. It does
//  not prevent UNDERFLOW -- x - m < -87 gives expf -> 0 -- but that is the
//  correct answer to within a denormal, and harmless because the denominator is
//  at least 1 (the max element itself contributes e^0 = 1).
//
//  ---------------------------------------------------------------------------
//  "ONE PASS" NAMES THE STATISTICS PASSES, NOT THE MEMORY PASSES
//
//  This is the single most misread thing about the online algorithm, so state it
//  up front and predict accordingly:
//
//      kThreePass      reads x THREE times (max, sum, write), writes y once
//      kOnlineOnePass  reads x TWICE       (max+sum together, write), writes y once
//
//  It is one pass over the STATISTICS -- max and sum computed in a single
//  traversal instead of two. You still need x a second time to produce y. So the
//  honest predicted speedup is 4/3 = 1.33x, NOT 3x. Anyone predicting 3x from
//  the name will then have to explain away the shortfall.
//
//  Netting out likely L2 reuse (below), the realistic prediction is 1.0-1.25x.
//
//  ---------------------------------------------------------------------------
//  A CACHING EFFECT THAT MAY MASK THE RESULT -- ANTICIPATE IT, DO NOT EXCUSE IT
//
//  Each row is 4096 * 4 = 16 KiB. With ~160 resident blocks the collective
//  working set is ~2.5 MiB, which FITS inside the T4's 4 MiB L2. So three-pass's
//  2nd and 3rd reads may never reach DRAM at all, and it could beat its own
//  traffic model. If that happens it is the FINDING, not a measurement error --
//  and ncu's dram__bytes_read.sum compared against 3*N*4 settles which it was.
//
//  The variant that would make "one pass" literally true is to cache the row in
//  shared memory (16 KiB for 4096 floats) and produce y from the cache: 1R+1W, a
//  true 2x, and structurally what FlashAttention does with SRAM tiling.
//  DELIBERATELY NOT BUILT HERE, because it changes two variables at once
//  (statistics-pass count AND caching) and would destroy the attribution this
//  table exists to provide. Worth noting its occupancy cost for later: 16 KiB +
//  32 B per block means only 3 blocks fit in 64 KiB rather than 4, so a row
//  cache silently drops this chip from 100% to 75% occupancy.
// =============================================================================
#include "mcke/kernels/kernels.hpp"
#include "mcke/kernels/softmax_online.hpp"
#include "mcke/runtime/cuda_check.hpp"

#include "reduce_ops.cuh"

namespace mcke::kernels {
namespace {

constexpr int kThreads = 256;

// -----------------------------------------------------------------------------
// Block-wide combine for the online state. Same shape as block_reduce in
// reduce_ops.cuh, but carrying a PAIR through the shuffles rather than a scalar.
//
// The mask is 0xffffffff for the same reason documented there: these shuffles
// are never executed from inside a divergent branch. Stage B sits inside
// `if (warp == 0)`, which is warp-uniform, so all 32 lanes of warp 0 are active.
// -----------------------------------------------------------------------------
__device__ __forceinline__ OnlineState online_warp_reduce(OnlineState s) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    OnlineState o;
    o.m = __shfl_down_sync(0xffffffffu, s.m, off);
    o.d = __shfl_down_sync(0xffffffffu, s.d, off);
    s = online_combine(s, o);
  }
  return s;
}

__device__ __forceinline__ OnlineState online_block_reduce(OnlineState s,
                                                           OnlineState* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  s = online_warp_reduce(s);
  if (lane == 0) smem[warp] = s;
  __syncthreads();
  if (warp == 0) {
    // Lanes past the warp count are padded with the IDENTITY rather than masked
    // out, so the shuffle mask stays full and unchanging. This is exactly the
    // case where online_identity()'s -FLT_MAX (not -INFINITY) matters: with
    // -INFINITY these padding lanes would produce 0 * expf(NaN) = NaN and poison
    // the whole reduction.
    s = (lane < n_warps) ? smem[lane] : online_identity();
    s = online_warp_reduce(s);
  }
  return s;                                  // meaningful in thread 0 only
}

// -----------------------------------------------------------------------------
// Variant 1: three-pass.
// -----------------------------------------------------------------------------
__global__ void softmax_three_pass_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          std::int64_t cols) {
  __shared__ float smem[kThreads / 32];
  __shared__ float bcast;                    // separate from smem: block_reduce
                                             // is still using smem[0..7]
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * cols;

  // Pass 1 -- the row max.
  float m = MaxOp::identity();
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    m = fmaxf(m, x[base + c]);
  m = block_reduce<MaxOp>(m, smem);
  if (threadIdx.x == 0) bcast = m;
  __syncthreads();
  m = bcast;
  __syncthreads();                           // before smem is reused below

  // Pass 2 -- sum of exp. Re-reads x from global.
  float s = 0.0f;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    s += expf(x[base + c] - m);
  s = block_reduce<SumOp>(s, smem);
  if (threadIdx.x == 0) bcast = s;
  __syncthreads();
  // One divide per ROW rather than one per element: a float divide is ~4x a
  // multiply, and at 33.5M elements that is worth doing even on a kernel whose
  // limiter is memory.
  const float inv_s = 1.0f / bcast;

  // Pass 3 -- write. Re-reads x and recomputes expf with the SAME m, so the
  // numerator and denominator are bit-consistent (see the numerics note in the
  // online variant, where they are not).
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    y[base + c] = expf(x[base + c] - m) * inv_s;
}

// -----------------------------------------------------------------------------
// Variant 2: online one-pass over the statistics.
// -----------------------------------------------------------------------------
__global__ void softmax_online_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      std::int64_t cols) {
  __shared__ OnlineState smem[kThreads / 32];
  __shared__ OnlineState bcast;
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * cols;

  // Pass 1 -- max AND sum, in ONE traversal of the row.
  OnlineState st = online_identity();
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    st = online_update(st, x[base + c]);

  st = online_block_reduce(st, smem);         // the same associative operator
  if (threadIdx.x == 0) bcast = st;
  __syncthreads();
  st = bcast;
  const float inv_d = 1.0f / st.d;

  // Pass 2 -- write.
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    y[base + c] = expf(x[base + c] - st.m) * inv_d;
}

}  // namespace

Status launch_row_softmax_f32(const float* x, float* y,
                              std::int64_t rows, std::int64_t cols,
                              SoftmaxVariant variant, rt::StreamHandle stream) {
  if (rows == 0 || cols == 0) return OkStatus();
  if (rows < 0 || cols < 0)
    return InvalidArgumentError("launch_row_softmax_f32: negative extent");
  if (!x || !y) return InvalidArgumentError("launch_row_softmax_f32: null pointer");
  // __restrict__ on both pointers is a promise they do not alias; an in-place
  // call would break it silently rather than loudly.
  if (x == y)
    return InvalidArgumentError("launch_row_softmax_f32: in-place (x == y) violates "
                                "the __restrict__ contract");
  if (rows > 2147483647LL)
    return InvalidArgumentError("launch_row_softmax_f32: rows exceeds gridDim.x limit");

  const unsigned grid = static_cast<unsigned>(rows);
  if (variant == SoftmaxVariant::kThreePass)
    softmax_three_pass_kernel<<<grid, kThreads, /*dynamic smem=*/0, stream>>>(x, y, cols);
  else
    softmax_online_kernel<<<grid, kThreads, /*dynamic smem=*/0, stream>>>(x, y, cols);
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace mcke::kernels
