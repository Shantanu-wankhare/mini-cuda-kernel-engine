// =============================================================================
//  kernels/bias_act.cu
//
//  WHAT: Phase 3a. `y = act(x + bias)` fused into one kernel, plus the two
//        separate kernels that do it unfused, so the fusion speedup has an
//        honest denominator.
//
//  WHY .cu: contains __global__ functions and <<<>>> launch sites.
//
//  ---------------------------------------------------------------------------
//  WHY FUSION IS THE CHEAPEST LARGE WIN IN THE WHOLE PROJECT
//
//  Unfused, this is two kernels, each reading and writing the entire tensor:
//        k1: read x, read bias, write tmp
//        k2: read tmp,          write y
//  Fused, it is one kernel that reads x once and writes y once. With
//  N = rows*cols elements:
//
//        unfused bytes = (4N + cols) * 4        fused bytes = (2N + cols) * 4
//
//  Exactly half the traffic. On a kernel whose arithmetic intensity is between
//  0.125 and 2.75 FLOP/byte -- against this GPU's ridge point of 34.2 -- time is
//  proportional to bytes moved and nothing else. So halving traffic should halve
//  the time, and that is the entire prediction.
//
//  The bias term is 0.006% of the total and is in the formula only because
//  RESULTS.md rule 4 requires the ideal count to be reconstructible. It is also
//  an over-count of real DRAM traffic rather than an under-count: every block
//  reads the same 16 KiB bias vector, so after the first touch it lives in L2.
//
//  Not fusing also costs a full extra 128 MiB intermediate tensor. That is the
//  memory-planner argument Phase 4 will make, and it is measured here.
//
//  ---------------------------------------------------------------------------
//  WHY A 2-D GRID, AND NOT THE OBVIOUS FLAT GRID-STRIDE LOOP
//
//  The natural port of elementwise.cu is a flat loop over N with
//  `col = i % cols`. That is wrong here, and by a large margin rather than a
//  pedantic one:
//
//  There is no hardware 64-bit integer divide on any NVIDIA GPU. A `%` by a
//  RUNTIME value compiles to a ~10-instruction reciprocal sequence. Per SM at
//  8192x4096: 33.5M/40 = 838K elements, at width 4 that is 210K iterations x 10
//  instructions = 2.1M instructions, at ~4 IPC ~= 525K cycles ~= 0.33 ms --
//  against a memory floor of 1.14 ms. **29% of the time budget spent computing
//  an index.**
//
//  Instead: `col` comes from blockIdx.x (pure multiply-add, no division ever),
//  and `row` from a blockIdx.y grid-stride. Two further properties fall out:
//    * the bias load is loop-INVARIANT in col, so it hoists out of the row loop
//      entirely -- each thread reads bias once for the whole kernel rather than
//      once per element. That is the structural reason the bias term is
//      negligible, expressed in code rather than argued in a comment.
//    * coalescing is unchanged: lane t reads base + t*VW, so a warp still covers
//      one contiguous 32*VW*4 byte span.
//
//  (The production alternative is magic-number division -- precompute a
//  (multiplier, shift) pair on the host since `cols` is launch-invariant, and
//  use __umulhi. It keeps the flat 1-D grid and is what real libraries do.
//  Rejected here because the 2-D grid gets the same result with no cleverness,
//  and a teaching kernel should not open with a Granlund-Montgomery derivation.)
//
//  ---------------------------------------------------------------------------
//  WHY vector_width IS PREDICTED NOT TO HELP HERE (and why we measure it anyway)
//
//  Wider loads issue fewer memory instructions for the same bytes. The usual
//  claim is that this matters because instruction issue, not bandwidth, becomes
//  the limiter. On a T4 at full occupancy that claim is FALSE:
//    * the chip absorbs 235.4 GB/s / 1.59 GHz = 148 B/cycle = 3.7 B/cycle/SM;
//    * an SM issues ~4 memory instructions/cycle x 128 B = 512 B/cycle of
//      requests -- 138x more than DRAM can take.
//  Issue is nowhere near the limiter. Little's law agrees: sustaining
//  3.7 B/cycle at ~400 cycles of latency needs ~1480 B in flight per SM, and
//  1024 threads x 4 B of scalar load already gives 4096 B.
//
//  So: predict 0-10% at full occupancy, possibly inside the noise. The
//  experiment that makes that non-result informative is to run the SAME sweep
//  deliberately starved -- one block per SM, ~6% occupancy -- where the in-flight
//  budget collapses and 16 B/thread is the only lever left. Predict a large win
//  there. The pair of results is the real rule: **vectorised loads buy
//  memory-level parallelism, and MLP is only scarce when occupancy is scarce.**
// =============================================================================
#include "mcke/kernels/kernels.hpp"
#include "mcke/runtime/cuda_check.hpp"

namespace mcke::kernels {
namespace {

// -----------------------------------------------------------------------------
// Activation functors. Templated rather than switched inside the loop: a runtime
// branch would be perfectly predicted, so it is not a *branching* cost -- but it
// is one extra instruction per element on a loop body of three or four, and it
// blocks the compiler from contracting the add+activate into a single FFMA.
// -----------------------------------------------------------------------------
struct ActNone {
  static __device__ __forceinline__ float apply(float v) { return v; }
};
struct ActRelu {
  // fmaxf, not `v > 0 ? v : 0`: one FMNMX instruction, no branch, and it gives
  // relu(-0.0f) == +0.0f which the ternary does not.
  static __device__ __forceinline__ float apply(float v) { return fmaxf(v, 0.0f); }
};
struct ActGeluErf {
  // The exact definition. erff is ~20 FP32 instructions on sm_75.
  static __device__ __forceinline__ float apply(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));   // 1/sqrt(2)
  }
};
struct ActGeluTanh {
  // Hendrycks & Gimpel's approximation: ~8 FP32 + 1-2 MUFU (tanhf lowers to
  // MUFU.EX2 plus a reciprocal, on a different issue port than the FP32 pipe,
  // so the two partly overlap). Differs from the exact form in the 3rd decimal.
  static __device__ __forceinline__ float apply(float v) {
    const float k = 0.79788456080286536f;                        // sqrt(2/pi)
    return 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
  }
};

// -----------------------------------------------------------------------------
// Vector load/store. The launcher guarantees cols % VW == 0 and 4*VW-byte
// alignment, so these are always full-width -- no partial-vector tail handling
// anywhere in the kernel. That simplification is exactly what the strict
// precondition buys, and it is why the launcher rejects rather than downgrades.
// -----------------------------------------------------------------------------
template <int VW>
__device__ __forceinline__ void load_vec(const float* __restrict__ p, float (&out)[VW]) {
  if constexpr (VW == 1) {
    out[0] = *p;
  } else if constexpr (VW == 2) {
    const float2 v = *reinterpret_cast<const float2*>(p);   // one 64-bit LDG
    out[0] = v.x; out[1] = v.y;
  } else {
    const float4 v = *reinterpret_cast<const float4*>(p);   // one 128-bit LDG
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
  }
}

template <int VW>
__device__ __forceinline__ void store_vec(float* __restrict__ p, const float (&in)[VW]) {
  if constexpr (VW == 1) {
    *p = in[0];
  } else if constexpr (VW == 2) {
    *reinterpret_cast<float2*>(p) = make_float2(in[0], in[1]);
  } else {
    *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
  }
}

// -----------------------------------------------------------------------------
// The fused kernel.
// -----------------------------------------------------------------------------
template <int VW, class ActFn>
__global__ void bias_act_kernel(const float* __restrict__ x,
                                const float* __restrict__ bias,
                                float* __restrict__ y,
                                std::int64_t rows, std::int64_t cols) {
  // col is pure multiply-add. No integer division anywhere in this kernel --
  // see the banner for what the obvious `i % cols` version would have cost.
  const std::int64_t col =
      (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VW;
  if (col >= cols) return;          // warp-uniform for all but the last block

  // Hoisted out of the row loop: col is loop-invariant, so each thread touches
  // the bias vector exactly ONCE for the entire kernel rather than once per
  // element. This is the structural reason the bias byte count is negligible.
  float b[VW];
  load_vec<VW>(bias + col, b);

  // Grid-stride over rows. gridDim.y is capped at 65535 by hardware, which the
  // int64_t `rows` in the frozen signature quietly invites you to exceed -- so
  // the stride is mandatory, not an optimisation.
  for (std::int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const std::int64_t base = row * cols + col;
    float v[VW];
    load_vec<VW>(x + base, v);
    #pragma unroll
    for (int k = 0; k < VW; ++k) v[k] = ActFn::apply(v[k] + b[k]);
    store_vec<VW>(y + base, v);
  }
}

// -----------------------------------------------------------------------------
// The activation-only kernel — second half of the unfused baseline. No bias, so
// no per-column broadcast, so a flat grid-stride over n is correct here (and the
// modulo objection from the banner does not apply: there is no `% cols`).
// -----------------------------------------------------------------------------
template <int VW, class ActFn>
__global__ void activation_kernel(const float* __restrict__ x,
                                  float* __restrict__ y, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x * VW;
  std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VW;
  for (; i + VW <= n; i += stride) {
    float v[VW];
    load_vec<VW>(x + i, v);
    #pragma unroll
    for (int k = 0; k < VW; ++k) v[k] = ActFn::apply(v[k]);
    store_vec<VW>(y + i, v);
  }
}

constexpr int kThreads = 256;

// Shared validation for the two bias-taking launchers. The precondition is on
// COLS, not on the total element count: row r starts at element r*cols, so a
// float4 load needs every row start 16 B aligned, which requires cols to be a
// multiple of the width. (rows=4, cols=3 has n=12 divisible by 4 while every odd
// row start is misaligned -- an earlier version of kernels.hpp got this wrong.)
Status validate_2d(const char* who, const float* x, const float* y,
                   std::int64_t rows, std::int64_t cols, int vw) {
  if (!x || !y) return InvalidArgumentError(std::string(who) + ": null pointer");
  if (rows < 0 || cols < 0)
    return InvalidArgumentError(std::string(who) + ": negative extent");
  if (vw != 1 && vw != 2 && vw != 4)
    return InvalidArgumentError(std::string(who) + ": vector_width must be 1, 2 or 4, got " +
                                std::to_string(vw));
  if (cols % vw != 0)
    return InvalidArgumentError(std::string(who) + ": cols (" + std::to_string(cols) +
                                ") must be a multiple of vector_width (" +
                                std::to_string(vw) + "). Use max_vector_width_f32() "
                                "if you want a fallback -- this refuses rather than "
                                "silently running at width 1, because the signature "
                                "has no way to tell you it downgraded.");
  const std::size_t align = sizeof(float) * static_cast<std::size_t>(vw);
  if (reinterpret_cast<std::uintptr_t>(x) % align != 0 ||
      reinterpret_cast<std::uintptr_t>(y) % align != 0)
    return InvalidArgumentError(std::string(who) + ": pointer not " +
                                std::to_string(align) + "-byte aligned");
  return OkStatus();
}

// Grid for the 2-D (bias-taking) kernels.
dim3 grid_2d(std::int64_t rows, std::int64_t cols, int vw, int max_row_blocks) {
  const std::int64_t cols_vec = cols / vw;
  const unsigned gx = static_cast<unsigned>((cols_vec + kThreads - 1) / kThreads);
  // Hardware caps gridDim.y at 65535; the kernel grid-strides over the rest.
  std::int64_t gy64 = rows < 65535 ? rows : 65535;
  // Deliberate starvation for the occupancy experiment -- see kernels.hpp.
  if (max_row_blocks > 0 && gy64 > max_row_blocks) gy64 = max_row_blocks;
  const unsigned gy = static_cast<unsigned>(gy64);
  return dim3(gx ? gx : 1u, gy ? gy : 1u, 1u);
}

// Dispatch a runtime (vector_width, Activation) pair onto the instantiated
// templates. 3 widths x 4 activations = 12 instantiations; these kernels are
// tiny, so unlike the GEMM ladder the instantiation count costs nothing worth
// mitigating.
#define MCKE_DISPATCH_ACT(VW, LAUNCH)                                    \
  switch (act) {                                                         \
    case Activation::kNone:     LAUNCH(VW, ActNone);     break;          \
    case Activation::kRelu:     LAUNCH(VW, ActRelu);     break;          \
    case Activation::kGeluErf:  LAUNCH(VW, ActGeluErf);  break;          \
    case Activation::kGeluTanh: LAUNCH(VW, ActGeluTanh); break;          \
  }

#define MCKE_DISPATCH_VW(LAUNCH)                                         \
  switch (vector_width) {                                                \
    case 1: MCKE_DISPATCH_ACT(1, LAUNCH); break;                         \
    case 2: MCKE_DISPATCH_ACT(2, LAUNCH); break;                         \
    case 4: MCKE_DISPATCH_ACT(4, LAUNCH); break;                         \
    default: return InvalidArgumentError("bias_act: bad vector_width");   \
  }

}  // namespace

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------

Status launch_bias_act_f32(const float* x, const float* bias, float* y,
                           std::int64_t rows, std::int64_t cols,
                           Activation act, int vector_width, rt::StreamHandle stream,
                           int max_row_blocks) {
  if (rows == 0 || cols == 0) return OkStatus();     // an empty grid is a launch error
  if (!bias) return InvalidArgumentError("launch_bias_act_f32: null bias");
  MCKE_RETURN_IF_ERROR(validate_2d("launch_bias_act_f32", x, y, rows, cols, vector_width));
  {
    const std::size_t align = sizeof(float) * static_cast<std::size_t>(vector_width);
    if (reinterpret_cast<std::uintptr_t>(bias) % align != 0)
      return InvalidArgumentError("launch_bias_act_f32: bias pointer not aligned");
  }
  // __restrict__ on x and y is a promise they do not alias. An in-place call
  // would break that promise silently, so refuse it rather than miscompile.
  if (x == y) return InvalidArgumentError("launch_bias_act_f32: in-place (x == y) "
                                          "violates the __restrict__ contract");

  const dim3 grid = grid_2d(rows, cols, vector_width, max_row_blocks);
#define LAUNCH_FUSED(VW, ACT)                                                   \
  bias_act_kernel<VW, ACT><<<grid, kThreads, /*dynamic smem=*/0, stream>>>(     \
      x, bias, y, rows, cols)
  MCKE_DISPATCH_VW(LAUNCH_FUSED);
#undef LAUNCH_FUSED
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status launch_bias_add_f32(const float* x, const float* bias, float* y,
                           std::int64_t rows, std::int64_t cols,
                           int vector_width, rt::StreamHandle stream) {
  // Bias-add IS the fused kernel with the identity activation, so it shares the
  // code rather than duplicating it. That also keeps the baseline structurally
  // identical to the thing it is a baseline for -- same grid, same block, same
  // indexing -- which is what makes the comparison fair.
  return launch_bias_act_f32(x, bias, y, rows, cols, Activation::kNone,
                             vector_width, stream);
}

Status launch_activation_f32(const float* x, float* y, std::int64_t n,
                             Activation act, int vector_width, rt::StreamHandle stream) {
  if (n == 0) return OkStatus();
  if (!x || !y) return InvalidArgumentError("launch_activation_f32: null pointer");
  if (vector_width != 1 && vector_width != 2 && vector_width != 4)
    return InvalidArgumentError("launch_activation_f32: vector_width must be 1, 2 or 4");
  if (n % vector_width != 0)
    return InvalidArgumentError("launch_activation_f32: n must be a multiple of "
                                "vector_width");
  const std::size_t align = sizeof(float) * static_cast<std::size_t>(vector_width);
  if (reinterpret_cast<std::uintptr_t>(x) % align != 0 ||
      reinterpret_cast<std::uintptr_t>(y) % align != 0)
    return InvalidArgumentError("launch_activation_f32: pointer not aligned");

  const std::int64_t n_vec = n / vector_width;
  const std::int64_t need  = (n_vec + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(need < 4096 ? need : 4096);
#define LAUNCH_ACT(VW, ACT)                                                     \
  activation_kernel<VW, ACT><<<blocks, kThreads, /*dynamic smem=*/0, stream>>>( \
      x, y, n)
  MCKE_DISPATCH_VW(LAUNCH_ACT);
#undef LAUNCH_ACT
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace mcke::kernels
