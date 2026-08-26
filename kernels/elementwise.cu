// =============================================================================
//  kernels/elementwise.cu
//
//  WHY .cu: this file contains a __global__ function and a <<<>>> launch. Both
//  are CUDA-C++ extensions that only nvcc (or clang in CUDA mode) understands.
//  The .cu extension is how nvcc knows to run the device compiler over it and
//  emit both host code and PTX/SASS for each target architecture.
//
//  Phase 0 content: vector add. Its purpose is not usefulness — it is to prove,
//  on each new machine, that (1) nvcc found a GPU architecture it can target,
//  (2) the runtime can launch a kernel, (3) event timing works, (4) our Status
//  plumbing surfaces real CUDA errors. Every one of those has failed for me on a
//  fresh machine at some point; a 30-line smoke test finds it in seconds instead
//  of inside a 500-line GEMM.
// =============================================================================
#include "mcke/kernels/kernels.hpp"
#include "mcke/runtime/cuda_check.hpp"

namespace mcke::kernels {
namespace {

// -----------------------------------------------------------------------------
// Grid-stride loop.
//
// WHY NOT the "one thread per element" form (`if (i < n) out[i] = ...`)?
//   * It forces grid size to depend on n, so a huge n can exceed the max grid
//     dimension, and a small n launches a tiny grid that cannot fill the GPU.
//   * It gives the compiler no loop to unroll and no opportunity to reuse
//     per-thread setup across elements.
//   * A grid-stride loop lets us pick the grid size for *occupancy* (e.g. enough
//     blocks to fill every SM a few times over) independently of the problem
//     size. That decoupling is why every CUDA sample written after ~2013 uses
//     this form.
//
// WHY the stride is `blockDim.x * gridDim.x` (the whole grid, not the block):
// consecutive threads must touch consecutive addresses *on every iteration* so
// each warp's 32 loads coalesce into 4 x 32-byte sectors. Striding by blockDim
// alone would make different blocks revisit the same lines; striding by the full
// grid keeps each warp's access a contiguous 128-byte span.
//
// __restrict__ on all three pointers tells nvcc that a, b and out do not alias.
// Without it the compiler must assume a store to out[i] could modify a[i+1],
// which blocks it from batching loads ahead of stores. On memory-bound
// elementwise kernels this single keyword is routinely worth 10-20%.
// -----------------------------------------------------------------------------
__global__ void vector_add_f32_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      std::size_t n) {
  // 64-bit index arithmetic: with n > 2^31 elements a 32-bit index silently
  // overflows. The cost is a few extra registers; the alternative is corruption
  // on large tensors.
  const std::size_t tid    = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t i = tid; i < n; i += stride) {
    out[i] = a[i] + b[i];   // 1 FLOP per 12 bytes moved -> AI = 0.083: pure
                            // bandwidth test. Peak achievable is ~ (2/3) of the
                            // STREAM triad rate; do not expect FLOPs here.
  }
}

}  // namespace

Status launch_vector_add_f32(const float* a, const float* b, float* out,
                             std::size_t n, rt::StreamHandle stream) {
  if (n == 0) return OkStatus();               // launching an empty grid is an error
  if (!a || !b || !out) return InvalidArgumentError("launch_vector_add_f32: null pointer");

  // 256 threads/block: a multiple of the 32-wide warp (so no partially-filled
  // warps waste issue slots) and small enough that several blocks fit per SM,
  // which is what lets the scheduler hide memory latency by switching warps.
  // 128-512 is the usual sweet spot; 1024 often *hurts* because it caps the
  // number of resident blocks per SM at 1-2.
  constexpr int kThreads = 256;

  // Cap the grid so we launch roughly "enough blocks to keep every SM busy with
  // several waves" rather than one block per element. 4096 blocks * 256 threads
  // = 1M threads in flight, which saturates any current GPU; the grid-stride
  // loop handles the rest.
  const std::size_t blocks_needed = (n + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks_needed < 4096 ? blocks_needed : 4096);

  vector_add_f32_kernel<<<blocks, kThreads, /*dynamic smem=*/0, stream>>>(a, b, out, n);

  // Catch launch-configuration errors immediately. Execution errors (illegal
  // address etc.) surface later; MCKE_CUDA_CHECK_LAUNCH in debug builds forces
  // them out here instead. See cuda_check.hpp for why the two are different.
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace mcke::kernels
