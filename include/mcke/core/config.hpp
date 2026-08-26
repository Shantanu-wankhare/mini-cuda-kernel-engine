// =============================================================================
//  mcke/core/config.hpp
//
//  WHAT: Compile-time configuration, portability macros, and the one place
//        where we decide "am I being compiled by nvcc or by a host compiler".
//  WHY .hpp: pure C++ declarations/macros, no device code. Anything a host-only
//        translation unit (clang++ on macOS) must be able to include lives in a
//        .hpp. Files that contain __global__/__device__ code and must be fed to
//        nvcc use .cuh (headers) or .cu (sources).
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

// -----------------------------------------------------------------------------
// MCKE_WITH_CUDA — a *build-wide* switch, set by CMake.
//
// Do not confuse it with __CUDACC__:
//   MCKE_WITH_CUDA : "this build has a GPU backend at all"      (per project)
//   __CUDACC__     : "nvcc is compiling this file right now"    (per TU)
// A .cpp file in a CUDA-enabled build sees MCKE_WITH_CUDA==1 but no __CUDACC__.
// Getting these two mixed up is the #1 source of confusing link errors in
// mixed C++/CUDA projects, so we name them differently on purpose.
// -----------------------------------------------------------------------------
#ifndef MCKE_WITH_CUDA
#  define MCKE_WITH_CUDA 0
#endif

// -----------------------------------------------------------------------------
// Function-space qualifiers.
//
// Small math helpers (shape indexing, buddy-tree arithmetic) are wanted in both
// spaces: on the host for planning, on the device for index computation. Rather
// than duplicate them we tag them MCKE_HOST_DEVICE, which expands to nothing
// when a host compiler is reading the header — so the same header stays valid
// C++ for clang++ on the Mac.
// -----------------------------------------------------------------------------
#if defined(__CUDACC__)
#  define MCKE_HOST_DEVICE   __host__ __device__
#  define MCKE_DEVICE        __device__
#  define MCKE_HOST          __host__
#  define MCKE_FORCEINLINE   __forceinline__
#  define MCKE_GLOBAL        __global__
#else
#  define MCKE_HOST_DEVICE
#  define MCKE_DEVICE
#  define MCKE_HOST
#  define MCKE_FORCEINLINE   inline
#  define MCKE_GLOBAL        // never valid on host; guarded by #if MCKE_WITH_CUDA
#endif

namespace mcke {

// -----------------------------------------------------------------------------
// Hardware constants we rely on architecturally.
//
// These are *not* "magic numbers": every NVIDIA GPU from Kepler through
// Blackwell has a 32-thread warp, and the warp size shows up in kernel design
// (shuffle reductions, tile shapes, coalescing width). We hard-code 32 and
// assert it at runtime rather than querying it everywhere, because a runtime
// warp size would make our tile constants non-constexpr and kill unrolling.
// -----------------------------------------------------------------------------
inline constexpr int kWarpSize = 32;

// Global memory transactions are serviced in 32-byte sectors / 128-byte cache
// lines. Coalescing analysis in Phase 3 is expressed in these units.
inline constexpr std::size_t kSectorBytes    = 32;
inline constexpr std::size_t kCacheLineBytes = 128;

// Default alignment for every device allocation handed out by our allocators.
// 256 B because:
//   - cudaMalloc itself guarantees 256 B alignment, and code (ours and cuBLAS')
//     silently assumes it;
//   - it is a multiple of 128, so a tensor's first element always starts on a
//     cache-line boundary — otherwise every coalesced load in a tiled GEMM
//     would straddle two lines and we would pay ~2x the transactions.
inline constexpr std::size_t kDeviceAlignment = 256;

// Smallest block our buddy allocator will ever hand out. Below this, internal
// fragmentation is irrelevant and metadata dominates.
inline constexpr std::size_t kMinBlockBytes = 256;

}  // namespace mcke
