// =============================================================================
//  scripts/fakecuda/cuda_lang_prelude.h
//
//  Fakes the CUDA *language* extensions so a .cu file's HOST logic can be
//  type-checked by plain clang++ on a machine with no nvcc.
//
//  WHAT THIS PROVES, AND WHAT IT DOES NOT:
//    proves  — signatures, include chains, launcher control flow, dispatch
//              tables, host-side argument marshalling, and that the
//              MCKE_WITH_CUDA=1 branches of every header actually compile.
//    does NOT — anything about device code generation. It says nothing about
//              whether a kernel is correct, whether loads get hoisted, whether
//              registers spill, or what SASS comes out. Those need real nvcc.
//
//  It exists because the alternative is discovering a missing include or a
//  wrong signature *on Colab*, burning a session on something a laptop could
//  have caught. Phase 2 used this to make its first GPU session compile clean
//  on the first try; Phase 3 has four kernels' worth of the same risk.
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(...)

// dim3 is a real CUDA type (used in host code to build launch geometry), so it
// must exist here even though the <<<>>> that consumes it gets stripped.
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
static dim3 blockIdx, threadIdx, gridDim, blockDim;

inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_block() {}
inline long long clock64() { return 0; }

// Warp intrinsics (Phase 3b: the shuffle reduction).
inline float    __shfl_down_sync(unsigned, float v, unsigned, int = 32) { return v; }
inline float    __shfl_sync(unsigned, float v, int, int = 32) { return v; }
inline float    __shfl_xor_sync(unsigned, float v, int, int = 32) { return v; }
inline unsigned __activemask() { return 0xFFFFFFFFu; }
inline unsigned __ballot_sync(unsigned, int) { return 0u; }

// Cache-control loads (Phase 2d used __ldcg; GEMM may use __ldg).
inline float __ldg(const float* p)  { return *p; }
inline float __ldcg(const float* p) { return *p; }
inline float __ldcv(const float* p) { return *p; }

// Atomics.
inline unsigned atomicAdd(unsigned* p, unsigned v) { unsigned o = *p; *p += v; return o; }
inline int      atomicAdd(int* p, int v)           { int o = *p; *p += v; return o; }
inline float    atomicAdd(float* p, float v)       { float o = *p; *p += v; return o; }
inline double   atomicAdd(double* p, double v)     { double o = *p; *p += v; return o; }
inline unsigned atomicAnd(unsigned* p, unsigned v) { unsigned o = *p; *p &= v; return o; }
inline float    atomicMax(float* p, float v)       { float o = *p; if (v > *p) *p = v; return o; }

// Vector types (Phase 3a: the float2/float4 load-width sweep).
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

// Fast-math device intrinsics used by the activations and softmax.
inline float __expf(float x)   { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __frcp_rn(float x) { return 1.0f / x; }
