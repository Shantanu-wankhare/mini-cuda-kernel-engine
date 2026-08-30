// Fake CUDA runtime header. NOT for shipping — a scratch harness that lets the
// MCKE_WITH_CUDA=1 code paths be type-checked on a machine with no CUDA at all.
// It validates signatures and the include chain, NOT runtime behaviour.
#pragma once
#include <cstddef>
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st*  cudaEvent_t;
typedef enum {
  cudaSuccess = 0, cudaErrorNotReady = 600, cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidValue = 1
} cudaError_t;
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
               cudaMemcpyDeviceToDevice = 3 } cudaMemcpyKind;
enum { cudaStreamNonBlocking = 1, cudaEventDefault = 0, cudaEventDisableTiming = 2,
       cudaHostAllocMapped = 2 };
struct cudaDeviceProp {
  char name[256]; int major, minor, multiProcessorCount, maxThreadsPerMultiProcessor;
  int maxThreadsPerBlock, regsPerMultiprocessor, regsPerBlock;
  std::size_t totalGlobalMem, sharedMemPerBlock, sharedMemPerBlockOptin, sharedMemPerMultiprocessor;
  int l2CacheSize, memoryBusWidth, memoryClockRate, cooperativeLaunch;
  int memoryPoolsSupported, asyncEngineCount;
  // The 4th occupancy limiter. Added for Phase 3d: sm_75 caps resident blocks at
  // 16 (Volta was 32, Turing halved it), and an occupancy calculator that
  // hardcodes either value is wrong on some target in CLAUDE.md's table.
  int maxBlocksPerMultiProcessor;
};
const char* cudaGetErrorName(cudaError_t);
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaSetDevice(int);
template <class T> cudaError_t cudaMalloc(T**, std::size_t);   // real API is a template
cudaError_t cudaFree(void*);
cudaError_t cudaMemGetInfo(std::size_t*, std::size_t*);
cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t*, unsigned, int);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
template <class T> cudaError_t cudaHostAlloc(T**, std::size_t, unsigned);
template <class T> cudaError_t cudaHostGetDevicePointer(T**, void*, unsigned);
cudaError_t cudaFreeHost(void*);

// NOTE: the cuBLAS declarations used to live here. They moved to the sibling
// `cublas_v2.h` in Phase 3d, because kernels/gemm.cu includes BOTH this header
// (transitively, via runtime/cuda_check.hpp) and <cublas_v2.h>. Declaring
// CUBLAS_STATUS_SUCCESS and CUBLAS_OP_N in two headers that meet in one TU is a
// hard redeclaration error, so the surface has to live in exactly one place --
// which is also how the real toolkit ships it.

// Versions, for the RESULTS.md rule-1 environment banner.
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaGetDevice(int*);

// cudaFuncAttributes: the bench reads regs/thread + smem/block from this rather
// than parsing -Xptxas output, so the RESULTS.md occupancy columns self-populate.
struct cudaFuncAttributes {
  std::size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
  int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA;
  int maxDynamicSharedSizeBytes, preferredShmemCarveout;
};
template <class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, T*);

// TEMPLATED ON THE KERNEL TYPE, deliberately -- an earlier version of this line
// declared the first kernel parameter as `const void*`, matching the C API. That
// does not compile at the call site: converting a __global__ function pointer
// (`void(*)(const float*, ...)`) to `const void*` is ill-formed in standard C++.
// The real toolkit resolves this with C++ template overloads in cuda_runtime.h,
// which nvcc force-includes into every .cu but which this project never includes
// directly. cudaFuncGetAttributes above already had the right shape.
template <class T>
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, T*, int, std::size_t);

cudaError_t cudaMemset(void*, int, std::size_t);
cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
