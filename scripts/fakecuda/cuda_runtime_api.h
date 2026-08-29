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
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 } cudaMemcpyKind;
enum { cudaStreamNonBlocking = 1, cudaEventDefault = 0, cudaEventDisableTiming = 2,
       cudaHostAllocMapped = 2 };
struct cudaDeviceProp {
  char name[256]; int major, minor, multiProcessorCount, maxThreadsPerMultiProcessor;
  int maxThreadsPerBlock, regsPerMultiprocessor, regsPerBlock;
  std::size_t totalGlobalMem, sharedMemPerBlock, sharedMemPerBlockOptin, sharedMemPerMultiprocessor;
  int l2CacheSize, memoryBusWidth, memoryClockRate, cooperativeLaunch;
  int memoryPoolsSupported, asyncEngineCount;
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

// --- Added for Phase 3 (kernels). Signatures only; no behaviour. ---
typedef struct cublasContext* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS = 0 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
typedef enum { CUBLAS_PEDANTIC_MATH = 0, CUBLAS_DEFAULT_MATH = 1 } cublasMath_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0 } cublasPointerMode_t;
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
cublasStatus_t cublasSetStream_v2(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t, cublasPointerMode_t);
cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                              int m, int n, int k, const float* alpha,
                              const float* A, int lda, const float* B, int ldb,
                              const float* beta, float* C, int ldc);
#define cublasCreate         cublasCreate_v2
#define cublasDestroy        cublasDestroy_v2
#define cublasSetStream      cublasSetStream_v2
#define cublasSetPointerMode cublasSetPointerMode_v2
#define cublasSgemm          cublasSgemm_v2

// cudaFuncAttributes: the bench reads regs/thread + smem/block from this rather
// than parsing -Xptxas output, so the RESULTS.md occupancy columns self-populate.
struct cudaFuncAttributes {
  std::size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
  int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA;
  int maxDynamicSharedSizeBytes, preferredShmemCarveout;
};
template <class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, T*);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, const void*, int, std::size_t);
cudaError_t cudaMemset(void*, int, std::size_t);
cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
