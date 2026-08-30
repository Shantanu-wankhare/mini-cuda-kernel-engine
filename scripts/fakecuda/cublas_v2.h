// Fake cuBLAS header. NOT for shipping -- part of the scratch harness that lets
// the MCKE_WITH_CUDA=1 code paths be type-checked on a machine with no CUDA at
// all. Signatures only; no behaviour.
//
// WHY THIS IS A SEPARATE FILE, when these declarations used to live inside
// scripts/fakecuda/cuda_runtime_api.h: kernels/gemm.cu includes both that header
// (transitively, through mcke/runtime/cuda_check.hpp) and this one (through
// mcke/runtime/cublas_check.hpp). Two unnamed enums each declaring
// CUBLAS_STATUS_SUCCESS, meeting in one translation unit, is a hard
// redeclaration error -- so the surface has to live in exactly one header. The
// real toolkit ships it this way for the same reason.
#pragma once
#include <cuda_runtime_api.h>   // cudaStream_t

typedef struct cublasContext* cublasHandle_t;

// Real enumerator VALUES, not invented ones. This costs nothing here and means
// the fake and the real header agree if anyone ever prints a raw status code.
// Note the gaps (2, 4, 5, 6, ...) are genuine -- cuBLAS's status enum is sparse.
typedef enum {
  CUBLAS_STATUS_SUCCESS          = 0,
  CUBLAS_STATUS_NOT_INITIALIZED  = 1,
  CUBLAS_STATUS_ALLOC_FAILED     = 3,
  CUBLAS_STATUS_INVALID_VALUE    = 7,
  CUBLAS_STATUS_ARCH_MISMATCH    = 8,
  CUBLAS_STATUS_MAPPING_ERROR    = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR   = 14,
  CUBLAS_STATUS_NOT_SUPPORTED    = 15,
  CUBLAS_STATUS_LICENSE_ERROR    = 16
} cublasStatus_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;

// CUBLAS_PEDANTIC_MATH is the one that matters for this project, and an earlier
// version of the fake header had its value swapped with CUBLAS_DEFAULT_MATH.
// Phase 3d requires PEDANTIC so cublasSgemm cannot silently drop to TF32 tensor
// cores on an L4/A100/5060 -- which would measure the ceiling row in different
// arithmetic than the seven hand-written rows it is the ceiling for.
typedef enum {
  CUBLAS_DEFAULT_MATH                             = 0,
  CUBLAS_PEDANTIC_MATH                            = 1,
  CUBLAS_TF32_TENSOR_OP_MATH                      = 3,
  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
  CUBLAS_POINTER_MODE_HOST   = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
cublasStatus_t cublasSetStream_v2(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t, cublasPointerMode_t);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t, int*);

// CUDA 11.4+. Needed so cublas_check.hpp can mirror cuda_status()'s
// name-plus-prose error text instead of printing a bare integer.
const char* cublasGetStatusName(cublasStatus_t);
const char* cublasGetStatusString(cublasStatus_t);

cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                              int m, int n, int k, const float* alpha,
                              const float* A, int lda, const float* B, int ldb,
                              const float* beta, float* C, int ldc);

#define cublasCreate         cublasCreate_v2
#define cublasDestroy        cublasDestroy_v2
#define cublasSetStream      cublasSetStream_v2
#define cublasSetPointerMode cublasSetPointerMode_v2
#define cublasGetVersion     cublasGetVersion_v2
#define cublasSgemm          cublasSgemm_v2
