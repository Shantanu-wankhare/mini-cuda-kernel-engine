// =============================================================================
//  mcke/runtime/cublas_check.hpp
//
//  WHAT: cublasStatus_t -> mcke::Status, and MCKE_CUBLAS_RETURN_IF_ERROR.
//        The exact mirror of runtime/cuda_check.hpp, one library over.
//
//  WHY A SEPARATE HEADER RATHER THAN A FEW LINES IN gemm.cu: so that
//  <cublas_v2.h> is included by exactly ONE translation unit in the project.
//  cuBLAS is a 300 KB header and a dependency that only the kCublasRef ceiling
//  variant needs; letting it leak into the general include graph would mean
//  every bench and every .cu pays for it, and would quietly make cuBLAS look
//  like part of MCKE's runtime rather than the external reference point it is.
//  CMakeLists.txt links CUDA::cublas PRIVATE to mcke_kernels for the same reason
//  -- PRIVATE still propagates the link requirement to consumers, but not the
//  include directories.
//
//  WHY WE NEED IT AT ALL: cuBLAS does not share CUDA's error vocabulary. It has
//  its own sparse status enum and its own name/string accessors, so a bare
//  `if (st != CUBLAS_STATUS_SUCCESS)` would either lose the reason entirely or
//  print an integer. CLAUDE.md's rule is "never a bare CUDA call"; the same
//  applies here, and the rule is only enforceable if the macro exists.
// =============================================================================
#pragma once

#include "mcke/core/config.hpp"
#include "mcke/core/status.hpp"

#if MCKE_WITH_CUDA

#include <cublas_v2.h>

#include <string>

namespace mcke::rt {

// cublasGetStatusName / cublasGetStatusString are CUDA 11.4+. They are the
// direct analogues of cudaGetErrorName / cudaGetErrorString, and we want both
// for the same reason: the name to grep for, the prose to read.
[[nodiscard]] inline Status cublas_status(cublasStatus_t st, const char* expr,
                                          const char* file, int line) {
  if (st == CUBLAS_STATUS_SUCCESS) return OkStatus();

  const std::string detail = std::string(cublasGetStatusName(st)) + " (" +
                             cublasGetStatusString(st) + ") at " + file + ":" +
                             std::to_string(line) + " in `" + expr + "`";

  // Map onto our vocabulary where a distinct code is genuinely more useful than
  // the generic one. ALLOC_FAILED becoming OutOfMemoryError matters because the
  // allocators built in Phase 2 already treat OOM as an expected, recoverable
  // condition rather than a bug -- and a cuBLAS handle allocates a workspace, so
  // this path is reachable when the device is nearly full. ARCH_MISMATCH and
  // NOT_SUPPORTED become Unimplemented because they mean "this build of cuBLAS
  // will never run here", which is a different action for the caller than "the
  // call failed, retry or report".
  switch (st) {
    case CUBLAS_STATUS_ALLOC_FAILED:  return OutOfMemoryError(detail);
    case CUBLAS_STATUS_INVALID_VALUE: return InvalidArgumentError(detail);
    case CUBLAS_STATUS_ARCH_MISMATCH:
    case CUBLAS_STATUS_NOT_SUPPORTED: return UnimplementedError(detail);
    default:                          return CudaError(detail);
  }
}

}  // namespace mcke::rt

// Use in functions that return Status. Non-fatal, propagates upward.
#define MCKE_CUBLAS_RETURN_IF_ERROR(expr)                                       \
  do {                                                                          \
    cublasStatus_t _s = (expr);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS)                                            \
      return ::mcke::rt::cublas_status(_s, #expr, __FILE__, __LINE__);          \
  } while (0)

#define MCKE_CUBLAS_CHECK(expr)                                                 \
  ::mcke::rt::cublas_status((expr), #expr, __FILE__, __LINE__).throw_if_error()

#endif  // MCKE_WITH_CUDA
