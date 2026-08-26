// =============================================================================
//  mcke/runtime/cuda_check.hpp
//
//  WHAT: The two macros that wrap every CUDA API call in the project.
//  WHY a separate header: it is the only header that includes
//       <cuda_runtime_api.h>, so host-only translation units on macOS never
//       accidentally pull in CUDA headers. Everything is behind
//       `#if MCKE_WITH_CUDA`, so the file is still *includable* on macOS —
//       it just expands to nothing.
//
//  ---------------------------------------------------------------------------
//  THE SINGLE MOST IMPORTANT THING TO UNDERSTAND ABOUT CUDA ERROR HANDLING:
//  CUDA is asynchronous. `cudaMemcpyAsync` and kernel launches return
//  *immediately*, before the work has run. So there are two distinct error
//  classes:
//
//    (a) Synchronous / launch-time errors — bad grid dims, too much shared
//        memory, invalid pointer. Reported by the launch call's return value
//        (or the next `cudaGetLastError()`).
//    (b) Asynchronous / execution-time errors — illegal address, misaligned
//        access, device assert, ECC error. These surface at some *later*,
//        unrelated CUDA call. That is why a segfault in kernel A frequently
//        appears as "cudaErrorIllegalAddress" thrown from an innocent
//        cudaMemcpy three lines down.
//
//  Consequence: MCKE_CUDA_CHECK on a launch catches only (a). To attribute (b)
//  correctly you must synchronise, which costs a pipeline drain — so we do it
//  only in debug builds via MCKE_CUDA_CHECK_LAUNCH.
//
//  Note also: once a context hits an async error it is *sticky*. Every
//  subsequent call in that context returns the same error until the process
//  dies. There is no recovery. So checking early is the only strategy.
// =============================================================================
#pragma once

#include "mcke/core/config.hpp"
#include "mcke/core/status.hpp"

#if MCKE_WITH_CUDA

#include <cuda_runtime_api.h>

#include <string>

namespace mcke::rt {

// Converts a cudaError_t into our Status vocabulary, preserving the CUDA name
// and description plus the source location. `cudaGetErrorName` gives the enum
// spelling ("cudaErrorIllegalAddress"), `cudaGetErrorString` the prose — you
// want both: the name to grep for, the prose to read.
[[nodiscard]] inline Status cuda_status(cudaError_t err, const char* expr,
                                        const char* file, int line) {
  if (err == cudaSuccess) return OkStatus();
  return CudaError(std::string(cudaGetErrorName(err)) + " (" +
                   cudaGetErrorString(err) + ") at " + file + ":" +
                   std::to_string(line) + " in `" + expr + "`");
}

}  // namespace mcke::rt

// Use in functions that return Status. Non-fatal, propagates upward.
#define MCKE_CUDA_RETURN_IF_ERROR(expr)                                         \
  do {                                                                          \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess)                                                      \
      return ::mcke::rt::cuda_status(_e, #expr, __FILE__, __LINE__);            \
  } while (0)

// Use in constructors / destructors / main where there is nothing to return.
// Throws, because a failed cudaSetDevice is not a recoverable condition.
#define MCKE_CUDA_CHECK(expr)                                                   \
  ::mcke::rt::cuda_status((expr), #expr, __FILE__, __LINE__).throw_if_error()

// Debug-only full barrier after a kernel launch, to attribute async faults to
// the kernel that caused them. cudaDeviceSynchronize() (not
// cudaStreamSynchronize) because an illegal access on *any* stream poisons the
// context, and we want the report regardless of which stream misbehaved.
// This is compiled out in release: it would serialise the whole pipeline and
// destroy the async-scheduling benefits we are building in Phase 4.
#ifndef NDEBUG
#  define MCKE_CUDA_CHECK_LAUNCH()                                              \
     do {                                                                       \
       MCKE_CUDA_CHECK(cudaGetLastError());        /* class (a): launch config */\
       MCKE_CUDA_CHECK(cudaDeviceSynchronize());   /* class (b): execution     */\
     } while (0)
#else
#  define MCKE_CUDA_CHECK_LAUNCH() MCKE_CUDA_CHECK(cudaGetLastError())
#endif

#endif  // MCKE_WITH_CUDA
