// Fake NVTX v3 header. NOT for shipping -- part of the scratch harness that
// type-checks the MCKE_WITH_CUDA=1 path on a machine with no CUDA.
//
// WHY IT EXISTS: without it, the MCKE_USE_NVTX code path in profiler.hpp and
// every NvtxRange in src/graph/executor.cpp would reach Explorer completely
// unverified -- and Phase 4's exit criterion is an nsys timeline, so that path
// is not optional decoration. scripts/typecheck_cuda.sh runs a second pass with
// -DMCKE_USE_NVTX for exactly this reason.
#pragma once
inline int  nvtxRangePushA(const char*) { return 0; }
inline int  nvtxRangePop(void)          { return 0; }
inline void nvtxMarkA(const char*)      {}
