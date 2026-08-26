// =============================================================================
//  src/core/device.cpp
//
//  WHY .cpp and not .cu:
//  This file calls the CUDA *runtime API* (cudaGetDeviceProperties, cudaMalloc)
//  but contains no device code — no __global__, no __device__, no <<<>>>.
//  Host-only CUDA API calls are just C function calls into libcudart; any host
//  compiler can compile them as long as it can see <cuda_runtime_api.h> and the
//  linker gets -lcudart. nvcc is required *only* for translation units that
//  contain device code or the `<<<...>>>` launch syntax.
//
//  This distinction is worth burning in, because it decides your file layout:
//    - .cu  : kernels + launch sites (nvcc, slow to compile, arch-specific)
//    - .cpp : everything else, including most CUDA API usage (fast, portable)
//  Keeping .cu files few and small is the main lever on build time in CUDA
//  projects — nvcc compiles each file once per target architecture.
// =============================================================================
#include "mcke/core/device.hpp"

#include <cstdlib>
#include <sstream>

#include "mcke/runtime/cuda_check.hpp"

namespace mcke {

std::string DeviceInfo::describe() const {
  std::ostringstream os;
  os << "[device " << index << "] " << name
     << "  sm_" << cc_major << cc_minor << '\n'
     << "  SMs                     : " << sm_count << '\n'
     << "  max threads / SM        : " << max_threads_per_sm
     << "  (" << warps_per_sm << " warps)\n"
     << "  regs / SM               : " << regs_per_sm << '\n'
     << "  shared mem / block      : " << shared_mem_per_block / 1024 << " KiB"
     << "  (opt-in max " << shared_mem_per_block_optin / 1024 << " KiB)\n"
     << "  shared mem / SM         : " << shared_mem_per_sm / 1024 << " KiB\n"
     << "  L2 cache                : " << l2_cache_bytes / 1024 << " KiB\n"
     << "  global memory           : " << global_mem_bytes / (1024ull * 1024 * 1024)
     << " GiB\n"
     << "  memory bus              : " << memory_bus_width_bits << "-bit @ "
     << memory_clock_khz / 1000 << " MHz\n"
     << "  peak DRAM bandwidth     : " << peak_dram_gb_s() << " GB/s\n"
     << "  copy engines            : " << async_engine_count << '\n'
     << "  cooperative launch      : " << (supports_cooperative_launch ? "yes" : "no") << '\n'
     << "  stream-ordered mem pools: " << (supports_mem_pools ? "yes" : "no") << '\n';
  return os.str();
}

#if MCKE_WITH_CUDA
// -----------------------------------------------------------------------------
// CUDA backend
// -----------------------------------------------------------------------------

int device_count() {
  int n = 0;
  // Deliberately swallow the error: "no CUDA driver installed" is a legitimate
  // state we want to report as 0 devices, not as an exception. Note this also
  // *initialises* the CUDA driver on first call (lazy context creation), which
  // can take 100-300 ms — do not put it inside a timed region.
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

StatusOr<DeviceInfo> query_device(int index) {
  if (index < 0 || index >= device_count())
    return InvalidArgumentError("query_device: no such device " + std::to_string(index));

  cudaDeviceProp p{};
  cudaError_t e = cudaGetDeviceProperties(&p, index);
  if (e != cudaSuccess) return rt::cuda_status(e, "cudaGetDeviceProperties", __FILE__, __LINE__);

  DeviceInfo d;
  d.index                     = index;
  d.name                      = p.name;
  d.cc_major                  = p.major;
  d.cc_minor                  = p.minor;
  d.sm_count                  = p.multiProcessorCount;
  d.max_threads_per_sm        = p.maxThreadsPerMultiProcessor;
  d.max_threads_per_block     = p.maxThreadsPerBlock;
  d.warps_per_sm              = p.maxThreadsPerMultiProcessor / kWarpSize;
  d.regs_per_sm               = p.regsPerMultiprocessor;
  d.regs_per_block            = p.regsPerBlock;
  d.global_mem_bytes          = p.totalGlobalMem;
  d.shared_mem_per_block      = p.sharedMemPerBlock;
  d.shared_mem_per_block_optin = p.sharedMemPerBlockOptin;
  d.shared_mem_per_sm         = p.sharedMemPerMultiprocessor;
  d.l2_cache_bytes            = p.l2CacheSize;
  d.memory_bus_width_bits     = p.memoryBusWidth;
  d.memory_clock_khz          = p.memoryClockRate;
  d.supports_cooperative_launch = p.cooperativeLaunch != 0;
  d.supports_mem_pools        = p.memoryPoolsSupported != 0;
  d.async_engine_count        = p.asyncEngineCount;
  return d;
}

Status set_device(int index) {
  MCKE_CUDA_RETURN_IF_ERROR(cudaSetDevice(index));
  return OkStatus();
}

StatusOr<void*> raw_device_malloc(std::size_t bytes) {
  if (bytes == 0) return InvalidArgumentError("raw_device_malloc: zero-byte request");
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e == cudaErrorMemoryAllocation) {
    // Map the CUDA-specific OOM onto our own kOutOfMemory so the allocator
    // layer can implement a fallback policy without knowing about CUDA.
    // IMPORTANT: cudaMalloc failing does NOT poison the context — unlike an
    // async fault, OOM is cleanly recoverable, which is what makes a
    // "try large, fall back to small" pool policy viable at all.
    return OutOfMemoryError("cudaMalloc(" + std::to_string(bytes) + ") failed");
  }
  if (e != cudaSuccess) return rt::cuda_status(e, "cudaMalloc", __FILE__, __LINE__);
  return p;
}

Status raw_device_free(void* ptr) {
  if (ptr == nullptr) return OkStatus();          // free(nullptr) is a no-op, by convention
  MCKE_CUDA_RETURN_IF_ERROR(cudaFree(ptr));
  return OkStatus();
}

StatusOr<MemInfo> device_mem_info() {
  MemInfo m{};
  MCKE_CUDA_RETURN_IF_ERROR(cudaMemGetInfo(&m.free_bytes, &m.total_bytes));
  return m;
}

#else
// -----------------------------------------------------------------------------
// Host backend (macOS / any machine without CUDA).
//
// This is NOT a GPU emulator and never will be. Its only job is to let the
// host-side half of the runtime — allocator bookkeeping, graph topology,
// liveness analysis, scheduling decisions — be compiled and unit-tested on the
// MacBook. Kernels are simply absent from this build.
// -----------------------------------------------------------------------------

int device_count() { return 0; }

StatusOr<DeviceInfo> query_device(int) {
  return FailedPreconditionError(
      "query_device: this binary was built with MCKE_ENABLE_CUDA=OFF (host-only)");
}

Status set_device(int) { return OkStatus(); }

StatusOr<void*> raw_device_malloc(std::size_t bytes) {
  if (bytes == 0) return InvalidArgumentError("raw_device_malloc: zero-byte request");
  // aligned_alloc requires size to be a multiple of alignment, and we mimic
  // cudaMalloc's 256 B alignment guarantee so alignment bugs in the allocator
  // are reproducible on the host.
  const std::size_t rounded = (bytes + kDeviceAlignment - 1) / kDeviceAlignment * kDeviceAlignment;
  void* p = std::aligned_alloc(kDeviceAlignment, rounded);
  if (p == nullptr) return OutOfMemoryError("aligned_alloc(" + std::to_string(rounded) + ") failed");
  return p;
}

Status raw_device_free(void* ptr) { std::free(ptr); return OkStatus(); }

StatusOr<MemInfo> device_mem_info() {
  return FailedPreconditionError("device_mem_info: host-only build");
}

#endif  // MCKE_WITH_CUDA

}  // namespace mcke
