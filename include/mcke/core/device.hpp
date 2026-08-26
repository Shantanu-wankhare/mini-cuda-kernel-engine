// =============================================================================
//  mcke/core/device.hpp
//
//  WHAT: `DeviceInfo` — everything about the GPU that our *planning* code needs,
//        captured once into a plain struct, plus the raw device-memory calls.
//
//  DESIGN DECISION — snapshot the device properties into a POD.
//  cudaGetDeviceProperties fills a 1 KB struct and is not free (it can take
//  ~ms on first call). More importantly, `cudaDeviceProp` is a CUDA type: if
//  the graph scheduler took it as a parameter, the scheduler could no longer be
//  compiled on the Mac. So we copy the ~15 fields we actually use into our own
//  header-only struct with no CUDA dependency. Now tile-size selection and
//  occupancy math are testable on a laptop with a hand-written DeviceInfo.
//
//  This is the general pattern for the whole project: *CUDA types stop at the
//  runtime boundary*. Above that line everything is plain C++20.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "mcke/core/config.hpp"
#include "mcke/core/status.hpp"

namespace mcke {

struct DeviceInfo {
  int         index = 0;
  std::string name  = "host-emulated";

  // --- Compute capability. Drives every "can I use this instruction" decision:
  //     __shfl_down_sync needs >= 3.0, bf16 needs >= 8.0, async copy
  //     (cp.async / cuda::memcpy_async) needs >= 8.0, wgmma needs >= 9.0.
  int cc_major = 0;
  int cc_minor = 0;

  // --- Parallelism budget, used for occupancy and grid sizing.
  int sm_count                  = 0;   // multiprocessorCount
  int max_threads_per_sm        = 0;   // maxThreadsPerMultiProcessor
  int max_threads_per_block     = 0;
  int warps_per_sm              = 0;   // derived: max_threads_per_sm / 32
  int regs_per_sm               = 0;   // register file per SM (in 32-bit regs)
  int regs_per_block            = 0;

  // --- Memory hierarchy. `shared_mem_per_block_optin` is the one that matters:
  //     on sm_75+ the default limit is 48 KiB per block, but you can opt in to
  //     more (up to 64/100/164/228 KiB depending on arch) via
  //     cudaFuncSetAttribute. Our tiled GEMM in Phase 3 will need this.
  std::size_t global_mem_bytes             = 0;
  std::size_t shared_mem_per_block         = 0;
  std::size_t shared_mem_per_block_optin   = 0;
  std::size_t shared_mem_per_sm            = 0;
  int         l2_cache_bytes               = 0;
  int         memory_bus_width_bits        = 0;
  int         memory_clock_khz             = 0;   // effective (already DDR-doubled by CUDA)

  // --- Capabilities that change our algorithm, not just our constants.
  bool supports_cooperative_launch = false;
  bool supports_mem_pools          = false;  // cudaMallocAsync / stream-ordered
  int  async_engine_count          = 0;      // copy engines => H2D/D2H overlap depth

  // -------------------------------------------------------------------------
  // Derived quantities. These are the numbers that appear in the roofline plot,
  // so they live next to the data they come from rather than in a benchmark
  // script where they would drift.
  // -------------------------------------------------------------------------

  // Theoretical peak DRAM bandwidth, GB/s (decimal GB, matching NVIDIA specs).
  //   bytes/s = clock_Hz * bus_width_bytes
  // memory_clock_khz is already the effective data rate, so no extra x2 for DDR
  // — double-counting that factor is the classic way to report a "50% of peak"
  // kernel as "25% of peak".
  [[nodiscard]] double peak_dram_gb_s() const {
    return static_cast<double>(memory_clock_khz) * 1e3 *
           (static_cast<double>(memory_bus_width_bits) / 8.0) / 1e9;
  }

  [[nodiscard]] int compute_capability() const { return cc_major * 10 + cc_minor; }

  [[nodiscard]] std::string describe() const;
};

// Number of visible CUDA devices. Returns 0 in a host-only build — so callers
// must handle "no GPU" as a normal condition, which is exactly the discipline
// we want when half our development happens on a Mac.
[[nodiscard]] int device_count();

// Query + snapshot. `index` must be < device_count().
[[nodiscard]] StatusOr<DeviceInfo> query_device(int index = 0);

// Binds the calling *host thread* to a device. CUDA's current-device is
// thread-local, which is a real trap once we add a worker thread pool in
// Phase 4: a thread that never called cudaSetDevice operates on device 0.
Status set_device(int index);

// --- Raw device memory. Our allocators call these; nothing else should.
//     Deliberately thin: no logging, no bookkeeping. All policy lives one layer
//     up in DeviceAllocator implementations.
[[nodiscard]] StatusOr<void*> raw_device_malloc(std::size_t bytes);
Status                        raw_device_free(void* ptr);

// Free / total device memory *as the driver sees it*. Useful to prove that our
// pool really did stop calling cudaMalloc: free memory should go flat after
// warm-up.
struct MemInfo { std::size_t free_bytes = 0; std::size_t total_bytes = 0; };
[[nodiscard]] StatusOr<MemInfo> device_mem_info();

}  // namespace mcke
