// =============================================================================
//  tools/smoke_vector_add.cpp
//
//  WHAT: End-to-end Phase-0 check on a GPU machine. Allocates through our
//        allocator interface, copies H2D, launches a kernel on a non-default
//        stream, times it with events, verifies the result, reports achieved
//        bandwidth.
//
//  WHY .cpp and not .cu, when it launches a kernel: it does NOT launch a kernel
//  — it calls `launch_vector_add_f32`, a plain C++ function whose *body* (in
//  elementwise.cu) contains the <<<>>>. This is the boundary described in
//  kernels.hpp, and this file is the proof that it works: an ordinary host
//  translation unit driving GPU work.
//
//  WHY this is the first thing to run on every new machine: it fails distinctly
//  for each different problem. Wrong arch -> "no kernel image is available for
//  execution on the device". No driver -> device_count()==0. Broken alignment ->
//  wrong results. Bad stream flags -> it works but shows no overlap later.
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/kernels/kernels.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/stream.hpp"

using namespace mcke;

int main(int argc, char** argv) {
  // 64 Mi elements = 256 MiB per buffer, 768 MiB total. Big enough that the
  // measurement is dominated by DRAM bandwidth rather than launch overhead, and
  // big enough to blow well past any L2 (so we are measuring DRAM, not cache).
  const std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (64ull << 20);

  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to smoke-test\n");
    return 0;
  }
  query_device(0).status().throw_if_error();
  set_device(0).throw_if_error();
  auto dev = query_device(0);
  dev.status().throw_if_error();

  auto stream = rt::Stream::create();
  stream.status().throw_if_error();

  RawDeviceAllocator alloc;   // Phase 0 uses the baseline allocator on purpose:
                              // one variable at a time. The pool arrives in
                              // Phase 2 and must beat *this*.

  const std::size_t bytes = n * sizeof(float);
  auto da = alloc.allocate(bytes, stream->native());
  auto db = alloc.allocate(bytes, stream->native());
  auto dc = alloc.allocate(bytes, stream->native());
  da.status().throw_if_error();
  db.status().throw_if_error();
  dc.status().throw_if_error();

  // Host staging buffers. NOTE: these are *pageable* allocations, so the H2D
  // copy below goes through a driver-managed staging buffer and cannot exceed
  // ~6 GB/s regardless of the PCIe link. Pinned memory (cudaHostAlloc) is
  // needed for full-rate, truly-async transfers — that is a Phase 5 experiment,
  // and deliberately not done here so the smoke test stays dependency-free.
  std::vector<float> ha(n), hb(n), hc(n);
  for (std::size_t i = 0; i < n; ++i) { ha[i] = static_cast<float>(i % 1000); hb[i] = 1.0f; }

#if MCKE_WITH_CUDA
  MCKE_CUDA_CHECK(cudaMemcpyAsync(da->ptr, ha.data(), bytes, cudaMemcpyHostToDevice,
                                  stream->native()));
  MCKE_CUDA_CHECK(cudaMemcpyAsync(db->ptr, hb.data(), bytes, cudaMemcpyHostToDevice,
                                  stream->native()));
#endif

  Profiler prof;
  // Ideal traffic: read a, read b, write c = 3 * n * 4 bytes. 1 FLOP per element.
  const std::uint64_t ideal_bytes = 3ull * n * sizeof(float);
  const std::uint64_t ideal_flops = n;

  auto rec = prof.time_op(
      "vector_add", "grid_stride_256t", *stream, ideal_flops, ideal_bytes,
      /*warmup=*/5, /*iters=*/50,
      [&](const rt::Stream& s) {
        return kernels::launch_vector_add_f32(
            static_cast<const float*>(da->ptr), static_cast<const float*>(db->ptr),
            static_cast<float*>(dc->ptr), n, s.native());
      });
  rec.status().throw_if_error();

#if MCKE_WITH_CUDA
  MCKE_CUDA_CHECK(cudaMemcpyAsync(hc.data(), dc->ptr, bytes, cudaMemcpyDeviceToHost,
                                  stream->native()));
#endif
  stream->synchronize().throw_if_error();

  // Correctness before performance, always. A fast wrong kernel is worthless,
  // and it is startlingly easy to "optimise" a kernel into producing zeros.
  std::size_t bad = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (hc[i] != ha[i] + hb[i]) ++bad;
  std::printf("verification: %s (%zu mismatches of %zu)\n", bad ? "FAIL" : "OK", bad, n);

  Roofline rl;
  rl.peak_gb_s = dev->peak_dram_gb_s();
  std::printf("\n%s\n", prof.summary_table(rl).c_str());
  std::printf("achieved %.1f GB/s of %.1f GB/s peak = %.1f%% of DRAM bandwidth\n",
              rec->gb_per_s(), rl.peak_gb_s, rec->gb_per_s() / rl.peak_gb_s * 100.0);
  std::printf("\nallocator: %s\n%s\n", alloc.name().data(), alloc.stats().to_string().c_str());

  alloc.deallocate(*da, stream->native()).throw_if_error();
  alloc.deallocate(*db, stream->native()).throw_if_error();
  alloc.deallocate(*dc, stream->native()).throw_if_error();
  return bad ? 1 : 0;
}
