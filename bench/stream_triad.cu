// =============================================================================
//  bench/stream_triad.cu
//
//  WHAT: Measures *sustained achievable* DRAM bandwidth via the classic STREAM
//        triad:  a[i] = b[i] + scalar * c[i].
//
//  WHY THIS EXISTS — it produces the "measured bandwidth" denominator used
//  everywhere else in this project (docs/PROFILING.md sec 2). DeviceInfo::
//  peak_dram_gb_s() computes a THEORETICAL number from the memory clock and bus
//  width in cudaDeviceProp — a spec-sheet figure. Real GPUs never sustain 100%
//  of that: ECC overhead, refresh cycles, and the fact that read+write traffic
//  share the same physical bus all eat into it. Typical hardware sustains
//  80-90% of the spec-formula number. Using the spec number as the denominator
//  for "% of peak" would make every real kernel look worse than it actually is
//  relative to what's physically achievable — so we measure the real ceiling.
//
//  WHY THE TRIAD SPECIFICALLY (not a plain copy): `a[i]=b[i]` is 2 accesses
//  (1 read + 1 write) with 0 FLOPs — it doesn't exercise the "read two operand
//  streams" pattern our real kernels use (GEMM, reductions all read multiple
//  inputs). The triad reads two arrays and writes one (3 accesses, 1 FMA) and
//  is the standard McCalpin STREAM benchmark, ported to a single grid-stride
//  GPU kernel. It is deliberately near-zero arithmetic intensity
//  (AI = 2 flops / 12 bytes = 0.17) — a pure bandwidth test, structurally
//  identical to the Phase-0 vector_add smoke kernel, but under the standard
//  name so the number is comparable to published STREAM results if wanted.
//
//  WHY .cu HERE and not part of kernels/: this is a standalone diagnostic tool
//  — no Op subclass will ever call it. Keeping it in bench/ keeps kernels/
//  scoped to things the graph executor actually launches.
// =============================================================================
#include <cstdint>
#include <cstdio>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/cuda_check.hpp"
#include "mcke/runtime/stream.hpp"

using namespace mcke;

namespace {

// Grid-stride triad kernel. See kernels/elementwise.cu for the line-by-line
// rationale of __restrict__ and 64-bit indexing — identical reasoning applies
// here and is not repeated.
__global__ void triad_kernel(const float* __restrict__ b, const float* __restrict__ c,
                             float* __restrict__ a, float scalar, std::size_t n) {
  const std::size_t tid    = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    // One fused multiply-add — matches the reference STREAM triad's "1 FMA per
    // element" exactly. A separate `scalar*c[i]` then `+b[i]` would compile to
    // two instructions for the same bytes moved and understate the kernel's
    // real arithmetic intensity.
    a[i] = fmaf(scalar, c[i], b[i]);
  }
}

}  // namespace

int main() {
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }
  auto dev = query_device(0);
  dev.status().throw_if_error();
  set_device(0).throw_if_error();

  auto stream = rt::Stream::create();
  stream.status().throw_if_error();

  // Same working-set size as the Phase-0 smoke test (256 MiB/buffer, 768 MiB
  // total) — large enough to sit far outside L2 (a few MiB on every arch we
  // target) so this measures DRAM, not cache.
  const std::size_t n     = 64ull << 20;
  const std::size_t bytes = n * sizeof(float);

  // A bandwidth microbenchmark has no business exercising our pool allocator's
  // logic — that is tested separately in Phase 2. Use the plain baseline so
  // allocator overhead cannot leak into this measurement.
  RawDeviceAllocator alloc;
  auto da = alloc.allocate(bytes, stream->native());
  auto db = alloc.allocate(bytes, stream->native());
  auto dc = alloc.allocate(bytes, stream->native());
  da.status().throw_if_error();
  db.status().throw_if_error();
  dc.status().throw_if_error();

  std::vector<float> hb(n), hc(n);
  for (std::size_t i = 0; i < n; ++i) { hb[i] = static_cast<float>(i % 997); hc[i] = 1.0f; }

  MCKE_CUDA_CHECK(cudaMemcpyAsync(db->ptr, hb.data(), bytes, cudaMemcpyHostToDevice, stream->native()));
  MCKE_CUDA_CHECK(cudaMemcpyAsync(dc->ptr, hc.data(), bytes, cudaMemcpyHostToDevice, stream->native()));

  constexpr int kThreads = 256;
  const std::size_t blocks_needed = (n + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks_needed < 4096 ? blocks_needed : 4096);
  const float scalar = 3.14159f;

  Profiler prof;
  const std::uint64_t ideal_bytes = 3ull * n * sizeof(float);  // read b, read c, write a
  const std::uint64_t ideal_flops = 2ull * n;                  // 1 FMA = 2 flops/element

  auto rec = prof.time_op(
      "stream_triad", "grid_stride_256t", *stream, ideal_flops, ideal_bytes,
      /*warmup=*/5, /*iters=*/20,
      [&](const rt::Stream& s) {
        triad_kernel<<<blocks, kThreads, 0, s.native()>>>(
            static_cast<const float*>(db->ptr), static_cast<const float*>(dc->ptr),
            static_cast<float*>(da->ptr), scalar, n);
        MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
        return OkStatus();
      });
  rec.status().throw_if_error();
  stream->synchronize().throw_if_error();

  Roofline rl;
  rl.peak_gb_s = dev->peak_dram_gb_s();
  std::printf("\n%s\n", prof.summary_table(rl).c_str());
  std::printf(
      "achieved %.1f GB/s  (spec-formula peak %.1f GB/s -> %.1f%%)\n\n"
      "*** This achieved GB/s IS the \"measured bandwidth\" figure referenced\n"
      "*** throughout docs/PROFILING.md. Record it in RESULTS.md section 0 and\n"
      "*** use it -- not the spec-formula number -- as the denominator for every\n"
      "*** other kernel's bandwidth efficiency.\n",
      rec->gb_per_s(), rl.peak_gb_s, rec->gb_per_s() / rl.peak_gb_s * 100.0);

  alloc.deallocate(*da, stream->native()).throw_if_error();
  alloc.deallocate(*db, stream->native()).throw_if_error();
  alloc.deallocate(*dc, stream->native()).throw_if_error();
  return 0;
}
