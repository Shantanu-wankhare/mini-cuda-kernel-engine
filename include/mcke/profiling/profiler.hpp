// =============================================================================
//  mcke/profiling/profiler.hpp
//
//  WHAT: Per-node GPU timing (CUDA events), NVTX ranges for Nsight Systems, and
//        the derived-metric math (TFLOP/s, GB/s, % of peak, roofline position).
//
//  ---------------------------------------------------------------------------
//  THREE DIFFERENT TOOLS, THREE DIFFERENT JOBS — know which you want:
//
//   1. CUDA events (this file)      : "how long did this kernel take, on the
//      GPU, in this build?" Cheap (~0.5 us), always on, works everywhere
//      including Colab. This is what produces the numbers in RESULTS.md.
//   2. Nsight Systems (nsys)        : *timeline*. Where are the gaps? Is the CPU
//      starving the GPU? Did stream 2 actually overlap stream 1? Answers
//      scheduling questions. Needs NVTX ranges to be readable — hence the
//      NvtxRange class below.
//   3. Nsight Compute (ncu)         : *one kernel, in depth*. Occupancy,
//      memory-throughput breakdown per hierarchy level, warp-stall reasons,
//      SASS. Answers "why is this kernel slow?" Replays each kernel many times,
//      so it is far too slow to leave on, and it needs -lineinfo to map back to
//      source.
//
//  Using ncu to answer a scheduling question, or nsys to answer an occupancy
//  question, is the most common way people waste an afternoon.
//
//  ---------------------------------------------------------------------------
//  MEASUREMENT DISCIPLINE (baked into the API so we cannot skip it):
//   * warmup runs before timed runs — the first launch pays JIT/module load,
//     and the clocks are still ramping. A first-iteration number can be 10x off.
//   * report median and min over >= 20 iterations, not mean: the distribution is
//     right-skewed (OS jitter, other tenants on a shared HPC GPU), so the mean
//     tracks noise while the min approximates the machine's real capability and
//     the median is what you would actually experience.
//   * record clock state. On a laptop 5060 the GPU will thermally throttle
//     within seconds; the same kernel is 20% slower at minute 3. Explorer nodes
//     are far more stable, which is why serious numbers belong there.
// =============================================================================
#pragma once

#include <algorithm>   // std::nth_element, for the median in time_op
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mcke/core/config.hpp"
#include "mcke/core/device.hpp"
#include "mcke/core/status.hpp"
#include "mcke/runtime/stream.hpp"

#if MCKE_WITH_CUDA && defined(MCKE_USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace mcke {

// -----------------------------------------------------------------------------
// NvtxRange: RAII annotation that shows up as a named, coloured band in the
// Nsight Systems timeline.
//
// WHY it matters: without NVTX, an nsys timeline is a wall of
// "kernel_gemm_tiled" bars and you cannot tell which graph node each belongs
// to. With it, you see "node_7:Gemm" nested inside "graph_iter_3" and the gaps
// become interpretable. It costs ~100 ns and is compiled out when
// MCKE_USE_NVTX is off.
// -----------------------------------------------------------------------------
class NvtxRange {
 public:
  explicit NvtxRange([[maybe_unused]] const char* name) {
#if MCKE_WITH_CUDA && defined(MCKE_USE_NVTX)
    nvtxRangePushA(name);
    active_ = true;
#endif
  }
  ~NvtxRange() {
#if MCKE_WITH_CUDA && defined(MCKE_USE_NVTX)
    if (active_) nvtxRangePop();
#endif
  }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
#if MCKE_WITH_CUDA && defined(MCKE_USE_NVTX)
  bool active_ = false;
#endif
};

// -----------------------------------------------------------------------------
// One measurement.
// -----------------------------------------------------------------------------
struct KernelRecord {
  std::string   name;
  std::string   variant;          // e.g. "tiled_128x128x8_regblock_4x4"
  double        ms = 0.0;         // GPU time for one iteration
  std::uint64_t flops = 0;        // ideal, from OpCost
  std::uint64_t bytes = 0;        // ideal, from OpCost
  int           iterations = 1;
  int           stream_idx = 0;

  [[nodiscard]] double seconds() const { return ms * 1e-3; }
  [[nodiscard]] double tflops() const {
    return seconds() > 0 ? static_cast<double>(flops) / seconds() / 1e12 : 0.0;
  }
  [[nodiscard]] double gb_per_s() const {
    return seconds() > 0 ? static_cast<double>(bytes) / seconds() / 1e9 : 0.0;
  }
  [[nodiscard]] double arithmetic_intensity() const {
    return bytes ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;
  }
};

// -----------------------------------------------------------------------------
// Roofline evaluation. Given a device's peaks, tell us which side of the ridge
// point we are on and what fraction of the *attainable* performance we got.
//
//   attainable_flops = min(peak_compute, AI * peak_bandwidth)
//   ridge_point AI   = peak_compute / peak_bandwidth
//
// This is the number that makes an optimisation story credible: "62% of peak
// bandwidth on a memory-bound kernel at AI = 0.25" is meaningful in a way that
// "1.8x faster than my first version" is not — the first tells you how much is
// left, the second tells you how bad you were.
// -----------------------------------------------------------------------------
struct Roofline {
  double peak_tflops = 0.0;   // f32 FMA peak, from spec or a microbenchmark
  double peak_gb_s   = 0.0;   // from DeviceInfo::peak_dram_gb_s() or a STREAM-like test

  [[nodiscard]] double ridge_point_ai() const {
    return peak_gb_s > 0 ? (peak_tflops * 1e12) / (peak_gb_s * 1e9) : 0.0;
  }
  [[nodiscard]] double attainable_tflops(double ai) const {
    const double bw_bound = ai * peak_gb_s * 1e9 / 1e12;
    return bw_bound < peak_tflops ? bw_bound : peak_tflops;
  }
  [[nodiscard]] bool memory_bound(double ai) const { return ai < ridge_point_ai(); }
  [[nodiscard]] double efficiency(const KernelRecord& r) const {
    const double att = attainable_tflops(r.arithmetic_intensity());
    return att > 0 ? r.tflops() / att : 0.0;
  }
};

// Theoretical f32 FMA peak: SMs * (FP32 cores per SM) * 2 (FMA counts as 2
// FLOPs) * clock. `cores_per_sm` is NOT queryable from the CUDA API — it is a
// per-architecture table (64 on sm_70/sm_80 for f32 FMA pipes, 128 on
// sm_75/86/89/120 counting the second FP32 pipe). We keep the table explicit and
// visible rather than hidden in a helper, because using the wrong value is how
// people end up reporting 200% of peak.
[[nodiscard]] double theoretical_peak_f32_tflops(const DeviceInfo& d);

// -----------------------------------------------------------------------------
// Profiler: owns a timing-event pool and the record list.
// -----------------------------------------------------------------------------
class Profiler {
 public:
  // Time a callable that enqueues work on `stream`. Handles warmup, repeats,
  // and event bracketing. Returns the median record.
  //
  // Templated on the callable so the timed region inlines — a std::function
  // here would add an indirect call inside the timed loop, which for a 5 us
  // kernel is measurable.
  template <typename EnqueueFn>
  [[nodiscard]] StatusOr<KernelRecord> time_op(const std::string& name,
                                               const std::string& variant,
                                               const rt::Stream& stream,
                                               std::uint64_t flops, std::uint64_t bytes,
                                               int warmup, int iters, EnqueueFn&& fn);

  void add(KernelRecord r) { records_.push_back(std::move(r)); }
  [[nodiscard]] const std::vector<KernelRecord>& records() const { return records_; }

  // CSV, because RESULTS.md tables and any plotting script both want it, and
  // because a CSV diff between two commits is a readable regression report.
  [[nodiscard]] Status write_csv(const std::string& path, const Roofline& rl) const;
  [[nodiscard]] std::string summary_table(const Roofline& rl) const;

 private:
  std::vector<KernelRecord> records_;
};

// -----------------------------------------------------------------------------
// Profiler::time_op — the measurement loop, in one place so every benchmark in
// the project follows the same discipline.
//
// Structure, and why each part is there:
//   1. `warmup` untimed iterations   : pay module load / JIT, let clocks ramp,
//                                      warm the L2 with the input data.
//   2. one event pair per iteration  : we time iterations *individually* rather
//                                      than timing N back-to-back and dividing.
//                                      Dividing hides the variance, and variance
//                                      is the signal that tells you the GPU is
//                                      throttling or shared with someone else.
//   3. synchronize once at the end   : NOT inside the loop. A sync per iteration
//                                      would insert a host round-trip (~10 us)
//                                      between kernels and inflate short
//                                      kernels enormously. Events already
//                                      record device-side timestamps in stream
//                                      order, so we can read them all later.
//   4. median                        : see the header comment on distributions.
// -----------------------------------------------------------------------------
template <typename EnqueueFn>
StatusOr<KernelRecord> Profiler::time_op(const std::string& name, const std::string& variant,
                                        const rt::Stream& stream, std::uint64_t flops,
                                        std::uint64_t bytes, int warmup, int iters,
                                        EnqueueFn&& fn) {
  if (iters <= 0) return InvalidArgumentError("time_op: iters must be > 0");

  for (int i = 0; i < warmup; ++i) MCKE_RETURN_IF_ERROR(fn(stream));

  std::vector<rt::Event> starts, stops;
  starts.reserve(iters);
  stops.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto s = rt::Event::create(rt::Event::Purpose::kTiming);
    if (!s.ok()) return s.status();
    auto e = rt::Event::create(rt::Event::Purpose::kTiming);
    if (!e.ok()) return e.status();
    starts.push_back(std::move(*s));
    stops.push_back(std::move(*e));
  }

  for (int i = 0; i < iters; ++i) {
    MCKE_RETURN_IF_ERROR(starts[i].record(stream));
    MCKE_RETURN_IF_ERROR(fn(stream));
    MCKE_RETURN_IF_ERROR(stops[i].record(stream));
  }
  MCKE_RETURN_IF_ERROR(stream.synchronize());   // the ONE host barrier, at the end

  std::vector<double> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t = rt::Event::elapsed_ms(starts[i], stops[i]);
    if (!t.ok()) return t.status();
    ms.push_back(static_cast<double>(*t));
  }
  std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());

  KernelRecord r;
  r.name = name;
  r.variant = variant;
  r.ms = ms[ms.size() / 2];
  r.flops = flops;
  r.bytes = bytes;
  r.iterations = iters;
  records_.push_back(r);
  return r;
}

}  // namespace mcke
