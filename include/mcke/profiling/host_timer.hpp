// =============================================================================
//  mcke/profiling/host_timer.hpp
//
//  WHAT: Host-side latency measurement — `HostTimer` (the bracket),
//        `LatencyStats` (retained samples + percentiles + histogram), and
//        `ClockCalibration` (what this machine's clock can actually resolve).
//
//  ---------------------------------------------------------------------------
//  WHY THIS IS A SEPARATE HEADER, NOT ADDED TO profiler.hpp
//
//  profiler.hpp's own banner declares its job as GPU timing via CUDA events,
//  NVTX, and roofline math; it conditionally includes <nvtx3/nvToolsExt.h>.
//  `KernelRecord::tflops()` / `arithmetic_intensity()` / `Roofline` are all
//  roofline concepts, and an allocator's `allocate()` has no FLOPs and no
//  roofline position — it is pure host bookkeeping. Bolting a host-latency type
//  onto a GPU-roofline file would make that file's own banner false. A sibling
//  header costs one file and keeps both banners honest.
//
//  It also means this header can be included from a .cpp that never touches
//  CUDA at all (bench/alloc_bench.cpp), which is the whole point: allocator
//  latency is a host-only measurement and must work on a machine with no GPU.
//
//  ---------------------------------------------------------------------------
//  A HARD CONSTRAINT DISCOVERED WHILE DESIGNING THIS, WORTH KNOWING BEFORE YOU
//  READ A SINGLE NUMBER OUT OF IT
//
//  On Apple Silicon, `hw.tbfrequency` is 24 MHz, i.e. every host clock API
//  (`std::chrono::steady_clock`, `mach_absolute_time`, a raw `mrs cntvct_el0`)
//  ultimately reads the SAME counter, ticking every 1/24e6 s = 41.67 ns. Two
//  back-to-back calls to `now()` measure ~42 ns apart — the clock's own call
//  overhead is *itself* about one tick.
//
//  That has a consequence for how this file's numbers must be read: a fast pool
//  hit (predicted 40-150 ns) is only 1-4 ticks. The MEDIAN of such a
//  distribution will land on a small multiple of the tick (42, 83, 125 ns...)
//  and is not a physically meaningful measurement of an operation that fast —
//  it is a statement about the clock, not the allocator. The correct response
//  is NOT to "calibrate and subtract" the clock's overhead: the overhead
//  (~42 ns) is itself about one tick, so subtracting it from a one-tick sample
//  produces 0 ns, which is a worse lie than reporting 42. Instead:
//    - report the floor honestly, and mark any percentile at or below ~2 ticks
//      as instrument-limited (ClockCalibration::floor_ns, ClockCalibration::
//      is_below_floor());
//    - recover the fast-path cost a different way: bracket a whole batch of
//      operations with ONE pair of timestamps and divide (the "amortised"
//      measurement — see bench/alloc_bench.cpp). A 41.67 ns granularity over
//      100,000 ops amortises to 0.0004 ns of error;
//    - take the authoritative median/p99 from a machine with finer resolution
//      (an x86-64 box reading the TSC via vDSO typically resolves ~1 ns).
//  The TAIL (a slab growth, a rehash, a coalesce cascade — hundreds of ns to
//  tens of us) is 10-1000+ ticks and IS resolved correctly by this clock even
//  on a 24 MHz counter. That asymmetry — tail good, fast-path median coarse —
//  is why `bench/alloc_bench.cpp` reports both a per-op median/p99 AND a
//  separate amortised figure, rather than trusting the median alone.
//
//  ---------------------------------------------------------------------------
//  WHY std::chrono::steady_clock, and the trap that sounds like the right answer
//
//   - NOT system_clock: wall-clock time, adjustable by NTP. A step adjustment
//     mid-measurement yields a negative or absurd sample.
//   - NOT high_resolution_clock, despite the name: it is an alias for an
//     implementation-defined clock, and on libstdc++ it aliases system_clock —
//     exactly the one to avoid. The API that ADVERTISES resolution is the one
//     that silently hands you a non-monotonic clock. steady_clock is the only
//     one of the three the standard actually guarantees is monotonic.
//
//  ---------------------------------------------------------------------------
//  WHY INDIVIDUAL TIMING, NOT BATCH-AND-DIVIDE, FOR THE RETAINED SAMPLE
//
//  p99 exists specifically to reveal the rare expensive path (a slab growth, a
//  coalesce cascade, an unordered_map rehash). Averaging a batch of K
//  operations under one bracket provably ERASES exactly that signal: a single
//  50 us slab growth inside a batch of 32 divides down to 1.5 us and vanishes
//  into the noise floor. So every sample here is one real operation, timed
//  individually, at the cost of the fast path being quantised to the clock's
//  granularity — which is exactly why the amortised cross-check exists
//  alongside it rather than instead of it.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mcke {

// -----------------------------------------------------------------------------
// HostTimer — the bracket. Inline, deliberately, and NOT an RAII scope guard.
//
// Header-only for the same reason stream.hpp's wrappers are: this sits directly
// inside the region being timed, and a call the compiler cannot see through
// (an out-of-line function in a different translation unit) is itself a
// measurable cost at the ~100 ns scale we are trying to resolve.
//
// Not RAII: a destructor fires at a point the source text does not show and
// cannot be hoisted out of the loop by a reader's eye. Two explicit calls
// (`start()` / `stop_ns()`) keep the timed region visible exactly where it is.
// -----------------------------------------------------------------------------
class HostTimer {
 public:
  void start() noexcept { t0_ = std::chrono::steady_clock::now(); }

  [[nodiscard]] std::uint64_t stop_ns() const noexcept {
    const auto dt = std::chrono::steady_clock::now() - t0_;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
  }

 private:
  std::chrono::steady_clock::time_point t0_{};
};

// -----------------------------------------------------------------------------
// ClockCalibration — what THIS machine's clock can actually resolve, measured
// rather than assumed. See the header banner for why this matters here more
// than in most timing code: at the ~100 ns scale we're measuring, the
// difference between a 1 ns clock and a 42 ns clock is the difference between a
// meaningful median and instrument noise.
// -----------------------------------------------------------------------------
struct ClockCalibration {
  std::uint64_t tick_ns          = 0;   // smallest nonzero delta observed
  std::uint64_t paired_median_ns = 0;   // median of back-to-back now();now() calls
  std::uint64_t floor_ns         = 0;   // max(tick_ns, paired_median_ns)

  // Measures itself: `samples` back-to-back start/stop pairs with nothing
  // between them, so every nonzero delta is pure clock/call overhead.
  [[nodiscard]] static ClockCalibration measure(int samples = 100000);

  // A measurement at or below ~2 floor ticks cannot be trusted as a value in
  // its own right — it is telling you about the clock, not the operation.
  [[nodiscard]] bool is_below_floor(std::uint64_t ns) const {
    return floor_ns > 0 && ns <= 2 * floor_ns;
  }

  [[nodiscard]] std::string describe() const;
};

// -----------------------------------------------------------------------------
// LatencyStats — a RETAINED sample of durations, with exact (not approximated)
// percentiles.
//
// WHY RETAIN THE FULL SAMPLE rather than a streaming summary or a fixed-bucket
// histogram: p99 is an order statistic. Computing it exactly needs either the
// full sample or an approximation structure (t-digest, HDR histogram) — and at
// the scale this project actually uses (10^4-10^5 samples = tens to hundreds of
// KB of uint64_t), the "big data" answer is strictly worse: approximate, more
// code, for no benefit. That tradeoff flips well past 10^7-10^8 samples, where
// retention stops fitting in memory — know when the other answer becomes
// right, rather than defaulting to it.
//
// WHY NEAREST-RANK, NOT INTERPOLATED, PERCENTILES: interpolating between two
// quantised clock ticks invents precision that the instrument does not have.
// Nearest-rank means literally "at least P% of observed operations completed
// in this time or less" — a statement defensible without qualification.
// -----------------------------------------------------------------------------
class LatencyStats {
 public:
  // Reserve up front: a std::vector reallocation triggered from INSIDE a timed
  // region would itself be a malloc contaminating the very allocator-latency
  // measurement this class exists to take.
  explicit LatencyStats(std::size_t expected_count) { samples_.reserve(expected_count); }

  void add(std::uint64_t ns) { samples_.push_back(ns); }

  // Sorts the retained sample. O(n log n), and deliberately done ONCE, outside
  // any timed region, rather than via repeated std::nth_element calls (which is
  // what Profiler::time_op does for its single median) — here we want five-plus
  // different order statistics plus a histogram, and at n <= ~10^5 one sort
  // (~5 ms) is simpler code than five nth_element passes for a cost nobody
  // will notice.
  void finalize() {
    std::sort(samples_.begin(), samples_.end());
    finalized_ = true;
  }

  [[nodiscard]] std::size_t count() const { return samples_.size(); }
  [[nodiscard]] bool empty() const { return samples_.empty(); }

  // Nearest-rank percentile: index = ceil(p/100 * n) - 1, clamped into range.
  [[nodiscard]] std::uint64_t percentile(double p) const {
    if (samples_.empty()) return 0;
    const std::size_t n = samples_.size();
    long long idx = static_cast<long long>(std::ceil(p / 100.0 * static_cast<double>(n))) - 1;
    if (idx < 0) idx = 0;
    if (idx >= static_cast<long long>(n)) idx = static_cast<long long>(n) - 1;
    return samples_[static_cast<std::size_t>(idx)];
  }

  [[nodiscard]] std::uint64_t median() const { return percentile(50.0); }
  [[nodiscard]] std::uint64_t p90() const { return percentile(90.0); }
  [[nodiscard]] std::uint64_t p99() const { return percentile(99.0); }
  [[nodiscard]] std::uint64_t p999() const { return percentile(99.9); }
  [[nodiscard]] std::uint64_t min() const { return samples_.empty() ? 0 : samples_.front(); }
  [[nodiscard]] std::uint64_t max() const { return samples_.empty() ? 0 : samples_.back(); }

  [[nodiscard]] double mean_ns() const {
    if (samples_.empty()) return 0.0;
    double sum = 0.0;
    for (std::uint64_t v : samples_) sum += static_cast<double>(v);
    return sum / static_cast<double>(samples_.size());
  }

  // Log2-bucketed histogram: [0, floor_ns], (floor, 2*floor], (2*floor, 4*floor],
  // ... This is the single most informative artifact in the whole bench, because
  // it separates "unimodal at the clock floor" from "bimodal — a fast path plus
  // an occasional slab growth" in a way a single median number cannot.
  [[nodiscard]] std::string log2_histogram(std::uint64_t floor_ns) const;

  [[nodiscard]] bool is_finalized() const { return finalized_; }

 private:
  std::vector<std::uint64_t> samples_;
  bool finalized_ = false;
};

}  // namespace mcke
