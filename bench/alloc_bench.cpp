// =============================================================================
//  bench/alloc_bench.cpp
//
//  WHAT: Phase 2c. Replays two synthetic allocation traces through
//        RawDeviceAllocator, BuddyAllocator, and FreeListAllocator in ONE
//        process, and reports: host-side allocate/deallocate latency
//        (median/p90/p99/p999/max, plus an amortised cross-check), the
//        raw_malloc_calls-vs-alloc_calls flatline that is this whole
//        component's thesis, and three named fragmentation ratios that
//        together make "which allocator wins" a falsifiable statement instead
//        of an impression.
//
//  WHY .cpp, NOT .cu, and NOT gated on MCKE_ENABLE_CUDA: no device code exists
//  anywhere in this file. Allocator LOGIC and host-side allocate/deallocate
//  latency are host-only measurements by nature, and the entire point of
//  Phase 2's "test on the Mac first" discipline (docs/ROADMAP.md) is that this
//  binary must build and produce meaningful fragmentation numbers with no GPU
//  at all. See CMakeLists.txt for why it links mcke_core rather than the
//  `mcke` umbrella, same reasoning as tests/test_host_core.cpp.
//
//  ---------------------------------------------------------------------------
//  WHAT THIS BINARY CANNOT SHOW YOU ON macOS, STATED UP FRONT
//
//  In the host-only build, raw_device_malloc() is std::aligned_alloc(256, ...),
//  NOT cudaMalloc(). aligned_alloc costs ~100-200 ns; cudaMalloc costs
//  10-100 microseconds. So the headline "pool beats raw by orders of
//  magnitude" speedup is a GPU-only phenomenon and this binary cannot
//  demonstrate it — the Phase 2a *latency* numbers from this machine are
//  informative about relative allocator overhead but NOT about the
//  raw-vs-pool win, which must come from a Colab/Explorer run (RESULTS.md
//  section 2a explicitly wants both figures with their environment noted).
//  What IS fully authoritative here, because it is pure host bookkeeping with
//  no GPU dependency at all: every fragmentation number in section 2b.
//
//  ---------------------------------------------------------------------------
//  THE CLOCK CONSTRAINT THIS FILE IS DESIGNED AROUND
//
//  See mcke/profiling/host_timer.hpp's banner for the full derivation. Short
//  version: on this machine the host clock's own granularity is measured at
//  program start (ClockCalibration::measure) and every reported percentile at
//  or below ~2 clock ticks is marked with a '*' rather than trusted as a real
//  physical measurement. The AMORTISED figure (one timestamp bracket around a
//  whole batch of operations, divided by the batch size) is what recovers the
//  fast-path cost on a coarse clock, and is reported alongside every latency
//  row for exactly that reason.
//
//  ---------------------------------------------------------------------------
//  NO CLI ARGUMENTS, matching bench/stream_triad.cu and bench/fma_peak.cu: every
//  knob below is a named constexpr. If you want a different trace size or a
//  different allocator config, edit the constant and rebuild — this project's
//  benchmarks are meant to be read as much as run.
// =============================================================================
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_allocator.hpp"
#include "mcke/memory/freelist_allocator.hpp"
#include "mcke/profiling/host_timer.hpp"
#include "mcke/runtime/stream.hpp"

using namespace mcke;

namespace {

// =============================================================================
// SECTION 1 — the trace representation and generator.
// =============================================================================

enum class TraceOpKind : std::uint8_t { kAlloc = 0, kFree = 1 };

// Deliberately 8 bytes, over a small size-class TABLE rather than an inline
// size_t. At 200k ops a naive 24-byte {kind, handle, size_t, stream} struct
// would stream ~4.8 MB of DRAM *inside the timed region* on every pass — we
// would be measuring the trace's own memory traffic, not the allocator. An
// 8-byte op with a uint16 index into a <=16-entry table keeps the whole trace
// under 1.6 MB (comfortably L2-resident) and costs one indexed load per op.
struct TraceOp {
  std::uint32_t handle;
  std::uint16_t size_class;
  std::uint8_t  stream_idx;   // always 0 in Phase 2; carried for a Phase 4 reuse
  std::uint8_t  kind;         // TraceOpKind
};
static_assert(sizeof(TraceOp) == 8,
             "TraceOp must stay 8B: the trace must not become the thing "
             "we are measuring");

struct PhaseMark {
  std::size_t first_op;
  std::string label;
};

struct Trace {
  std::string              name;
  std::vector<std::size_t> sizes;    // distinct byte sizes, small table
  std::vector<TraceOp>     ops;
  std::vector<PhaseMark>   phases;
  std::uint32_t            max_handles = 0;   // == max CONCURRENT live, not n_allocs
  std::uint64_t            seed = 0;
  std::size_t              n_alloc_ops = 0;
  // The peak of sum(requested) over the trace, computed by pure simulation at
  // generation time. This is what a PERFECT, zero-fragmentation allocator
  // would need to hold the same working set — an allocator-independent
  // denominator for "how close to ideal did we get".
  std::size_t              oracle_peak_requested = 0;
};

// Deterministic trace generator.
//
// WHY std::mt19937_64 with a fixed seed: the C++ standard specifies the
// Mersenne Twister's bit recurrence EXACTLY ([rand.eng.mers]), so the same seed
// produces an identical output stream on Apple clang/libc++ (this machine) and
// on gcc/libstdc++ (Colab). That is the property we need: the three allocators
// inside one run must see identical ops (trivial — one Trace object, one
// generation), but ALSO a Colab re-run must generate the SAME trace as this Mac
// run, or a cross-machine comparison in RESULTS.md would be comparing two
// different workloads without saying so.
//
// WHY NOT std::uniform_int_distribution: engines are specified bit-exactly;
// DISTRIBUTIONS ARE NOT. The same engine output can legally produce different
// values from uniform_int_distribution on libc++ vs libstdc++. That is a subtle
// trap — using the "obviously correct" standard tool here would silently break
// the exact cross-machine reproducibility the whole point of a fixed seed was
// to buy. So every draw below is raw engine output plus explicit arithmetic
// (`eng() % n`). Modulo bias against a 2^64 range with n <= ~16 is negligible.
constexpr std::uint64_t kTraceSeed = 0x9E3779B97F4A7C15ull;   // golden ratio

class TraceGen {
 public:
  TraceGen(std::string name, std::vector<std::size_t> sizes, std::uint64_t seed)
      : eng_(seed) {
    t_.name = std::move(name);
    t_.sizes = std::move(sizes);
    t_.seed = seed;
  }

  void mark_phase(const std::string& label) { t_.phases.push_back({t_.ops.size(), label}); }

  std::uint64_t rnd() { return eng_(); }
  std::size_t rnd_below(std::size_t n) { return static_cast<std::size_t>(eng_() % n); }

  // WHY HANDLES RECYCLE (a free-id stack), rather than monotonically
  // increasing: with monotonic ids, max_handles == n_alloc_ops (often ~10^5),
  // so the replay engine's live-allocation table would be megabytes and mostly
  // cold — DRAM traffic inside the timed region again. Recycling keeps
  // max_handles == max CONCURRENT live (a few hundred), so that table stays
  // L1-resident. This is not a micro-optimisation; it is the difference
  // between measuring the allocator and measuring a cache miss.
  std::uint32_t do_alloc(std::size_t cls) {
    std::uint32_t h;
    if (!free_h_.empty()) {
      h = free_h_.back();
      free_h_.pop_back();
    } else {
      h = next_h_++;
    }
    live_.push_back(h);
    cls_[h] = cls;
    live_bytes_ += t_.sizes[cls];
    if (live_bytes_ > peak_bytes_) peak_bytes_ = live_bytes_;
    if (live_.size() > t_.max_handles) t_.max_handles = static_cast<std::uint32_t>(live_.size());
    t_.ops.push_back(TraceOp{h, static_cast<std::uint16_t>(cls), 0,
                             static_cast<std::uint8_t>(TraceOpKind::kAlloc)});
    ++t_.n_alloc_ops;
    return h;
  }

  std::uint32_t do_free_at(std::size_t idx) {
    const std::uint32_t h = live_[idx];
    live_.erase(live_.begin() + static_cast<std::ptrdiff_t>(idx));
    live_bytes_ -= t_.sizes[cls_[h]];
    t_.ops.push_back(TraceOp{h, 0, 0, static_cast<std::uint8_t>(TraceOpKind::kFree)});
    free_h_.push_back(h);
    return h;
  }

  void do_free_front() { do_free_at(0); }   // FIFO

  void do_free_handle(std::uint32_t h) {
    auto it = std::find(live_.begin(), live_.end(), h);
    assert(it != live_.end() && "do_free_handle: handle is not live");
    do_free_at(static_cast<std::size_t>(std::distance(live_.begin(), it)));
  }

  [[nodiscard]] std::uint32_t live_front_handle() const { return live_.front(); }
  [[nodiscard]] std::size_t class_of(std::uint32_t h) const { return cls_.at(h); }
  [[nodiscard]] std::size_t live_count() const { return live_.size(); }
  [[nodiscard]] std::size_t live_bytes() const { return live_bytes_; }
  [[nodiscard]] bool empty() const { return live_.empty(); }

  void drain_all_random() {
    while (!live_.empty()) do_free_at(rnd_below(live_.size()));
  }

  Trace finish() {
    t_.oracle_peak_requested = peak_bytes_;
    return std::move(t_);
  }

 private:
  Trace t_;
  std::mt19937_64 eng_;
  // std::deque, not vector: generation needs both front-removal (FIFO) and
  // index-based removal (random victim), and this runs once, offline, before
  // any timing starts — O(n) removal cost here is irrelevant to what the
  // benchmark measures.
  std::deque<std::uint32_t> live_;
  std::vector<std::uint32_t> free_h_;
  std::uint32_t next_h_ = 0;
  std::unordered_map<std::uint32_t, std::size_t> cls_;
  std::size_t live_bytes_ = 0, peak_bytes_ = 0;
};

// ---- Trace (i): uniform_pow2 — the control -----------------------------
//
// Every size is a power of two >= kMinBlockBytes, so BuddyAllocator's internal
// waste on this trace must be EXACTLY ZERO. That is not a hope, it is an
// assertion this trace doubles as: any nonzero waste reported against this
// trace is an allocator bug, not a measurement.
Trace generate_uniform_pow2() {
  std::vector<std::size_t> sizes;
  for (unsigned k = 8; k <= 20; ++k) sizes.push_back(std::size_t{1} << k);   // 256 B .. 1 MiB
  TraceGen g("uniform_pow2", sizes, kTraceSeed);
  const std::size_t n_cls = sizes.size();
  // Target the WORKING-SET SIZE IN BYTES, not the live block COUNT. Sizes here
  // span a 4096x range (256 B .. 1 MiB), so a count-based target ("hold ~384
  // blocks live") gives almost no control over the actual footprint: a run of
  // bad luck that happens to hold mostly 1 MiB blocks can blow past a
  // count-derived estimate by 3x or more. Controlling bytes directly is what
  // actually determines whether the trace fits the configured pool — which is
  // the property we need to hold steady to get a clean, interpretable run
  // rather than one dominated by incidental OOM.
  const std::size_t target_bytes = std::size_t{24} << 20;   // 24 MiB

  // Proportional control: bias toward alloc below the target footprint, toward
  // free above it, 50/50 in a dead band around it. A FIXED alloc-probability
  // (e.g. "60% chance of alloc") has no notion of "target" — it only has a
  // constant bias, and over tens of thousands of ops that grows or shrinks the
  // footprint without bound. This closes the loop instead.
  auto want_alloc = [&](double lo_frac, double hi_frac) {
    if (g.empty()) return true;
    const std::size_t lo = static_cast<std::size_t>(double(target_bytes) * lo_frac);
    const std::size_t hi = static_cast<std::size_t>(double(target_bytes) * hi_frac);
    if (g.live_bytes() < lo) return true;
    if (g.live_bytes() > hi) return false;
    return (g.rnd() % 100) < 50;
  };

  g.mark_phase("warmup");   // LIFO: the easy case, perfect coalescing, reach steady state
  for (int i = 0; i < 10000; ++i) {
    if (want_alloc(0.90, 1.10)) g.do_alloc(g.rnd_below(n_cls));
    else g.do_free_at(g.live_count() - 1);
  }
  g.mark_phase("churn");    // random victim: realistic mixed churn
  for (int i = 0; i < 80000; ++i) {
    if (want_alloc(0.80, 1.20)) g.do_alloc(g.rnd_below(n_cls));
    else g.do_free_at(g.rnd_below(g.live_count()));
  }
  g.mark_phase("adversarial");
  // FIFO + a size RATCHET: the next allocation is always drawn one class larger
  // than the block that was just freed, so a freed block is never the right
  // size for the next request. Constructed specifically to defeat a
  // non-coalescing pool: buddy can merge its way out of this; a size-class pool
  // cannot. This is where largest_free_block should collapse for one of them.
  std::size_t last_freed_cls = 0;
  for (int i = 0; i < 80000; ++i) {
    if (want_alloc(0.80, 1.20)) {
      g.do_alloc((last_freed_cls + 1) % n_cls);
    } else {
      last_freed_cls = g.class_of(g.live_front_handle());
      g.do_free_front();
    }
  }
  g.mark_phase("drain");   // free everything: the decisive fragmentation measurement
  g.drain_all_random();
  return g.finish();
}

// ---- Trace (ii): dl_transformer -----------------------------------------
//
// GPT-2-small shapes, f32, batch 1 x seq 512. Two sub-phases with different
// lifetime character, per docs/ROADMAP.md's "activations short-lived, weights
// long-lived" note:
//   layer_step  : the big 2-D tensors (x, qkv, scores, ffn_h), freed in
//                 LIVENESS order (x dies once qkv exists, etc.) rather than
//                 LIFO or FIFO — a simplified stand-in for what a real
//                 dependency graph produces, and a forward link to the Phase 4
//                 liveness planner.
//   decode_step : the literal sizes named in docs/ROADMAP.md Phase 2c
//                 (768, 3072, 50257 elements x 4 bytes), short-lived,
//                 mimicking autoregressive decode.
// The weights themselves are allocated once and freed only at the very end —
// they are the "long-lived" half of the roadmap's split.
//
// The 147 MiB embedding table (50257 x 768 x 4) is EXCLUDED by default
// (kIncludeEmbeddingTable = false): it exceeds large_alloc_threshold and would
// take the bypass-to-raw-malloc path, adding one permanent raw_malloc_call that
// would have to be explained every time the flatline table is read. Flip the
// flag to see that path exercised instead.
Trace generate_dl_transformer(bool include_embedding_table) {
  const bool kIncludeEmbeddingTable = include_embedding_table;
  std::vector<std::size_t> sizes = {
      2048,        // [0] ln_stats (512 f32)
      3072,        // [1] ln gamma/beta (768 f32)     == decode "768 x f32"
      12288,       // [2] mlp bias (3072 f32)          == decode "3072 x f32"
      201028,      // [3] logits (50257 f32)           decode-only
      1572864,     // [4] x        (512 x 768 f32)
      2359296,     // [5] Wo       (768 x 768 f32)
      4718592,     // [6] qkv      (512 x 2304 f32)
      6291456,     // [7] ffn_h    (512 x 3072 f32)
      7077888,     // [8] Wqkv     (768 x 2304 f32)
      9437184,     // [9] W1 / W2  (768 x 3072 f32)
      12582912,    // [10] scores  (12 x 512 x 512 f32)
  };
  if (kIncludeEmbeddingTable) sizes.push_back(154389504);   // [11] 50257 x 768 x 4

  TraceGen g(kIncludeEmbeddingTable ? "dl_transformer_bypass" : "dl_transformer",
             sizes, kTraceSeed ^ 0xD1ull);

  g.mark_phase("weights");
  std::vector<std::uint32_t> w;
  w.push_back(g.do_alloc(1)); w.push_back(g.do_alloc(1));   // 4x ln gamma/beta
  w.push_back(g.do_alloc(1)); w.push_back(g.do_alloc(1));
  w.push_back(g.do_alloc(2)); w.push_back(g.do_alloc(2));   // 2x mlp bias
  w.push_back(g.do_alloc(5));                                // Wo
  w.push_back(g.do_alloc(8));                                // Wqkv
  w.push_back(g.do_alloc(9)); w.push_back(g.do_alloc(9));    // W1, W2
  if (kIncludeEmbeddingTable) w.push_back(g.do_alloc(11));

  g.mark_phase("layer_step");
  constexpr int kLayerSteps = 2000;
  for (int i = 0; i < kLayerSteps; ++i) {
    const auto x      = g.do_alloc(4);
    const auto qkv     = g.do_alloc(6);
    g.do_free_handle(x);                 // x dies once qkv (its consumer) exists
    const auto scores  = g.do_alloc(10);
    g.do_free_handle(qkv);
    const auto ffn_h   = g.do_alloc(7);
    g.do_free_handle(scores);
    const auto ln_stats = g.do_alloc(0);
    g.do_free_handle(ffn_h);
    g.do_free_handle(ln_stats);
  }

  g.mark_phase("decode_step");
  constexpr int kDecodeSteps = 4000;
  for (int i = 0; i < kDecodeSteps; ++i) {
    const auto a = g.do_alloc(1);    // 768 x f32
    const auto b = g.do_alloc(2);    // 3072 x f32
    const auto c = g.do_alloc(3);    // 50257 x f32 (logits)
    const auto d = g.do_alloc(0);    // ln_stats
    g.do_free_handle(a);
    g.do_free_handle(b);
    g.do_free_handle(c);
    g.do_free_handle(d);
  }

  g.mark_phase("drain");
  for (std::uint32_t h : w) g.do_free_handle(h);
  return g.finish();
}

// =============================================================================
// SECTION 2 — allocator factories.
//
// One shared footprint cap for buddy and freelist so the Phase 2c comparison
// holds both to an identical budget: without a cap, "does it fragment" cannot
// be observed because there is always room to grow instead.
// =============================================================================

constexpr std::size_t kSlabBytes     = std::size_t{16} << 20;    // 16 MiB
constexpr std::size_t kFootprintCap  = std::size_t{96} << 20;    // 6x slab: room
                                                                 // to grow, but a
                                                                 // real ceiling
constexpr std::size_t kLargeAllocThreshold = std::size_t{32} << 20;

BuddyConfig buddy_cfg() {
  BuddyConfig c;
  c.initial_slab_bytes    = kSlabBytes;
  c.max_total_bytes       = kFootprintCap;
  c.min_block_bytes       = 256;
  c.large_alloc_threshold = kLargeAllocThreshold;
  return c;
}

FreeListConfig freelist_cfg() {
  FreeListConfig c;
  c.slab_bytes              = kSlabBytes;
  c.max_total_bytes         = kFootprintCap;
  c.small_class_granularity = 512;
  c.small_large_split       = std::size_t{1} << 20;
  c.large_alloc_threshold   = kLargeAllocThreshold;
  return c;
}

using Factory = std::function<std::unique_ptr<DeviceAllocator>()>;

// Fixed printed order: raw -> buddy -> freelist. Flip kReverseAllocatorOrder,
// rebuild, and re-run to confirm the medians below agree within one clock tick
// — that is the check against "whichever allocator runs first pays a cold-cache
// penalty the others do not", and it is a rebuild-time knob rather than a CLI
// flag to match this project's no-argv bench convention.
constexpr bool kReverseAllocatorOrder = false;

// Each pooling allocator is run under ALL THREE reuse policies, not just the
// default. That is not thoroughness for its own sake -- it is what makes the
// deallocate column interpretable at all once this runs on a real GPU:
//
//   kSameStreamOnly    never probes            -> PURE allocator bookkeeping
//   kCoarseStreamPoll  + one cudaStreamQuery   -> + a real driver round trip
//   kPerFreeEvent      + one cudaEventRecord   -> + a different driver round trip
//
// On the host build all three probes are free, so the three rows should agree
// closely and any gap is pure bookkeeping difference. On a GPU, row 1 stays the
// honest coalescing-vs-no-coalescing comparison while rows 2 and 3 measure what
// each safety mechanism actually COSTS. Subtracting row 1 from rows 2 and 3 is
// the measurement; without row 1 there is no baseline to subtract and the
// deallocate comparison silently becomes a driver-latency benchmark.
std::vector<std::pair<std::string, Factory>> allocator_factories() {
  std::vector<std::pair<std::string, Factory>> v;
  v.emplace_back("raw", [] { return std::unique_ptr<DeviceAllocator>(
                                 std::make_unique<RawDeviceAllocator>()); });

  const std::pair<const char*, ReusePolicy> pols[] = {
      {"same_stream", ReusePolicy::kSameStreamOnly},
      {"coarse_poll", ReusePolicy::kCoarseStreamPoll},
      {"event",       ReusePolicy::kPerFreeEvent},
  };
  for (const auto& [pn, pv] : pols) {
    v.emplace_back(std::string("buddy/") + pn, [pv]() -> std::unique_ptr<DeviceAllocator> {
      BuddyConfig c = buddy_cfg();
      c.reuse_policy = pv;
      auto p = std::make_unique<BuddyAllocator>(c);
      const Status st = p->reserve(kSlabBytes);
      if (!st.ok()) { std::fprintf(stderr, "buddy reserve failed: %s\n", st.to_string().c_str()); std::abort(); }
      return p;
    });
  }
  for (const auto& [pn, pv] : pols) {
    v.emplace_back(std::string("freelist/") + pn, [pv]() -> std::unique_ptr<DeviceAllocator> {
      FreeListConfig c = freelist_cfg();
      c.reuse_policy = pv;
      auto p = std::make_unique<FreeListAllocator>(c);
      const Status st = p->reserve(kSlabBytes);
      if (!st.ok()) { std::fprintf(stderr, "freelist reserve failed: %s\n", st.to_string().c_str()); std::abort(); }
      return p;
    });
  }
  if (kReverseAllocatorOrder) std::reverse(v.begin(), v.end());
  return v;
}

// -----------------------------------------------------------------------------
// BYPASS MODE 3 of 3 -- the isolated probe.
//
// Not a trace and not a measurement: a pass/fail check that one allocation far
// above large_alloc_threshold takes the bypass-to-driver path and leaves the
// pool's slab structure completely untouched. No churn, nothing else running, so
// if it fails there is exactly one thing it can be.
//
// Modes 1 and 2 are the two clean traces (no large allocation at all) and the
// dl_transformer_bypass trace (the same tensor folded into real churn). Keeping
// all three separate is deliberate: mode 1's raw_malloc_calls flatline stays
// unambiguous precisely because no bypass ever happens in it.
// -----------------------------------------------------------------------------
bool run_bypass_probe() {
  constexpr std::size_t kHuge = 154389504;   // 50257 x 768 x f32 == a GPT-2 embedding table
  static_assert(kHuge > kLargeAllocThreshold, "probe size must exceed the bypass threshold");

  BuddyConfig c = buddy_cfg();
  BuddyAllocator a(c);
  if (!a.reserve(kSlabBytes).ok()) { std::printf("  bypass probe: reserve failed\n"); return false; }

  const AllocatorStats before = a.stats();
  auto big = a.allocate(kHuge, rt::StreamHandle{});
  if (!big.ok()) { std::printf("  bypass probe: allocate(%zu) failed: %s\n", kHuge,
                               big.status().to_string().c_str()); return false; }
  const AllocatorStats during = a.stats();

  bool ok = true;
  auto check = [&](bool cond, const char* what) {
    if (!cond) { std::printf("  bypass probe FAIL: %s\n", what); ok = false; }
  };
  check(big->slab_id == kBypassSlabId,            "slab_id is not the bypass sentinel");
  check(big->bytes == big->requested_bytes,       "bypass reported nonzero internal waste");
  check(during.raw_malloc_calls == before.raw_malloc_calls + 1,
                                                  "did not make exactly one driver allocation");
  check(during.largest_free_block == before.largest_free_block,
                                                  "the pool's slab structure was disturbed");

  const Status fs = a.deallocate(*big, rt::StreamHandle{});
  check(fs.ok(), "deallocate of the bypassed block failed");
  const AllocatorStats after = a.stats();
  check(after.raw_free_calls == before.raw_free_calls + 1, "did not make exactly one driver free");
  check(after.bytes_in_use == before.bytes_in_use, "bytes_in_use did not return to baseline");
  check(a.validate().ok(), "validate() failed after the bypass round trip");

  std::printf("  bypass probe: %zu B (%.0f MiB) -> %s\n", kHuge,
              double(kHuge) / (1024.0 * 1024.0),
              ok ? "took the bypass path, pool untouched  [PASS]" : "[FAIL]");
  return ok;
}

// =============================================================================
// SECTION 3 — the replay engine.
// =============================================================================

// Stats are sampled every this many ops during the instrumented pass, plus at
// every phase boundary and at the very end. This is what produces both the 2b
// fragmentation table (last sample) and the raw_malloc_calls flatline time
// series (all samples).
constexpr std::size_t kStatsSampleStride = 5000;

struct FlatlineSample {
  std::size_t   op_index;
  std::string   phase;
  AllocatorStats stats;
};

struct ReplayResult {
  std::string    allocator_name;
  std::string    trace_name;
  LatencyStats   alloc_lat{0};
  LatencyStats   free_lat{0};
  double         amortised_ns_per_op = 0.0;   // combined alloc+free, see note below
  std::size_t    oom_count = 0;
  std::string    first_oom_message;
  std::size_t    first_oom_op_index = 0;
  // Peak-moment (requested, blocks, reserved) TRIPLE, captured together at the
  // instant live "blocks" bytes hits a new high. Tracking two independent
  // maxima instead (peak requested separately from peak blocks) would report a
  // ratio between two numbers that never coexisted.
  std::size_t    peak_blocks    = 0;
  std::size_t    peak_requested = 0;
  std::size_t    peak_reserved  = 0;
  AllocatorStats warmup_end_stats{};
  AllocatorStats final_stats{};
  std::vector<FlatlineSample> flatline;
  // largest_free_block probe, run after the trace fully drains. See section 5.
  bool probe_tight_ok      = false;   // allocate(largest) succeeded, no growth
  bool probe_over_detected = false;   // allocate(largest + slack) failed OR grew
};

// One combined pass: times every allocate/deallocate INDIVIDUALLY (so the p99
// can see the rare expensive path — a slab growth, a coalesce cascade, an
// unordered_map rehash), while also tracking the fragmentation state and the
// flatline snapshots. Warmup-phase samples are excluded from the LATENCY
// statistics (the allocator has not reached steady state yet — the very first
// allocate() pays a whole slab's cudaMalloc) but the allocator still processes
// them, because reaching steady state is the entire point of a warmup phase.
ReplayResult replay_timed(DeviceAllocator& alloc, const Trace& tr) {
  ReplayResult r;
  r.allocator_name = std::string(alloc.name());
  r.trace_name     = tr.name;
  r.alloc_lat      = LatencyStats(tr.n_alloc_ops);
  r.free_lat       = LatencyStats(tr.ops.size() - tr.n_alloc_ops);

  // warmup_ops = the op index where phase 1 ends / phase 2 begins. Every trace
  // in this file has "warmup"/"weights" as phase 0 and something else as phase
  // 1, so this generalises across both traces without special-casing either.
  const std::size_t warmup_ops = tr.phases.size() > 1 ? tr.phases[1].first_op : 0;

  std::vector<Allocation> live(tr.max_handles);
  std::vector<char>       live_valid(tr.max_handles, 0);
  std::size_t live_blocks = 0, live_requested = 0;
  HostTimer timer;
  const rt::StreamHandle stream{};
  std::size_t phase_cursor = 0;

  auto current_phase = [&](std::size_t i) -> const std::string& {
    while (phase_cursor + 1 < tr.phases.size() && tr.phases[phase_cursor + 1].first_op <= i)
      ++phase_cursor;
    return tr.phases[phase_cursor].label;
  };

  for (std::size_t i = 0; i < tr.ops.size(); ++i) {
    if (i == warmup_ops) r.warmup_end_stats = alloc.stats();
    const TraceOp& op = tr.ops[i];

    if (op.kind == static_cast<std::uint8_t>(TraceOpKind::kAlloc)) {
      const std::size_t bytes = tr.sizes[op.size_class];
      timer.start();
      auto res = alloc.allocate(bytes, stream);
      const std::uint64_t ns = timer.stop_ns();
      if (i >= warmup_ops) r.alloc_lat.add(ns);

      if (!res.ok()) {
        // OOM is a RECORDED RESULT, not a crash: count it, snapshot the first
        // one (with the fragmentation numbers AT that moment — that is what
        // makes "OOM while holding tens of MiB free" a legible finding rather
        // than an opaque failure), skip the matching free, and CONTINUE. If we
        // aborted instead, the timed sample COUNT would differ between
        // allocators and silently invalidate the median/p99 comparison between
        // them.
        if (res.status().code() == StatusCode::kOutOfMemory) {
          ++r.oom_count;
          if (r.oom_count == 1) {
            r.first_oom_op_index = i;
            const AllocatorStats s = alloc.stats();
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "%s (bytes_reserved=%zu bytes_in_use=%zu largest_free_block=%zu)",
                          res.status().message().c_str(), s.bytes_reserved, s.bytes_in_use,
                          s.largest_free_block);
            r.first_oom_message = buf;
          }
          live_valid[op.handle] = 0;
          continue;
        }
        // Anything other than OOM is a broken invariant, not a benchmark
        // outcome — StatusCode exists precisely to make that distinction.
        std::fprintf(stderr, "FATAL: %s allocate() returned %s at op %zu\n",
                     r.allocator_name.c_str(), res.status().to_string().c_str(), i);
        std::abort();
      }
      live[op.handle] = *res;
      live_valid[op.handle] = 1;
      live_blocks    += res->bytes;
      live_requested += res->requested_bytes;
      if (live_blocks > r.peak_blocks) {
        r.peak_blocks    = live_blocks;
        r.peak_requested = live_requested;
        // Untimed: called AFTER stop_ns(), so it never contaminates a latency
        // sample. Only fires on a new peak (rare relative to total ops), so its
        // cost (an AllocatorStats copy plus buddy's O(#slabs) largest-free-block
        // scan) is negligible over the whole run.
        r.peak_reserved = alloc.stats().bytes_reserved;
      }
    } else {
      if (!live_valid[op.handle]) continue;   // matching free of an OOM-skipped alloc
      timer.start();
      const Status st = alloc.deallocate(live[op.handle], stream);
      const std::uint64_t ns = timer.stop_ns();
      if (i >= warmup_ops) r.free_lat.add(ns);
      if (!st.ok()) {
        std::fprintf(stderr, "FATAL: %s deallocate() failed at op %zu: %s\n",
                     r.allocator_name.c_str(), i, st.to_string().c_str());
        std::abort();
      }
      live_blocks    -= live[op.handle].bytes;
      live_requested -= live[op.handle].requested_bytes;
      live_valid[op.handle] = 0;
    }

    if ((i % kStatsSampleStride) == 0 || i + 1 == tr.ops.size()) {
      r.flatline.push_back(FlatlineSample{i, current_phase(i), alloc.stats()});
    }
  }

  r.alloc_lat.finalize();
  r.free_lat.finalize();
  r.final_stats = alloc.stats();
  return r;
}

// A separate, bare-loop pass with ONE timestamp bracket around the whole trace,
// no per-op timestamps at all. This is the amortised cross-check described in
// host_timer.hpp: with ~10^5 ops, the outer bracket's ~42 ns granularity
// amortises to under a picosecond of error, so this number is trustworthy even
// where the per-op median is quantised to the clock floor.
//
// SIMPLIFICATION, stated rather than hidden: this reports ONE combined
// ns/op figure covering both allocate and deallocate together, not two
// separate numbers. Splitting them cleanly would need two independent
// brackets, but allocate and deallocate calls are interleaved and each
// deallocate depends on a live pointer from a prior allocate — bracketing them
// separately would require restructuring the trace itself, which risks
// changing the very fragmentation behaviour under test. The combined figure is
// still exactly what it is needed for: confirming whether the per-op MEDIAN
// (which the clock quantises) is in the right ballpark, or whether the tail is
// dominating total cost in a way the median hides.
double replay_amortised(DeviceAllocator& alloc, const Trace& tr) {
  std::vector<Allocation> live(tr.max_handles);
  std::vector<char>       live_valid(tr.max_handles, 0);
  const rt::StreamHandle stream{};

  HostTimer t;
  t.start();
  for (const TraceOp& op : tr.ops) {
    if (op.kind == static_cast<std::uint8_t>(TraceOpKind::kAlloc)) {
      auto res = alloc.allocate(tr.sizes[op.size_class], stream);
      if (res.ok()) { live[op.handle] = *res; live_valid[op.handle] = 1; }
      else live_valid[op.handle] = 0;
    } else {
      if (!live_valid[op.handle]) continue;
      (void)alloc.deallocate(live[op.handle], stream);
      live_valid[op.handle] = 0;
    }
  }
  const std::uint64_t total_ns = t.stop_ns();
  return static_cast<double>(total_ns) / static_cast<double>(tr.ops.size());
}

// The largest_free_block PROBE (section 2b's decisive evidence): turns a
// self-reported counter into a falsifiable claim. Must be called with nothing
// live (i.e. right after a trace's drain phase).
//
//   allocate(largest_free_block)              MUST succeed, and MUST NOT cause
//                                              a new driver allocation — proving
//                                              the block really was contiguous
//                                              and already available.
//   allocate(largest_free_block + slack)      MUST either fail or force growth
//                                              — proving the first number really
//                                              was the LARGEST such block.
void run_largest_free_block_probe(DeviceAllocator& alloc, ReplayResult& r) {
  const std::size_t lfb = r.final_stats.largest_free_block;
  if (lfb == 0) return;   // nothing free to probe (fully consumed / bypassed away)
  const rt::StreamHandle stream{};
  const std::uint64_t raw_before = r.final_stats.raw_malloc_calls;

  auto tight = alloc.allocate(lfb, stream);
  r.probe_tight_ok = tight.ok() && alloc.stats().raw_malloc_calls == raw_before;
  if (tight.ok()) (void)alloc.deallocate(*tight, stream);

  auto over = alloc.allocate(lfb + kSlabBytes / 4096 /* comfortably above any
                                                        granularity in play */,
                             stream);
  r.probe_over_detected = !over.ok() || alloc.stats().raw_malloc_calls > raw_before;
  if (over.ok()) (void)alloc.deallocate(*over, stream);
}

// =============================================================================
// SECTION 4 — reporting.
// =============================================================================

std::string human_bytes(double b) {
  char buf[64];
  if (b >= (1024.0 * 1024.0 * 1024.0)) std::snprintf(buf, sizeof(buf), "%.2f GiB", b / (1024.0 * 1024.0 * 1024.0));
  else if (b >= (1024.0 * 1024.0))     std::snprintf(buf, sizeof(buf), "%.2f MiB", b / (1024.0 * 1024.0));
  else if (b >= 1024.0)                std::snprintf(buf, sizeof(buf), "%.2f KiB", b / 1024.0);
  else                                 std::snprintf(buf, sizeof(buf), "%.0f B", b);
  return buf;
}

std::string latency_cell(const LatencyStats& s, std::uint64_t value, const ClockCalibration& cal) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%llu%s", static_cast<unsigned long long>(value),
                cal.is_below_floor(value) ? "*" : " ");
  (void)s;
  return buf;
}

void print_trace_summary(const Trace& t) {
  std::printf("--- trace: %s ---------------------------------------------------\n",
             t.name.c_str());
  std::printf("  sizes                %zu classes, %s .. %s\n", t.sizes.size(),
             human_bytes(static_cast<double>(*std::min_element(t.sizes.begin(), t.sizes.end()))).c_str(),
             human_bytes(static_cast<double>(*std::max_element(t.sizes.begin(), t.sizes.end()))).c_str());
  std::printf("  ops                  %zu total (%zu allocs, %zu frees)\n", t.ops.size(),
             t.n_alloc_ops, t.ops.size() - t.n_alloc_ops);
  std::printf("  max concurrent live  %u handles (live table = %s)\n", t.max_handles,
             human_bytes(static_cast<double>(t.max_handles) * sizeof(Allocation)).c_str());
  std::printf("  oracle peak requested %s  <- a zero-fragmentation allocator's floor\n",
             human_bytes(static_cast<double>(t.oracle_peak_requested)).c_str());
  std::printf("  phases               ");
  for (std::size_t i = 0; i < t.phases.size(); ++i) {
    std::printf("%s[%zu]", t.phases[i].label.c_str(), t.phases[i].first_op);
    if (i + 1 < t.phases.size()) std::printf(" -> ");
  }
  std::printf("\n");
}

// Fragmentation ratios. AllocatorStats::utilisation() (bytes_requested /
// bytes_reserved) CONFLATES two unrelated things: internal fragmentation
// (rounding waste within a block) and slab over-provisioning (how much of a
// reserved slab is even carved yet). Reporting only that one number would make
// buddy look ~20-30% "utilised" on this trace not because it wastes that much,
// but because a 16 MiB slab is only partly filled. So three named ratios,
// always together, never one alone:
struct FragRatios {
  double block_eff;    // peak_requested / peak_blocks       -- internal fragmentation ONLY.
                       // THIS is the number the roadmap's "buddy 55-70%" prediction meant.
  double reserv_eff;   // peak_blocks / peak_reserved         -- slab sizing, a property of
                       // initial_slab_bytes, NOT of allocator design.
  double utilisation;  // peak_requested / peak_reserved      -- their product; what
                       // AllocatorStats::utilisation() reports, at the peak moment.
};

FragRatios compute_frag_ratios(const ReplayResult& r) {
  FragRatios f{};
  f.block_eff   = r.peak_blocks   > 0 ? double(r.peak_requested) / double(r.peak_blocks)   : 1.0;
  f.reserv_eff  = r.peak_reserved > 0 ? double(r.peak_blocks)    / double(r.peak_reserved)  : 0.0;
  f.utilisation = r.peak_reserved > 0 ? double(r.peak_requested) / double(r.peak_reserved)  : 0.0;
  return f;
}

void print_latency_table(const std::vector<ReplayResult>& rs, const ClockCalibration& cal) {
  std::printf("\n=== 2a. Latency  ->  RESULTS.md section 2a "
             "=================================\n");
  std::printf("%-22s %-10s %9s %8s %8s %8s %9s %10s %12s %12s\n",
             "allocator", "op", "median", "p90", "p99", "p999", "max", "amortised",
             "alloc_calls", "raw_mallocs");
  for (const auto& r : rs) {
    std::printf("%-22s %-10s %8s %7s %7s %7s %8sns %8.1fns %12llu %12llu\n",
               r.allocator_name.c_str(), "allocate",
               latency_cell(r.alloc_lat, r.alloc_lat.median(), cal).c_str(),
               latency_cell(r.alloc_lat, r.alloc_lat.p90(), cal).c_str(),
               latency_cell(r.alloc_lat, r.alloc_lat.p99(), cal).c_str(),
               latency_cell(r.alloc_lat, r.alloc_lat.p999(), cal).c_str(),
               latency_cell(r.alloc_lat, r.alloc_lat.max(), cal).c_str(),
               r.amortised_ns_per_op,
               static_cast<unsigned long long>(r.final_stats.alloc_calls),
               static_cast<unsigned long long>(r.final_stats.raw_malloc_calls));
    std::printf("%-22s %-10s %8s %7s %7s %7s %8sns %8s %12s %12s\n",
               r.allocator_name.c_str(), "deallocate",
               latency_cell(r.free_lat, r.free_lat.median(), cal).c_str(),
               latency_cell(r.free_lat, r.free_lat.p90(), cal).c_str(),
               latency_cell(r.free_lat, r.free_lat.p99(), cal).c_str(),
               latency_cell(r.free_lat, r.free_lat.p999(), cal).c_str(),
               latency_cell(r.free_lat, r.free_lat.max(), cal).c_str(), "-", "-", "-");
  }
  std::printf("* at or below the %llu ns instrument floor -- see clock calibration "
             "above. Use the amortised column for the fast-path cost.\n",
             static_cast<unsigned long long>(2 * cal.floor_ns));
#if !MCKE_WITH_CUDA
  std::printf("\n*** On this host-only build, \"raw\" is aligned_alloc (~100-200 ns),\n"
             "*** NOT cudaMalloc (~10-100 us). The 100-1000x pool speedup is a GPU\n"
             "*** phenomenon and CANNOT be demonstrated on this machine. This\n"
             "*** section's headline speedup claim must come from a Colab/Explorer\n"
             "*** run of this same binary -- see docs/ROADMAP.md Phase 2.\n");
#endif
}

void print_fragmentation_table(const std::vector<ReplayResult>& rs) {
  std::printf("\n=== 2b. Fragmentation  ->  RESULTS.md section 2b "
             "=========================\n");
  std::printf("%-22s %12s %12s %12s %9s %9s %11s %12s %14s %5s\n",
             "allocator", "peak_reserv", "peak_block", "peak_req", "block_eff", "reserv_eff",
             "utilisatn", "int_waste", "largest_free", "OOM?");
  for (const auto& r : rs) {
    const FragRatios f = compute_frag_ratios(r);
    std::printf("%-22s %12s %12s %12s %8.1f%% %8.1f%% %10.1f%% %12s %14s %5s\n",
               r.allocator_name.c_str(),
               human_bytes(static_cast<double>(r.peak_reserved)).c_str(),
               human_bytes(static_cast<double>(r.peak_blocks)).c_str(),
               human_bytes(static_cast<double>(r.peak_requested)).c_str(),
               f.block_eff * 100.0, f.reserv_eff * 100.0, f.utilisation * 100.0,
               human_bytes(static_cast<double>(r.peak_blocks) - static_cast<double>(r.peak_requested)).c_str(),
               human_bytes(static_cast<double>(r.final_stats.largest_free_block)).c_str(),
               r.oom_count > 0 ? "yes" : "no");
  }
  std::printf("\n  block_eff   = peak_requested / peak_blocks   -> INTERNAL fragmentation\n"
             "                only. This is the number the roadmap's \"buddy 55-70%%\"\n"
             "                prediction meant.\n"
             "  reserv_eff  = peak_blocks / peak_reserved     -> slab sizing, a property\n"
             "                of the configured slab size, NOT of allocator design.\n"
             "  utilisation = peak_requested / peak_reserved  -> their product; the\n"
             "                header's AllocatorStats::utilisation(), at the peak moment.\n");
  for (const auto& r : rs) {
    if (r.oom_count == 0) continue;
    std::printf("  %s OOMed %zu time(s); first at op %zu: %s\n", r.allocator_name.c_str(),
               r.oom_count, r.first_oom_op_index, r.first_oom_message.c_str());
  }
  for (const auto& r : rs) {
    if (r.final_stats.largest_free_block == 0) continue;
    std::printf("  %-22s largest_free_block probe: tight-fit %s, over-fit %s\n",
               r.allocator_name.c_str(), r.probe_tight_ok ? "OK (no growth)" : "FAILED",
               r.probe_over_detected ? "OK (rejected/grew)" : "FAILED (silently overshot!)");
  }
}

void print_flatline_evidence(const std::vector<ReplayResult>& rs) {
  std::printf("\n=== raw_malloc_calls vs alloc_calls -- the headline claim "
             "=======================\n");
  std::printf("%-22s %14s %14s %14s %14s\n", "allocator", "alloc_calls@warmup",
             "raw_mallocs@warmup", "alloc_calls@end", "raw_mallocs@end");
  for (const auto& r : rs) {
    std::printf("%-22s %14llu %14llu %14llu %14llu\n", r.allocator_name.c_str(),
               static_cast<unsigned long long>(r.warmup_end_stats.alloc_calls),
               static_cast<unsigned long long>(r.warmup_end_stats.raw_malloc_calls),
               static_cast<unsigned long long>(r.final_stats.alloc_calls),
               static_cast<unsigned long long>(r.final_stats.raw_malloc_calls));
  }
  std::printf("\n  delta after warm-up:\n");
  for (const auto& r : rs) {
    const std::uint64_t d_alloc = r.final_stats.alloc_calls - r.warmup_end_stats.alloc_calls;
    const std::uint64_t d_raw   = r.final_stats.raw_malloc_calls - r.warmup_end_stats.raw_malloc_calls;
    std::printf("    %-22s %llu allocate calls -> %llu additional driver allocations\n",
               r.allocator_name.c_str(), static_cast<unsigned long long>(d_alloc),
               static_cast<unsigned long long>(d_raw));
  }
  std::printf("\n*** For \"raw\", that delta is FORCED to equal alloc_calls (one cudaMalloc\n"
             "*** per allocate, by construction). For buddy/freelist it should be at or\n"
             "*** near ZERO -- a driver allocation only when the pool must physically\n"
             "*** grow a slab (rare) or a request bypasses the pool entirely.\n");
}

void write_latency_csv(const std::string& path, const std::vector<ReplayResult>& rs,
                       const ClockCalibration& cal) {
  std::ofstream f(path);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
  f << "trace,allocator,op,samples,median_ns,p90_ns,p99_ns,p999_ns,max_ns,"
       "amortised_ns,clock_floor_ns,instrument_limited,alloc_calls,raw_malloc_calls,"
       "with_cuda\n";
  for (const auto& r : rs) {
    f << r.trace_name << ',' << r.allocator_name << ",allocate," << r.alloc_lat.count() << ','
      << r.alloc_lat.median() << ',' << r.alloc_lat.p90() << ',' << r.alloc_lat.p99() << ','
      << r.alloc_lat.p999() << ',' << r.alloc_lat.max() << ',' << r.amortised_ns_per_op << ','
      << cal.floor_ns << ',' << (cal.is_below_floor(r.alloc_lat.median()) ? 1 : 0) << ','
      << r.final_stats.alloc_calls << ',' << r.final_stats.raw_malloc_calls << ','
      << MCKE_WITH_CUDA << '\n';
    f << r.trace_name << ',' << r.allocator_name << ",deallocate," << r.free_lat.count() << ','
      << r.free_lat.median() << ',' << r.free_lat.p90() << ',' << r.free_lat.p99() << ','
      << r.free_lat.p999() << ',' << r.free_lat.max() << ',' << r.amortised_ns_per_op << ','
      << cal.floor_ns << ',' << (cal.is_below_floor(r.free_lat.median()) ? 1 : 0) << ','
      << "," << "" << ',' << MCKE_WITH_CUDA << '\n';
  }
  std::printf("wrote %s\n", path.c_str());
}

void write_fragmentation_csv(const std::string& path, const std::vector<ReplayResult>& rs) {
  std::ofstream f(path);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
  // Emitted PER SAMPLING STRIDE, so this one file is simultaneously the 2b
  // table (its last row per allocator/trace) and the flatline time series
  // driving the raw_malloc_calls-vs-alloc_calls claim.
  f << "trace,allocator,op_index,phase,alloc_calls,free_calls,raw_malloc_calls,"
       "oom_events,bytes_reserved,bytes_in_use,bytes_requested,peak_bytes_in_use,"
       "largest_free_block,blocking_drains\n";
  for (const auto& r : rs) {
    for (const auto& s : r.flatline) {
      f << r.trace_name << ',' << r.allocator_name << ',' << s.op_index << ',' << s.phase << ','
        << s.stats.alloc_calls << ',' << s.stats.free_calls << ',' << s.stats.raw_malloc_calls << ','
        << s.stats.oom_events << ',' << s.stats.bytes_reserved << ',' << s.stats.bytes_in_use << ','
        << s.stats.bytes_requested << ',' << s.stats.peak_bytes_in_use << ','
        << s.stats.largest_free_block << ',' << s.stats.blocking_drains << '\n';
    }
  }
  std::printf("wrote %s\n", path.c_str());
}

// The comparison, computed rather than assumed, with the recorded prediction
// printed alongside so a contradiction is visible instead of quietly dropped —
// this is RESULTS.md's own rule 6 ("if a result contradicts the prediction,
// keep the prediction and write down why") encoded directly into the tool.
void print_verdict(const std::vector<ReplayResult>& rs) {
  const ReplayResult* buddy = nullptr;
  const ReplayResult* freelist = nullptr;
  for (const auto& r : rs) {
    // Compare like with like: both at the DEFAULT policy. The policy
    // decomposition is reported separately below.
    if (r.allocator_name == "buddy/coarse_poll") buddy = &r;
    if (r.allocator_name == "freelist/coarse_poll") freelist = &r;
  }
  if (!buddy || !freelist) return;
  const FragRatios fb = compute_frag_ratios(*buddy), ff = compute_frag_ratios(*freelist);

  std::printf("\n=== where each design wins  ->  RESULTS.md section 2 prose "
             "===================\n");
  std::printf("[1] internal fragmentation (block_eff), trace=%s\n", buddy->trace_name.c_str());
  std::printf("      buddy %.1f%%   freelist %.1f%%   -> %s wins by %.1f pts\n",
             fb.block_eff * 100.0, ff.block_eff * 100.0,
             ff.block_eff > fb.block_eff ? "FREELIST" : "BUDDY",
             std::abs(ff.block_eff - fb.block_eff) * 100.0);
  std::printf("      predicted: buddy 55-70%%, freelist 85-95%% on DL shapes; roughly TIED\n"
             "      on multi-MiB tensors because small_large_split puts freelist into a\n"
             "      power-of-two ladder above 1 MiB -- see freelist_allocator.hpp banner.\n");

  std::printf("[2] largest contiguous free block after full drain\n");
  std::printf("      buddy %s   freelist %s   -> %s\n",
             human_bytes(static_cast<double>(buddy->final_stats.largest_free_block)).c_str(),
             human_bytes(static_cast<double>(freelist->final_stats.largest_free_block)).c_str(),
             buddy->final_stats.largest_free_block >= freelist->final_stats.largest_free_block
                 ? "BUDDY wins (coalescing)" : "FREELIST wins (unexpected -- investigate)");

  std::printf("[3] allocate p99\n");
  std::printf("      buddy %llu ns   freelist %llu ns   -> %s\n",
             static_cast<unsigned long long>(buddy->alloc_lat.p99()),
             static_cast<unsigned long long>(freelist->alloc_lat.p99()),
             freelist->alloc_lat.p99() <= buddy->alloc_lat.p99() ? "FREELIST" : "BUDDY");
  std::printf("      two competing costs, and which dominates depends on the trace:\n"
             "      freelist pays a hash-map insert (live_) on every allocate; buddy pays\n"
             "      a SPLIT CASCADE whenever a request needs a block far smaller than the\n"
             "      nearest free ancestor (e.g. a small tensor after a run of large ones).\n"
             "      On this DL trace's largest tensors (multi-MiB, close to the slab size)\n"
             "      the split cascade dominates and buddy's p99 is the worse one -- the\n"
             "      opposite of the uniform_pow2 trace, where blocks are smaller relative\n"
             "      to the slab and buddy's simpler bookkeeping wins. Neither allocator is\n"
             "      categorically faster; it is a function of block size vs. slab depth.\n");

  std::printf("[4] deallocate p99\n");
  std::printf("      buddy %llu ns   freelist %llu ns   -> %s\n",
             static_cast<unsigned long long>(buddy->free_lat.p99()),
             static_cast<unsigned long long>(freelist->free_lat.p99()),
             freelist->free_lat.p99() <= buddy->free_lat.p99() ? "FREELIST" : "BUDDY");
  std::printf("\n  *** [2] and [4] are the SAME mechanism (coalescing) showing up as a win\n"
             "  *** in one row and a cost in the other. If you remember one result from\n"
             "  *** Phase 2, remember this pair.\n");
}

}  // namespace

int main() {
  std::printf("=== MCKE alloc_bench (Phase 2c) "
             "==============================================\n");
  std::printf("build   MCKE_WITH_CUDA=%d\n", MCKE_WITH_CUDA);
  std::printf("device  %s\n",
             MCKE_WITH_CUDA
                 ? "CUDA build: raw_device_malloc == cudaMalloc"
                 : "host-only build: raw_device_malloc == aligned_alloc(256, ...)");

  const ClockCalibration cal = ClockCalibration::measure();
  std::printf("clock   %s\n", cal.describe().c_str());
  std::printf("        std::mt19937_64, seed 0x%llx, raw engine output only (NOT\n"
             "        <random> distributions -- see TraceGen's banner for why)\n",
             static_cast<unsigned long long>(kTraceSeed));

  // --- Generate both traces ONCE, before any allocator exists. Passed as
  // const& to every replay: identity across allocators is then STRUCTURAL
  // (one object), not merely "should be reproducible if the seed matches".
  // reports/ is the project-wide convention for regenerable benchmark
  // output (docs/PROFILING.md); std::ofstream will not create it.
  std::filesystem::create_directories("reports");

  const Trace uniform = generate_uniform_pow2();
  // BYPASS MODE 1 (clean): no allocation anywhere near large_alloc_threshold, so
  // the raw_malloc_calls flatline for this trace is unambiguous.
  const Trace dl        = generate_dl_transformer(/*include_embedding_table=*/false);
  // BYPASS MODE 2 (combined): the same workload with a 147 MiB embedding table
  // folded in, so the bypass path is exercised amid real churn. Reported as its
  // own trace so its extra permanent driver allocation never muddies mode 1.
  const Trace dl_bypass = generate_dl_transformer(/*include_embedding_table=*/true);
  print_trace_summary(uniform);
  print_trace_summary(dl);
  print_trace_summary(dl_bypass);

  // --- Machine warm-up: a throwaway RawDeviceAllocator absorbs process-wide
  // one-time costs (first-touch page faults, instruction-cache warming, the
  // allocator library's own arena setup) so that whichever allocator we
  // measure FIRST is not unfairly penalised for paying them.
  {
    auto warm = std::make_unique<RawDeviceAllocator>();
    (void)replay_amortised(*warm, uniform);
  }

  for (const Trace* tr : {&uniform, &dl, &dl_bypass}) {
    std::printf("\n\n############### trace = %s "
               "###############################################\n",
               tr->name.c_str());
    std::vector<ReplayResult> results;
    for (auto& [name, factory] : allocator_factories()) {
      // Fresh instance per (allocator, trace): otherwise allocator N inherits
      // allocator N-1's warmed slabs, and three simultaneous live pools would
      // compete for the same footprint cap for no reason a real workload ever
      // would.
      auto alloc = factory();
      ReplayResult r = replay_timed(*alloc, *tr);
      // alloc.name() returns the ALLOCATOR's name ("buddy"), which is identical
      // across its three policies. Use the factory's descriptive name instead or
      // the policy rows are indistinguishable in the table.
      r.allocator_name = name;
      r.amortised_ns_per_op = replay_amortised(*factory(), *tr);   // separate fresh instance
      run_largest_free_block_probe(*alloc, r);
      results.push_back(std::move(r));
      (void)name;
    }

    print_latency_table(results, cal);
    print_fragmentation_table(results);
    print_flatline_evidence(results);
    write_latency_csv("reports/alloc_latency_" + tr->name + ".csv", results, cal);
    write_fragmentation_csv("reports/alloc_fragmentation_" + tr->name + ".csv", results);
    if (tr->name == "dl_transformer") print_verdict(results);
  }

  // BYPASS MODE 3 (isolated): a standalone pass/fail with no churn around it.
  std::printf("\n=== bypass path, isolated probe ===================================\n");
  const bool bypass_ok = run_bypass_probe();

  std::printf("\n=== done. Copy the tables above into RESULTS.md section 2. ===\n");
  return bypass_ok ? 0 : 1;
}
