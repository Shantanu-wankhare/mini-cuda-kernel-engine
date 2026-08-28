// =============================================================================
//  tests/test_stream_safety.cu
//
//  WHAT: Phase 2d. Deliberately constructs the cross-stream reuse race on real
//        hardware, proves a naive pool LOSES it (produces wrong numbers), and
//        proves every stream-ordered reuse policy WINS it.
//
//  WHY .cu: contains __global__ kernels. WHY tests/ and not bench/: it produces
//  a pass/fail, not a number, and should run under ctest on every GPU machine.
//
//  ---------------------------------------------------------------------------
//  WHY THIS CANNOT RUN IN THE HOST-ONLY BUILD
//
//  With MCKE_WITH_CUDA=0, every rt::StreamHandle is nullptr -- so "stream 1" and
//  "stream 2" are the SAME handle, and pending_reusable()'s rule-1 short-circuit
//  (reuse_policy.hpp) fires legitimately. rt::stream_query is unconditionally
//  true, so nothing ever parks. And there is no concurrency at all, so there is
//  no race to construct. This is a clean statement of the host backend's LIMITS,
//  worth writing down next to all the places it is a strength.
//
//  ---------------------------------------------------------------------------
//  THE PROPERTY UNDER TEST
//
//  Freeing a block is HOST-SIDE BOOKKEEPING. The GPU kernel that reads it may
//  still be queued or running. If a pool immediately recycles those bytes to a
//  DIFFERENT stream, that stream's kernel overwrites memory the first kernel is
//  still reading -- silently, nondeterministically, wrong numbers with no crash.
//
//  ---------------------------------------------------------------------------
//  DETERMINISM: A HOST-RELEASED GATE, NOT A TIMED SPIN
//
//  The obvious construction is "make the reader spin ~100 ms so the host wins
//  the race". That is only PROBABILISTICALLY correct: it depends on a spin
//  iteration count tuned for one GPU, and it silently degrades on a faster chip,
//  a throttled one, or a debug build. A race test that is merely likely to work
//  is worse than none, because a false CLEAN reads as proof of safety.
//
//  Instead the reader blocks on a flag in MAPPED PINNED memory that only the
//  HOST can set, and the host sets it only AFTER cudaStreamSynchronize(S2)
//  confirms the corrupting write has actually landed. The ordering is therefore
//  structural rather than statistical:
//
//      corrupting write COMPLETED  (proven by the sync)
//        -> host sets the flag
//        -> reader is released and reads
//
//  No calibration, no per-GPU constants, no timing assumptions. The kernel keeps
//  a clock64() bailout purely so a host-side bug cannot wedge the GPU.
//
//  ---------------------------------------------------------------------------
//  THINGS THIS FILE DOES SPECIFICALLY TO AVOID A FALSE PASS
//
//   * The reader's loads carry an ADDRESS DEPENDENCY on the gate value, so nvcc
//     cannot hoist them above the wait (see kGateBias). Without it, -O3 may
//     issue the loads at kernel entry and every arm reports CLEAN.
//   * Loads use __ldcg (bypass L1, hit L2). Turing's L1 is per-SM and NOT
//     coherent; the fill kernel ran on all SMs, so the reader could otherwise
//     read a stale line and report CLEAN after a real corruption.
//   * The verdict is an INTEGER mismatch count, not a float sum. ANY deviation
//     is CORRUPT -- a partial overwrite is the likely real outcome and must not
//     fall between two buckets.
//   * A separate d_gate_seen word witnesses that the reader actually ran and saw
//     the release, so "never launched" is distinguishable from "scanned and found
//     nothing" -- otherwise a silently-failed launch reads as CLEAN.
//   * The naive arm asserts the pool actually ALIASED (same ptr, same size)
//     before its result is interpreted -- a broken control must not masquerade
//     as a proven property.
//   * Assertions are on per-trial pending_count DELTAS, never on
//     stats().deferred_reuses: that counter is incremented UNCONDITIONALLY by
//     kSameStreamOnly and kPerFreeEvent (see deallocate()), so asserting on it
//     would pass with an idle GPU and two identical streams. It counts parks,
//     not deferred cross-stream reuses.
//   * A same-stream CONTROL arm reuses the SAME pointer and must still be CLEAN.
//     Without it, "the cross-stream arm returned a different pointer" is not
//     evidence of stream-ordering -- it is equally consistent with the pool
//     having leaked the block.
// =============================================================================
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_allocator.hpp"
#include "mcke/memory/freelist_allocator.hpp"
#include "mcke/runtime/cuda_check.hpp"
#include "mcke/runtime/stream.hpp"
#include "test_access.hpp"

#ifdef __CUDACC_DEBUG__
#error "Do not build this test with -G. Device debug changes scheduling and timing enough to invalidate the race construction."
#endif

using namespace mcke;

namespace {

// 2 MiB. Deliberately NOT 1 MiB: that is exactly FreeListConfig's
// small_large_split, the single byte count where an off-by-one in the size-class
// ladder would live. Testing stream-ordering there would confound two unrelated
// bugs. 2 MiB sits unambiguously in the power-of-two regime.
constexpr int          kElems      = 512 * 1024;          // 2 MiB of f32
constexpr std::size_t  kBytes      = kElems * sizeof(float);
constexpr float        kPatternA   = 1.0f;                // what the reader expects
constexpr float        kCorrupt    = -7.0f;               // what the racer writes
// "The reader ran" is witnessed by d_gate_seen, NOT by a sentinel value in
// d_bad. A sentinel in d_bad cannot work: the clean case never touches d_bad at
// all (no mismatches, so no atomicAdd fires), so there is no way to distinguish
// "scanned, found nothing" from "never ran" using that word alone.
constexpr int          kTrials     = 20;
constexpr int          kReaderThreads = 256;
// Bailout only. Nothing depends on this duration; it exists so a host bug cannot
// leave a kernel spinning forever and wedge the GPU (or trip a display driver's
// watchdog on a desktop card).
constexpr long long    kBailoutCycles = 4'000'000'000LL;

// Pool geometry: small enough that reserve() is fast (a fresh allocator is built
// per trial), large enough for two 2 MiB blocks plus slack.
constexpr std::size_t  kSlabBytes  = std::size_t{32} << 20;

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__global__ void fill_kernel(float* __restrict__ buf, int n, float v) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) buf[i] = v;
}

// The reader. ONE block, on purpose:
//   * it leaves 39 of a T4's 40 SMs free, so the racing writer on the other
//     stream always finds room -- if this used a full grid, the two kernels
//     could serialise and the race would silently never manifest;
//   * one block grid-striding 512K elements is memory-latency-bound from a
//     single SM, so the read phase spans a wide interval. That width IS the
//     corruption window. Do not "optimise" this into a full grid.
__global__ void gated_read_kernel(const float* __restrict__ buf, int n,
                                  const volatile int* gate,
                                  unsigned* __restrict__ d_bad,
                                  int* __restrict__ d_gate_seen) {
  // Guard the single-block assumption: multiple blocks would race on d_gate_seen
  // and produce garbage that looks like a corruption.
  if (gridDim.x != 1) {
    if (threadIdx.x == 0) *d_bad = 0xDEADBEEFu;
    return;
  }

  __shared__ int s_gate;
  if (threadIdx.x == 0) {
    // Spin until the HOST releases us. `gate` is volatile and lives in mapped
    // pinned memory, so every read crosses PCIe and cannot be cached or hoisted.
    long long t0 = clock64();
    int seen = 0;
    while ((seen = *gate) == 0) {
      if (clock64() - t0 > kBailoutCycles) break;   // host-bug bailout only
    }
    s_gate = seen;
    *d_gate_seen = seen;
  }
  __syncthreads();

  // THE ADDRESS DEPENDENCY. s_gate is 1 when released, so kGateBias is 0 and we
  // read buf[i] as intended -- but nvcc cannot PROVE that at compile time
  // (s_gate came from memory the host writes), so it cannot hoist the loads
  // above the __syncthreads and the wait. Without this, -O3 is free to issue
  // every LDG at kernel entry, the reader would sample the buffer before the
  // racing write lands, and EVERY arm -- including the naive control -- would
  // report CLEAN.
  const int kGateBias = (s_gate > 1000000) ? 1 : 0;
  const float* p = buf + kGateBias;

  unsigned bad = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    // __ldcg: bypass L1, read from L2. Turing's L1 is per-SM and not coherent
    // across SMs, and fill_kernel ran on ALL SMs -- including this one. A normal
    // cached load could therefore return a stale 1.0f that this SM's own fill
    // left resident, reporting CLEAN after a genuine corruption. This is a
    // false-NEGATIVE guard, which is the dangerous direction for a safety test.
    if (__ldcg(p + i) != kPatternA) ++bad;
  }
  // Plain atomicAdd rather than a warp-shuffle reduction. The repo has no
  // __shfl_down_sync anywhere yet -- warp reductions are Phase 3b's teaching
  // content, and a hand-rolled one here would be one more place for a bug in the
  // TEST to masquerade as a bug in the allocator. 256 atomics is free.
  if (bad) atomicAdd(d_bad, bad);
}

// -----------------------------------------------------------------------------
// The naive pool: what "pooling without stream-ordering" actually means, in 25
// lines, sitting next to the assertion that it is broken. That is the lesson.
//
// Exactly ONE block, so handing the same pointer back is GUARANTEED rather than
// probabilistic -- the other half of making this test deterministic.
//
// Rejected alternative: a `stream_ordered = false` flag on BuddyConfig. It would
// be the cleanest single-variable experiment, but BuddyConfig is a public header
// and such a flag would let anyone switch the safety property off in production
// forever. A toy pool in a test translation unit cannot escape.
// -----------------------------------------------------------------------------
class NaivePool final : public DeviceAllocator {
 public:
  Status reserve(std::size_t bytes) {
    auto p = raw_device_malloc(bytes);
    if (!p.ok()) return p.status();
    block_ = *p;
    bytes_ = bytes;
    return OkStatus();
  }
  ~NaivePool() override { if (block_) (void)raw_device_free(block_); }

  StatusOr<Allocation> allocate(std::size_t bytes, rt::StreamHandle) override {
    ++stats_.alloc_calls;
    if (held_ || bytes > bytes_) return OutOfMemoryError("naive_pool: single block busy");
    held_ = true;
    Allocation a;
    a.ptr = block_;
    a.bytes = bytes_;
    a.requested_bytes = bytes;
    return a;
  }
  // THE BUG, on purpose: the stream argument is ignored entirely. No parking, no
  // completion proof, no event. Just "the caller said they were done".
  Status deallocate(const Allocation&, rt::StreamHandle) override {
    ++stats_.free_calls;
    held_ = false;
    return OkStatus();
  }
  [[nodiscard]] AllocatorStats stats() const override { return stats_; }
  [[nodiscard]] std::string_view name() const override { return "naive_pool"; }

 private:
  void*          block_ = nullptr;
  std::size_t    bytes_ = 0;
  bool           held_  = false;
  AllocatorStats stats_{};
};

// -----------------------------------------------------------------------------
// Verdicts. FOUR-valued, not two: a two-valued verdict is exactly how a broken
// control passes as a proven property.
// -----------------------------------------------------------------------------
enum class Verdict { kClean, kCorrupt, kInconclusive, kHarnessBug };

const char* to_str(Verdict v) {
  switch (v) {
    case Verdict::kClean:        return "CLEAN";
    case Verdict::kCorrupt:      return "CORRUPT";
    case Verdict::kInconclusive: return "INCONCLUSIVE";
    case Verdict::kHarnessBug:   return "HARNESS_BUG";
  }
  return "?";
}

struct TrialResult {
  Verdict     verdict = Verdict::kInconclusive;
  unsigned    bad_count = 0;
  bool        aliased = false;        // B2.ptr == B.ptr
  bool        overlapped = false;     // B and B2 regions overlap at all
  std::size_t pending_after_free = 0;
  std::size_t pending_after_alloc = 0;
  bool        refused_reclaim = false;  // the property, for pool arms
  bool        reclaimed_after_drain = false;  // policy discriminator
};

struct ArmSpec {
  std::string name;
  Verdict     expect;
  bool        same_stream_control = false;   // allocate B2 on S1 instead of S2
  bool        is_pool = false;               // has pending_/policy machinery
  std::function<std::unique_ptr<DeviceAllocator>()> make;
  // For pool arms: read pending_count through the right test-access shim.
  std::function<std::size_t(const DeviceAllocator&)> pending;
  ReusePolicy policy = ReusePolicy::kCoarseStreamPoll;
};

// -----------------------------------------------------------------------------
// Host-side mapped flag: the gate the reader blocks on.
// -----------------------------------------------------------------------------
struct HostGate {
  int* host_ptr = nullptr;
  int* dev_ptr  = nullptr;

  void init() {
    MCKE_CUDA_CHECK(cudaHostAlloc(&host_ptr, sizeof(int), cudaHostAllocMapped));
    MCKE_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    *host_ptr = 0;
  }
  void arm()    { *host_ptr = 0; __sync_synchronize(); }
  void release() { __sync_synchronize(); *host_ptr = 1; __sync_synchronize(); }
  void destroy() { if (host_ptr) MCKE_CUDA_CHECK(cudaFreeHost(host_ptr)); host_ptr = nullptr; }
};

bool regions_overlap(const Allocation& a, const Allocation& b) {
  const auto* pa = static_cast<const char*>(a.ptr);
  const auto* pb = static_cast<const char*>(b.ptr);
  return !(pb >= pa + a.bytes || pb + b.bytes <= pa);
}

// -----------------------------------------------------------------------------
// One trial.
// -----------------------------------------------------------------------------
TrialResult run_trial(const ArmSpec& arm, rt::Stream& s1, rt::Stream& s2,
                      HostGate& gate, unsigned* d_bad, int* d_gate_seen) {
  TrialResult r;

  // Fresh allocator per trial: every trial then starts from a provably identical
  // state, so per-trial deltas are comparable across trials. Sharing one
  // allocator would march the pool geometry forward (and, for kSameStreamOnly,
  // grow pending_ monotonically), making trial 20 a different experiment
  // from trial 1.
  auto alloc = arm.make();

  // Warm-up round trip on BOTH streams, BEFORE the measured window. This forces
  // any first-time driver call (cudaMalloc for a slab, cudaEventCreate for an
  // event-pool slot) to happen now rather than inside the race window, where it
  // could flush or synchronise and silently close the window.
  {
    auto w1 = alloc->allocate(kBytes, s1.native());
    if (w1.ok()) (void)alloc->deallocate(*w1, s1.native());
    auto w2 = alloc->allocate(kBytes, s2.native());
    if (w2.ok()) (void)alloc->deallocate(*w2, s2.native());
    MCKE_CUDA_CHECK(cudaStreamSynchronize(s1.native()));
    MCKE_CUDA_CHECK(cudaStreamSynchronize(s2.native()));
  }

  const std::uint64_t raw_before = alloc->stats().raw_malloc_calls;

  gate.arm();
  const unsigned zero_u = 0;
  MCKE_CUDA_CHECK(cudaMemcpyAsync(d_bad, &zero_u, sizeof(unsigned),
                                  cudaMemcpyHostToDevice, s1.native()));
  const int zero = 0;
  MCKE_CUDA_CHECK(cudaMemcpyAsync(d_gate_seen, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice, s1.native()));

  auto b1 = alloc->allocate(kBytes, s1.native());
  if (!b1.ok()) { r.verdict = Verdict::kHarnessBug; return r; }

  const int fill_blocks = 256;
  fill_kernel<<<fill_blocks, 256, 0, s1.native()>>>(
      static_cast<float*>(b1->ptr), kElems, kPatternA);
  MCKE_CUDA_CHECK(cudaGetLastError());

  gated_read_kernel<<<1, kReaderThreads, 0, s1.native()>>>(
      static_cast<const float*>(b1->ptr), kElems, gate.dev_ptr, d_bad, d_gate_seen);
  MCKE_CUDA_CHECK(cudaGetLastError());

  // ---- the measured window opens here -------------------------------------
  const std::size_t pending_before = arm.is_pool ? arm.pending(*alloc) : 0;
  const Status fst = alloc->deallocate(*b1, s1.native());
  if (!fst.ok()) { r.verdict = Verdict::kHarnessBug; return r; }
  r.pending_after_free = arm.is_pool ? arm.pending(*alloc) : 0;

  const rt::StreamHandle reuse_on = arm.same_stream_control ? s1.native() : s2.native();
  auto b2 = alloc->allocate(kBytes, reuse_on);
  if (!b2.ok()) { r.verdict = Verdict::kHarnessBug; return r; }
  r.pending_after_alloc = arm.is_pool ? arm.pending(*alloc) : 0;

  r.aliased    = (b2->ptr == b1->ptr);
  r.overlapped = regions_overlap(*b1, *b2);
  // THE PROPERTY, for cross-stream pool arms: the free parked the block, and the
  // cross-stream allocate walked pending_ and declined to reclaim it.
  r.refused_reclaim = arm.is_pool && !arm.same_stream_control &&
                      r.pending_after_free == pending_before + 1 &&
                      r.pending_after_alloc == r.pending_after_free;

  fill_kernel<<<fill_blocks, 256, 0, reuse_on>>>(
      static_cast<float*>(b2->ptr), kElems, kCorrupt);
  MCKE_CUDA_CHECK(cudaGetLastError());

  // Prove the corrupting write LANDED before releasing the reader. This single
  // sync is what makes the whole test deterministic instead of probabilistic.
  // (For the same-stream control this is a no-op ordering-wise: the write is
  // queued behind the still-gated reader on S1, which is exactly why that arm is
  // safe despite reusing the identical pointer.)
  if (!arm.same_stream_control) {
    MCKE_CUDA_CHECK(cudaStreamSynchronize(s2.native()));
  }
  gate.release();
  MCKE_CUDA_CHECK(cudaStreamSynchronize(s1.native()));
  MCKE_CUDA_CHECK(cudaGetLastError());
  // ---- window closed --------------------------------------------------------

  unsigned bad = 0;
  int gate_seen = 0;
  MCKE_CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
  MCKE_CUDA_CHECK(cudaMemcpy(&gate_seen, d_gate_seen, sizeof(int), cudaMemcpyDeviceToHost));
  r.bad_count = bad;

  // No driver allocation may have happened inside the window, or something in
  // there synchronised and the window was never really open.
  if (alloc->stats().raw_malloc_calls != raw_before) r.verdict = Verdict::kInconclusive;
  else if (gate_seen == 0) r.verdict = Verdict::kHarnessBug;   // reader never ran, or never saw the release
  else if (bad == 0xDEADBEEFu) r.verdict = Verdict::kHarnessBug;                   // grid assumption broken
  else r.verdict = (bad == 0) ? Verdict::kClean : Verdict::kCorrupt;

  // Policy discriminator: with S1 now fully drained, a THIRD allocate on S2 must
  // reclaim under kCoarseStreamPoll / kPerFreeEvent, but must STILL be refused
  // under kSameStreamOnly (which never reclaims cross-stream without an explicit
  // drain). Without this, all three policy arms could secretly be running the
  // same default policy and every other assertion would still pass.
  if (arm.is_pool && !arm.same_stream_control) {
    const std::size_t before3 = arm.pending(*alloc);
    auto b3 = alloc->allocate(kBytes, s2.native());
    if (b3.ok()) {
      r.reclaimed_after_drain = arm.pending(*alloc) < before3;
      (void)alloc->deallocate(*b3, s2.native());
    }
  }

  (void)alloc->deallocate(*b2, reuse_on);
  MCKE_CUDA_CHECK(cudaStreamSynchronize(s1.native()));
  MCKE_CUDA_CHECK(cudaStreamSynchronize(s2.native()));
  return r;
}

// -----------------------------------------------------------------------------
// RawDeviceAllocator gets its own check rather than an arm in the matrix.
//
// Its safety mechanism is different IN KIND: it does not defer, it synchronises.
// cudaFree blocks until the device is idle, so by the time the free returns the
// reader has already finished and there is no window to open at all. Forcing it
// through the gated harness would DEADLOCK -- cudaFree would wait for a reader
// that is itself waiting for a host flag the host cannot set until cudaFree
// returns. So we assert its actual mechanism directly and positively.
// -----------------------------------------------------------------------------
bool check_raw_synchronises(rt::Stream& s1, HostGate& gate, unsigned* d_bad, int* d_gate_seen) {
  RawDeviceAllocator alloc;
  gate.arm();
  const unsigned zero_u = 0;
  MCKE_CUDA_CHECK(cudaMemcpyAsync(d_bad, &zero_u, sizeof(unsigned),
                                  cudaMemcpyHostToDevice, s1.native()));
  const int zero = 0;
  MCKE_CUDA_CHECK(cudaMemcpyAsync(d_gate_seen, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice, s1.native()));

  auto b = alloc.allocate(kBytes, s1.native());
  if (!b.ok()) return false;
  fill_kernel<<<256, 256, 0, s1.native()>>>(static_cast<float*>(b->ptr), kElems, kPatternA);
  MCKE_CUDA_CHECK(cudaGetLastError());
  gated_read_kernel<<<1, kReaderThreads, 0, s1.native()>>>(
      static_cast<const float*>(b->ptr), kElems, gate.dev_ptr, d_bad, d_gate_seen);
  MCKE_CUDA_CHECK(cudaGetLastError());

  // Release BEFORE the free, precisely because cudaFree will block on the reader.
  gate.release();
  const Status st = alloc.deallocate(*b, s1.native());
  if (!st.ok()) return false;
  // The positive assertion: after a cudaFree returns, the stream it was racing
  // is idle. That IS raw's safety mechanism, measured rather than assumed -- and
  // it becomes a canary if a future CUDA release stops synchronising here.
  const bool idle = rt::stream_query(s1.native());
  MCKE_CUDA_CHECK(cudaStreamSynchronize(s1.native()));
  return idle;
}

}  // namespace

int main() {
  std::printf("=== MCKE stream-safety (Phase 2d) ==================================\n");

  if (device_count() == 0) {
    // Exit 77, registered as ctest's SKIP_RETURN_CODE. A CUDA-enabled toolchain
    // does NOT imply a GPU at runtime, and a machine that cannot run this test
    // reporting "passed" is exactly the false pass this file exists to prevent.
    std::printf("no CUDA device -- SKIPPING (a build host is not a run host)\n");
    return 77;
  }

  // Environment serialisers that would silently close the race window.
  if (const char* v = std::getenv("CUDA_LAUNCH_BLOCKING"); v && v[0] == '1') {
    std::printf("FAIL: CUDA_LAUNCH_BLOCKING=1 makes every launch synchronous, so the\n"
                "      race window can never open. Unset it and re-run.\n");
    return 1;
  }
  if (const char* v = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS"); v && std::atoi(v) < 2) {
    std::printf("FAIL: CUDA_DEVICE_MAX_CONNECTIONS=%s funnels both streams onto one\n"
                "      hardware channel and serialises them.\n", v);
    return 1;
  }

  auto devinfo = query_device(0);
  devinfo.status().throw_if_error();
  set_device(0).throw_if_error();
  std::printf("device  %s (sm_%d%d), %d SMs\n", devinfo->name.c_str(), devinfo->cc_major,
              devinfo->cc_minor, devinfo->sm_count);
  std::printf("config  %d elems (%zu KiB), %d trials/arm, gate = host-released "
              "mapped flag\n", kElems, kBytes / 1024, kTrials);

  auto s1r = rt::Stream::create();
  auto s2r = rt::Stream::create();
  s1r.status().throw_if_error();
  s2r.status().throw_if_error();
  rt::Stream s1 = std::move(*s1r), s2 = std::move(*s2r);

  // If these were ever equal, pending_reusable's rule-1 short-circuit would fire
  // and every arm would "correctly" refuse nothing.
  if (s1.native() == s2.native()) {
    std::printf("FAIL: the two streams have identical handles; rule 1 would make every\n"
                "      arm trivially safe and prove nothing.\n");
    return 1;
  }

  HostGate gate;
  gate.init();
  unsigned* d_bad = nullptr;
  int* d_gate_seen = nullptr;
  MCKE_CUDA_CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));
  MCKE_CUDA_CHECK(cudaMalloc(&d_gate_seen, sizeof(int)));

  auto buddy_with = [](ReusePolicy p) {
    return [p]() -> std::unique_ptr<DeviceAllocator> {
      BuddyConfig c;
      c.initial_slab_bytes = kSlabBytes;
      c.max_total_bytes    = kSlabBytes;
      c.min_block_bytes    = 256;
      c.large_alloc_threshold = SIZE_MAX;
      c.reuse_policy = p;
      auto a = std::make_unique<BuddyAllocator>(c);
      a->reserve(kSlabBytes).throw_if_error();
      return a;
    };
  };
  auto freelist_with = [](ReusePolicy p) {
    return [p]() -> std::unique_ptr<DeviceAllocator> {
      FreeListConfig c;
      c.slab_bytes      = kSlabBytes;
      c.max_total_bytes = kSlabBytes;
      c.large_alloc_threshold = SIZE_MAX;
      c.reuse_policy = p;
      auto a = std::make_unique<FreeListAllocator>(c);
      a->reserve(kSlabBytes).throw_if_error();
      return a;
    };
  };
  auto buddy_pending = [](const DeviceAllocator& a) {
    return BuddyTestAccess::pending_count(static_cast<const BuddyAllocator&>(a));
  };
  auto fl_pending = [](const DeviceAllocator& a) {
    return FreeListTestAccess::pending_count(static_cast<const FreeListAllocator&>(a));
  };

  std::vector<ArmSpec> arms;
  arms.push_back({"naive_pool", Verdict::kCorrupt, false, false,
                  []() -> std::unique_ptr<DeviceAllocator> {
                    auto p = std::make_unique<NaivePool>();
                    p->reserve(kBytes).throw_if_error();
                    return p;
                  }, nullptr, ReusePolicy::kCoarseStreamPoll});

  const std::pair<const char*, ReusePolicy> pols[] = {
      {"same_stream_only", ReusePolicy::kSameStreamOnly},
      {"coarse_poll",      ReusePolicy::kCoarseStreamPoll},
      {"per_free_event",   ReusePolicy::kPerFreeEvent}};
  for (const auto& [pn, pv] : pols)
    arms.push_back({std::string("buddy/") + pn, Verdict::kClean, false, true,
                    buddy_with(pv), buddy_pending, pv});
  for (const auto& [pn, pv] : pols)
    arms.push_back({std::string("freelist/") + pn, Verdict::kClean, false, true,
                    freelist_with(pv), fl_pending, pv});

  // The control: identical in every way except that B2 is taken on S1. Rule 1
  // fires, the SAME pointer comes back, and it must STILL be clean -- because
  // stream ordering guarantees the racing write is queued behind the reader.
  // This is what makes "the cross-stream arms returned a different pointer"
  // meaningful; alone, that observation proves nothing.
  arms.push_back({"buddy/same_stream_CONTROL", Verdict::kClean, true, true,
                  buddy_with(ReusePolicy::kCoarseStreamPoll), buddy_pending,
                  ReusePolicy::kCoarseStreamPoll});

  std::printf("\n%-28s %-9s %-16s %-8s %-9s %s\n", "arm", "expect", "observed",
              "aliased", "refused", "result");
  std::printf("%s\n", std::string(90, '-').c_str());

  int failures = 0;
  for (const ArmSpec& arm : arms) {
    int n_clean = 0, n_corrupt = 0, n_incon = 0, n_bug = 0;
    int n_aliased = 0, n_refused = 0, n_reclaimed = 0;
    unsigned worst_bad = 0;

    for (int t = 0; t < kTrials; ++t) {
      const TrialResult r = run_trial(arm, s1, s2, gate, d_bad, d_gate_seen);
      switch (r.verdict) {
        case Verdict::kClean:        ++n_clean; break;
        case Verdict::kCorrupt:      ++n_corrupt; worst_bad = std::max(worst_bad, r.bad_count); break;
        case Verdict::kInconclusive: ++n_incon; break;
        case Verdict::kHarnessBug:   ++n_bug; break;
      }
      if (r.aliased) ++n_aliased;
      if (r.refused_reclaim) ++n_refused;
      if (r.reclaimed_after_drain) ++n_reclaimed;
    }

    bool ok = (n_bug == 0 && n_incon == 0);
    std::string note;

    if (arm.expect == Verdict::kCorrupt) {
      // The control must actually alias, or its result means nothing.
      if (n_aliased != kTrials) { ok = false; note = "did not alias -- control broken"; }
      // Asymmetric on purpose: "the race must manifest" is the statistical
      // claim; "a pool must never corrupt" (below) is absolute.
      if (n_corrupt < kTrials) { ok = false; note = "race did not manifest in every trial"; }
    } else {
      if (n_clean != kTrials) { ok = false; note = "a stream-ordered pool CORRUPTED data"; }
      if (arm.is_pool && !arm.same_stream_control) {
        if (n_refused != kTrials) { ok = false; note = "did not park + refuse cross-stream reclaim"; }
        // Policy discriminator.
        const bool want_reclaim = (arm.policy != ReusePolicy::kSameStreamOnly);
        if (want_reclaim && n_reclaimed != kTrials) { ok = false; note = "never reclaimed even after drain"; }
        if (!want_reclaim && n_reclaimed != 0) { ok = false; note = "kSameStreamOnly reclaimed cross-stream"; }
      }
      if (arm.same_stream_control && n_aliased != kTrials) {
        ok = false; note = "rule 1 did not reuse the same block";
      }
    }
    if (!ok) ++failures;

    char observed[32];
    std::snprintf(observed, sizeof(observed), "%d/%d %s",
                  arm.expect == Verdict::kCorrupt ? n_corrupt : n_clean, kTrials,
                  arm.expect == Verdict::kCorrupt ? "CORRUPT" : "CLEAN");
    std::printf("%-28s %-9s %-16s %-8s %-9s %s%s%s\n", arm.name.c_str(),
                to_str(arm.expect), observed,
                n_aliased == kTrials ? "yes" : (n_aliased == 0 ? "no" : "mixed"),
                arm.is_pool && !arm.same_stream_control ? (n_refused == kTrials ? "yes" : "NO") : "-",
                ok ? "ok" : "FAIL", note.empty() ? "" : " -- ", note.c_str());
    if (n_incon || n_bug)
      std::printf("      (inconclusive=%d harness_bug=%d)\n", n_incon, n_bug);
    if (n_corrupt) std::printf("      worst-case corrupted elements: %u of %d\n", worst_bad, kElems);
  }

  // Raw gets its own positive check -- see the function's comment for why it
  // cannot be an arm in the matrix above.
  const bool raw_ok = check_raw_synchronises(s1, gate, d_bad, d_gate_seen);
  std::printf("%-28s %-9s %-16s %-8s %-9s %s\n", "raw(cudaMalloc)", "SYNCS",
              raw_ok ? "stream idle" : "STILL BUSY", "-", "-", raw_ok ? "ok" : "FAIL");
  if (!raw_ok) ++failures;

  std::printf("\n");
  if (failures == 0) {
    std::printf("=== %zu arms, 0 failures ===\n", arms.size() + 1);
    std::printf("naive_pool corrupted data on every trial; every stream-ordered policy\n"
                "parked the block, refused the cross-stream reclaim, and returned clean\n"
                "data -- and the same-stream control reused the IDENTICAL pointer and was\n"
                "still clean, which is what makes the cross-stream result meaningful.\n");
  } else {
    std::printf("=== %d ARM(S) FAILED ===\n", failures);
  }

  MCKE_CUDA_CHECK(cudaFree(d_bad));
  MCKE_CUDA_CHECK(cudaFree(d_gate_seen));
  gate.destroy();
  return failures == 0 ? 0 : 1;
}
