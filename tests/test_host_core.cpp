// =============================================================================
//  tests/test_host_core.cpp
//
//  WHAT: Unit tests for the host-only half of the runtime — the parts that must
//        be correct before any GPU is involved: buddy-tree index math, shape /
//        stride math, and the baseline allocator's accounting.
//
//  WHY .cpp and not a .py harness: these test C++ invariants (constexpr results,
//  integer overflow behaviour, alignment). A Python harness can only test the
//  process from outside. We WILL add Python later, but for a different job:
//  generating reference outputs with numpy to validate kernel numerics, and
//  plotting benchmark CSVs. Right tool per job — C++ for invariants, Python for
//  reference data and plots.
//
//  WHY no GTest: this file must build on a laptop with no network and no
//  package manager, in one command. 30 lines of macro is a fair price for
//  `clang++ -std=c++20 ... && ./a.out` working everywhere. Google Benchmark
//  arrives in Phase 5 where we genuinely need its statistics.
//
//  RUN (macOS, no CUDA, no CMake needed):
//    clang++ -std=c++20 -I include -DMCKE_WITH_CUDA=0 \
//        tests/test_host_core.cpp src/core/device.cpp src/memory/allocator.cpp \
//        src/core/host_timer.cpp \
//        src/memory/buddy_allocator.cpp src/memory/freelist_allocator.cpp \
//        -o /tmp/mcke_tests && /tmp/mcke_tests
//
//  NOTE: test_buddy_validate_detects_corruption deliberately abandons live
//  allocations after corrupting the tree, so BuddyAllocator's leak warning
//  appears on stderr during that test. That is expected output, not a failure.
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_allocator.hpp"
#include "mcke/memory/buddy_math.hpp"
#include "mcke/memory/freelist_allocator.hpp"
#include "mcke/profiling/host_timer.hpp"
#include "mcke/tensor/shape.hpp"

// -----------------------------------------------------------------------------
// Test access to BuddyAllocator internals.
//
// Declared a friend by the allocator (see buddy_allocator.hpp). Needed because
// the most valuable assertions are about free-list *shape* — the exact structural
// signature left by one split-down, and that a full free coalesces back to only
// the root being free — and no public API can expose that. Exposing internals
// publicly, or #ifdef-ing the tested binary away from the shipped one, would both
// be worse.
// -----------------------------------------------------------------------------
// Allocator-internals access shims, shared with tests/test_stream_safety.cu.
#include "test_access.hpp"
#include "reference.hpp"
#include "mcke/kernels/softmax_online.hpp"

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      ++g_failures;                                                            \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    ++g_checks;                                                                \
    auto _a = (a);                                                             \
    auto _b = (b);                                                             \
    if (!(_a == _b)) {                                                         \
      ++g_failures;                                                            \
      std::printf("  FAIL %s:%d  %s == %s  (%lld vs %lld)\n", __FILE__,        \
                  __LINE__, #a, #b, (long long)_a, (long long)_b);             \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Buddy-tree index math.
//
// The property-based checks matter more than the spot checks: walking every
// node of a small tree and asserting the *invariants* (buddies share a parent,
// offsets tile the arena exactly once per level, no two nodes at a level
// overlap) catches errors that hand-picked examples miss.
// ---------------------------------------------------------------------------
void test_buddy_math() {
  using namespace mcke::buddy;
  std::printf("test_buddy_math\n");

  constexpr unsigned K = 16;   // 64 KiB arena
  constexpr unsigned M = 8;    // 256 B min block
  constexpr unsigned L = max_level(K, M);
  CHECK_EQ(L, 8u);
  CHECK_EQ(node_count(L), (std::size_t{1} << 9) - 1);   // 511 nodes

  // Every level tiles the arena exactly: n_nodes * block_size == arena_size.
  for (unsigned l = 0; l <= L; ++l) {
    CHECK_EQ(nodes_at_level(l) * block_bytes_at_level(K, l), std::size_t{1} << K);
  }

  // Buddy relation is an involution, symmetric, and preserves the parent.
  for (std::size_t g = 1; g < node_count(L); ++g) {
    const std::size_t b = buddy_of(g);
    CHECK(b != g);
    CHECK_EQ(buddy_of(b), g);                     // involution
    CHECK_EQ(parent_of(b), parent_of(g));         // same parent
    CHECK_EQ(level_of(b), level_of(g));           // same level
  }

  // Children of a node cover exactly its byte range, in order.
  for (std::size_t g = 0; g < node_count(L - 1); ++g) {
    const unsigned l = level_of(g);
    if (l >= L) continue;
    const std::size_t lo = left_child_of(g), hi = right_child_of(g);
    CHECK_EQ(offset_of(lo, K), offset_of(g, K));
    CHECK_EQ(offset_of(hi, K), offset_of(g, K) + block_bytes_at_level(K, l + 1));
    CHECK_EQ(block_bytes_at_level(K, l) , 2 * block_bytes_at_level(K, l + 1));
  }

  // Offsets within a level are distinct and aligned to the block size — this is
  // the property the allocator's alignment guarantee rests on.
  for (unsigned l = 0; l <= L; ++l) {
    std::set<std::size_t> seen;
    const std::size_t bs = block_bytes_at_level(K, l);
    for (std::size_t i = 0; i < nodes_at_level(l); ++i) {
      const std::size_t off = offset_of(heap_index(l, i), K);
      CHECK_EQ(off % bs, 0u);
      CHECK(seen.insert(off).second);
      CHECK_EQ(node_at(off, l, K), heap_index(l, i));   // inverse mapping
    }
    CHECK_EQ(seen.size(), nodes_at_level(l));
  }

  // Size -> level, including the rounding behaviour that causes internal
  // fragmentation. These are the numbers we will quote when reporting waste.
  CHECK_EQ(level_for_size(1, K, M), L);          // rounds up to the 256 B min
  CHECK_EQ(level_for_size(256, K, M), L);
  CHECK_EQ(level_for_size(257, K, M), L - 1);    // 257 B costs a 512 B block
  CHECK_EQ(level_for_size(std::size_t{1} << K, K, M), 0u);
  CHECK_EQ(level_for_size((std::size_t{1} << K) + 1, K, M), L + 1);  // sentinel

  // Worst-case internal waste is just under 2x, and it happens at 2^n + 1.
  const std::size_t req = (std::size_t{1} << 12) + 1;              // 4097 B
  const std::size_t got = block_bytes_at_level(K, level_for_size(req, K, M));
  CHECK_EQ(got, std::size_t{1} << 13);                             // 8192 B
  std::printf("  worst-case internal waste at %zu B request: %zu B block "
              "(%.1f%% wasted)\n", req, got,
              100.0 * double(got - req) / double(got));
}

// ---------------------------------------------------------------------------
// Shape / stride math.
// ---------------------------------------------------------------------------
void test_shape() {
  using mcke::DType;
  using mcke::Shape;
  std::printf("test_shape\n");

  Shape s{2, 3, 4};
  CHECK_EQ(s.rank(), 3);
  CHECK_EQ(s.numel(), 24);
  CHECK_EQ(s.bytes(DType::kF32), 96u);
  CHECK(s.is_contiguous());

  // Row-major strides: last dim has stride 1.
  CHECK_EQ(s.stride(2), 1);
  CHECK_EQ(s.stride(1), 4);
  CHECK_EQ(s.stride(0), 12);

  // The rows/cols collapse that lets one row-wise kernel serve every rank.
  CHECK_EQ(s.rows(), 6);      // 2*3
  CHECK_EQ(s.cols(), 4);

  // offset_of must agree with the manual linear index.
  const mcke::dim_t idx[3] = {1, 2, 3};
  CHECK_EQ(s.offset_of(idx), 1 * 12 + 2 * 4 + 3);

  Shape v{1000};
  CHECK_EQ(v.rows(), 1);
  CHECK_EQ(v.cols(), 1000);

  CHECK(Shape({2, 2}) == Shape({2, 2}));
  CHECK(!(Shape({2, 2}) == Shape({4})));
}

// ---------------------------------------------------------------------------
// Baseline allocator accounting. On macOS raw_device_malloc is aligned_alloc,
// so this exercises the *bookkeeping* — which is the part that can be wrong in
// a way the GPU cannot tell us about.
// ---------------------------------------------------------------------------
void test_raw_allocator() {
  std::printf("test_raw_allocator\n");
  mcke::RawDeviceAllocator alloc;
  mcke::rt::StreamHandle s{};

  std::vector<mcke::Allocation> live;
  for (int i = 0; i < 8; ++i) {
    auto a = alloc.allocate(1024 * (i + 1), s);
    CHECK(a.ok());
    if (a.ok()) {
      // The 256 B alignment guarantee our kernels rely on for coalesced,
      // cache-line-aligned loads.
      CHECK_EQ(reinterpret_cast<std::uintptr_t>(a->ptr) % mcke::kDeviceAlignment, 0u);
      live.push_back(*a);
    }
  }
  CHECK_EQ(alloc.stats().alloc_calls, 8u);
  CHECK_EQ(alloc.stats().raw_malloc_calls, 8u);   // baseline: 1 syscall per alloc
  CHECK_EQ(alloc.stats().bytes_in_use, 1024u * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8));
  CHECK_EQ(alloc.stats().peak_bytes_in_use, alloc.stats().bytes_in_use);
  CHECK_EQ(alloc.stats().internal_waste(), 0u);   // raw wastes nothing internally

  for (const auto& a : live) CHECK(alloc.deallocate(a, s).ok());
  CHECK_EQ(alloc.stats().bytes_in_use, 0u);
  CHECK_EQ(alloc.stats().raw_free_calls, 8u);

  // This assertion is the thesis of Phase 2, written down as a test *before* the
  // pool exists: the pool must reach raw_malloc_calls << alloc_calls where the
  // baseline is forced into equality.
  CHECK_EQ(alloc.stats().raw_malloc_calls, alloc.stats().alloc_calls);
}

// =============================================================================
//  BuddyAllocator
//
//  Every test uses a deliberately TINY arena: 64 KiB with a 256 B minimum block,
//  so K=16, M=8, L=8, 511 nodes, 256 leaves. Small enough to enumerate
//  exhaustively, deep enough to have 9 levels — and deliberately the SAME K and M
//  as test_buddy_math above, so both groups reason about one identical tree.
//
//  Property-based over spot checks wherever possible, matching the philosophy of
//  the buddy-math tests. Randomness comes from a fixed-seed LCG written inline, so
//  a failure reproduces byte for byte with no <random> portability questions.
// =============================================================================

using mcke::BuddyTestAccess;
using NodeState = mcke::BuddyTestAccess::NodeState;

constexpr std::size_t kTestSlab = std::size_t{1} << 16;   // 64 KiB
constexpr std::size_t kTestMin  = 256;
constexpr unsigned    kTestK    = 16;   // log2(64 KiB)
constexpr unsigned    kTestL    = 8;    // = kTestK - log2(kTestMin)

// A config that cannot grow, so exhaustion tests actually reach OOM instead of
// quietly reserving more memory.
mcke::BuddyConfig fixed_config() {
  mcke::BuddyConfig c;
  c.initial_slab_bytes    = kTestSlab;
  c.max_total_bytes       = kTestSlab;      // no growth
  c.min_block_bytes       = kTestMin;
  c.large_alloc_threshold = SIZE_MAX;       // nothing bypasses: test the tree
  return c;
}

// Deterministic LCG (Numerical Recipes constants). Inline rather than <random>
// because the standard specifies engine recurrences bit-exactly but NOT
// distributions, so uniform_int_distribution would not reproduce across
// libc++/libstdc++ — and a property test you cannot reproduce is not a test.
struct Lcg {
  std::uint32_t s;
  std::uint32_t next() { s = s * 1664525u + 1013904223u; return s; }
  std::size_t below(std::size_t n) { return next() % n; }
};

void test_buddy_geometry() {
  std::printf("test_buddy_geometry\n");

  // Slab size rounds UP to a power of two, so reserve(N) actually guarantees an
  // N-byte request can be served.
  struct { std::size_t ask; std::size_t want; } cases[] = {
      {1, kTestMin}, {256, 256}, {257, 512}, {4096, 4096}, {4097, 8192},
      {kTestSlab, kTestSlab}, {kTestSlab + 1, kTestSlab * 2},
  };
  for (const auto& c : cases) {
    mcke::BuddyConfig cfg;
    cfg.min_block_bytes = kTestMin;
    cfg.max_total_bytes = 0;
    mcke::BuddyAllocator a(cfg);
    CHECK(a.reserve(c.ask).ok());
    CHECK_EQ(BuddyTestAccess::slab_bytes(a, 0), c.want);
    CHECK(a.validate().ok());
  }

  {  // reserve(0) is an argument error, not an OOM
    mcke::BuddyAllocator a(fixed_config());
    const mcke::Status st = a.reserve(0);
    CHECK(!st.ok());
    CHECK_EQ(static_cast<int>(st.code()), static_cast<int>(mcke::StatusCode::kInvalidArgument));
  }

  // Config geometry must be REJECTED, never silently clamped: a min block below
  // kDeviceAlignment would hand out misaligned pointers, whose only symptom is
  // uncoalesced loads three phases later.
  const std::size_t bad_mins[] = {128, 300};
  for (std::size_t m : bad_mins) {
    mcke::BuddyConfig cfg = fixed_config();
    cfg.min_block_bytes = m;
    mcke::BuddyAllocator a(cfg);
    CHECK(!a.reserve(kTestSlab).ok());
  }
  {
    mcke::BuddyConfig cfg = fixed_config();
    cfg.growth_factor = 1.0;              // could never grow: reject, don't clamp
    mcke::BuddyAllocator a(cfg);
    CHECK(!a.reserve(kTestSlab).ok());
  }
}

void test_buddy_exact_levels() {
  std::printf("test_buddy_exact_levels\n");
  // For every level, a request of exactly that block size must come back with
  // zero waste, at offset 0, and with the block_id the always-descend-left policy
  // predicts. Asserting the exact block_id is what pins that policy.
  for (unsigned l = 0; l <= kTestL; ++l) {
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    const std::size_t want = mcke::buddy::block_bytes_at_level(kTestK, l);
    auto r = a.allocate(want, mcke::rt::StreamHandle{});
    CHECK(r.ok());
    if (!r.ok()) continue;
    CHECK_EQ(r->bytes, want);
    CHECK_EQ(r->requested_bytes, want);
    CHECK_EQ(r->internal_waste(), 0u);
    CHECK_EQ(r->block_id, mcke::buddy::heap_index(l, 0));
    CHECK(r->ptr == BuddyTestAccess::base(a, 0));       // leftmost => offset 0
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(r->ptr) % mcke::kDeviceAlignment, 0u);
    CHECK(a.validate().ok());
    CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
    CHECK(a.validate().ok());
  }
}

void test_buddy_split_signature() {
  std::printf("test_buddy_split_signature\n");
  // One allocation of the SMALLEST block from a fresh slab must split the tree
  // all the way down, leaving exactly one free block at every level except the
  // root. This exact shape is unreachable through the public API, and it is the
  // cheapest possible proof that alloc_node pushes one buddy per level descended
  // rather than churning.
  mcke::BuddyAllocator a(fixed_config());
  CHECK(a.reserve(kTestSlab).ok());
  auto r = a.allocate(kTestMin, mcke::rt::StreamHandle{});
  CHECK(r.ok());
  if (!r.ok()) return;

  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 0u);   // root consumed
  for (unsigned l = 1; l <= kTestL; ++l)
    CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, l), 1u);
  CHECK_EQ(r->block_id, mcke::buddy::heap_index(kTestL, 0));  // leftmost leaf
  CHECK(a.validate().ok());
  CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
  // ...and freeing it must merge all the way back to a single free root.
  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 1u);
  for (unsigned l = 1; l <= kTestL; ++l)
    CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, l), 0u);
  CHECK(a.validate().ok());
}

void test_buddy_exhaust_and_coalesce() {
  std::printf("test_buddy_exhaust_and_coalesce\n");
  // Densest possible state: every leaf allocated. Covers exhaustion, OOM, the
  // raw_malloc_calls claim, and address-space tiling in one test.
  const std::size_t n_leaves = kTestSlab / kTestMin;      // 256
  CHECK_EQ(n_leaves, 256u);

  for (int order = 0; order < 3; ++order) {
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());

    std::vector<mcke::Allocation> live;
    for (std::size_t i = 0; i < n_leaves; ++i) {
      auto r = a.allocate(kTestMin, mcke::rt::StreamHandle{});
      CHECK(r.ok());
      if (r.ok()) live.push_back(*r);
    }
    CHECK_EQ(live.size(), n_leaves);

    // One more must fail, and must fail as OOM rather than anything else.
    auto over = a.allocate(kTestMin, mcke::rt::StreamHandle{});
    CHECK(!over.ok());
    CHECK_EQ(static_cast<int>(over.status().code()),
             static_cast<int>(mcke::StatusCode::kOutOfMemory));

    mcke::AllocatorStats st = a.stats();
    CHECK_EQ(st.alloc_calls, n_leaves + 1);
    CHECK_EQ(st.oom_events, 1u);
    CHECK_EQ(st.bytes_in_use, kTestSlab);
    CHECK_EQ(st.largest_free_block, 0u);
    // THE THESIS: 257 allocate calls, one driver allocation.
    CHECK_EQ(st.raw_malloc_calls, 1u);
    CHECK(a.validate().ok());

    // The 256 blocks must tile [base, base+64K) exactly: no gap, no overlap.
    std::vector<std::uintptr_t> addrs;
    addrs.reserve(live.size());
    for (const auto& al : live) addrs.push_back(reinterpret_cast<std::uintptr_t>(al.ptr));
    std::sort(addrs.begin(), addrs.end());
    CHECK_EQ(addrs[0], reinterpret_cast<std::uintptr_t>(BuddyTestAccess::base(a, 0)));
    bool tiled = true;
    for (std::size_t i = 0; i + 1 < addrs.size(); ++i)
      if (addrs[i + 1] - addrs[i] != kTestMin) tiled = false;
    CHECK(tiled);

    // Free everything in three different orders. The permutation is what finds
    // real coalescing bugs; the ordered runs make a failure readable.
    if (order == 1) std::reverse(live.begin(), live.end());
    if (order == 2) {
      Lcg rng{0xC0FFEEu};
      for (std::size_t i = live.size(); i > 1; --i) std::swap(live[i - 1], live[rng.below(i)]);
    }
    for (const auto& al : live) CHECK(a.deallocate(al, mcke::rt::StreamHandle{}).ok());

    // Maximal coalescing: after a full free, ONLY the root is free.
    CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 1u);
    for (unsigned l = 1; l <= kTestL; ++l)
      CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, l), 0u);
    st = a.stats();
    CHECK_EQ(st.bytes_in_use, 0u);
    CHECK_EQ(st.largest_free_block, kTestSlab);
    CHECK_EQ(st.raw_malloc_calls, 1u);       // still one, after 256 frees
    CHECK(a.validate().ok());
  }
}

void test_buddy_property_no_overlap() {
  std::printf("test_buddy_property_no_overlap\n");
  // The flagship test. Mixed power-of-two and DL-shaped sizes, random churn, with
  // every invariant checked on every iteration against ground truth the test
  // maintains itself (rather than trusting the allocator's own counters).
  mcke::BuddyConfig cfg = fixed_config();
  cfg.max_total_bytes = kTestSlab;
  mcke::BuddyAllocator a(cfg);
  CHECK(a.reserve(kTestSlab).ok());

  const std::size_t sizes[] = {1, 255, 256, 257, 768, 3072, 4097, 8192};
  Lcg rng{0x1234567u};

  struct Live { mcke::Allocation al; unsigned char magic; };
  std::vector<Live> live;
  std::size_t truth_in_use = 0, truth_requested = 0, truth_peak = 0;
  std::size_t oom = 0;
  int overlap_failures = 0, magic_failures = 0, invariant_failures = 0;

  for (int iter = 0; iter < 20000; ++iter) {
    const bool do_alloc = live.empty() || (rng.next() % 100) < 60;
    if (do_alloc) {
      const std::size_t want = sizes[rng.below(sizeof(sizes) / sizeof(sizes[0]))];
      auto r = a.allocate(want, mcke::rt::StreamHandle{});
      if (!r.ok()) { ++oom; continue; }
      if (r->bytes < want) ++invariant_failures;
      if (!mcke::buddy::is_power_of_two(r->bytes)) ++invariant_failures;
      if (r->bytes < kTestMin) ++invariant_failures;
      if (reinterpret_cast<std::uintptr_t>(r->ptr) % mcke::kDeviceAlignment) ++invariant_failures;
      // Actually WRITE the block. Pointer bookkeeping that agrees with itself can
      // still be wrong (correct offset, oversized `bytes`); a magic-byte check
      // catches genuine aliasing that no amount of accounting would.
      //
      // HOST BUILD ONLY, and this is not a cop-out. In a CUDA build `r->ptr` is a
      // cudaMalloc pointer and a host memset on it segfaults immediately. But the
      // thing under test here — where the allocator decides a block lives — is
      // `offset_of()` / the bump pointer / the class ladder, all of which are pure
      // host integer arithmetic that is byte-for-byte IDENTICAL in both builds.
      // The backend only changes what `raw_device_malloc` returns, never how we
      // carve it. So host coverage of this property is complete; a device version
      // (cudaMemset + D2H cudaMemcpy per op) would add ~40k driver calls to a
      // 20k-iteration test to re-prove something backend-independent.
      const unsigned char magic = static_cast<unsigned char>(1 + (r->block_id % 251));
#if !MCKE_WITH_CUDA
      std::memset(r->ptr, magic, r->bytes);
#endif
      live.push_back({*r, magic});
      truth_in_use    += r->bytes;
      truth_requested += r->requested_bytes;
      truth_peak = std::max(truth_peak, truth_in_use);
    } else {
      const std::size_t i = rng.below(live.size());
      const Live v = live[i];
      // Verify nothing else scribbled on us while we were live. Host-only for the
      // same reason as the write above.
#if !MCKE_WITH_CUDA
      const unsigned char* p = static_cast<const unsigned char*>(v.al.ptr);
      for (std::size_t b = 0; b < v.al.bytes; ++b)
        if (p[b] != v.magic) { ++magic_failures; break; }
#endif
      if (!a.deallocate(v.al, mcke::rt::StreamHandle{}).ok()) ++invariant_failures;
      truth_in_use    -= v.al.bytes;
      truth_requested -= v.al.requested_bytes;
      live[i] = live.back();
      live.pop_back();
    }

    // Independently maintained ground truth vs the allocator's own counters. This
    // is the check that catches every stats bug.
    const mcke::AllocatorStats st = a.stats();
    if (st.bytes_in_use != truth_in_use || st.bytes_requested != truth_requested ||
        st.peak_bytes_in_use != truth_peak)
      ++invariant_failures;

    // Live blocks must be pairwise disjoint in the address space.
    if ((iter % 50) == 0) {
      std::map<std::uintptr_t, std::size_t> ivs;
      for (const auto& v : live) ivs[reinterpret_cast<std::uintptr_t>(v.al.ptr)] = v.al.bytes;
      std::uintptr_t prev_end = 0;
      for (const auto& [off, len] : ivs) {
        if (off < prev_end) { ++overlap_failures; break; }
        prev_end = off + len;
      }
    }
    if ((iter % 100) == 0 && !a.validate().ok()) ++invariant_failures;
  }

  CHECK_EQ(overlap_failures, 0);
  CHECK_EQ(magic_failures, 0);
  CHECK_EQ(invariant_failures, 0);
  CHECK(oom > 0);            // a 64 KiB arena under this size mix must hit OOM
  CHECK(a.validate().ok());

  for (const auto& v : live) CHECK(a.deallocate(v.al, mcke::rt::StreamHandle{}).ok());
  CHECK_EQ(a.stats().bytes_in_use, 0u);
  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 1u);   // merged back to root
  CHECK_EQ(a.stats().raw_malloc_calls, 1u);
  CHECK(a.validate().ok());
  std::printf("  20000 ops, %zu OOM (expected: 64 KiB arena), coalesced back to root\n", oom);
}

void test_buddy_internal_waste() {
  std::printf("test_buddy_internal_waste\n");
  // Pin the exact rounding numbers that will appear in RESULTS.md section 2b.
  struct { std::size_t ask; std::size_t block; } cases[] = {
      {257, 512}, {768, 1024}, {3072, 4096}, {4097, 8192},
  };
  mcke::BuddyAllocator a(fixed_config());
  CHECK(a.reserve(kTestSlab).ok());
  std::size_t sum_ask = 0, sum_block = 0;
  std::vector<mcke::Allocation> live;
  for (const auto& c : cases) {
    auto r = a.allocate(c.ask, mcke::rt::StreamHandle{});
    CHECK(r.ok());
    if (!r.ok()) continue;
    CHECK_EQ(r->bytes, c.block);
    CHECK_EQ(r->internal_waste(), c.block - c.ask);
    sum_ask += c.ask;
    sum_block += c.block;
    live.push_back(*r);
  }
  const mcke::AllocatorStats st = a.stats();
  CHECK_EQ(st.bytes_requested, sum_ask);
  CHECK_EQ(st.bytes_in_use, sum_block);
  CHECK_EQ(st.internal_waste(), sum_block - sum_ask);
  std::printf("  DL-shaped sizes: requested %zu B, blocks %zu B, waste %zu B (%.1f%%)\n",
              sum_ask, sum_block, sum_block - sum_ask,
              100.0 * double(sum_block - sum_ask) / double(sum_block));
  for (const auto& al : live) CHECK(a.deallocate(al, mcke::rt::StreamHandle{}).ok());
}

void test_buddy_too_large_and_edges() {
  std::printf("test_buddy_too_large_and_edges\n");
  mcke::BuddyAllocator a(fixed_config());
  CHECK(a.reserve(kTestSlab).ok());

  // Larger than the arena, growth forbidden => clean OOM.
  auto big = a.allocate(kTestSlab + 1, mcke::rt::StreamHandle{});
  CHECK(!big.ok());
  CHECK_EQ(static_cast<int>(big.status().code()),
           static_cast<int>(mcke::StatusCode::kOutOfMemory));

  // SIZE_MAX must not crash. This works only because the level_for_size sentinel
  // is checked BEFORE any align_up — buddy_math.hpp documents that align_up wraps
  // near SIZE_MAX, so the ordering is load-bearing, not incidental.
  auto huge = a.allocate(SIZE_MAX, mcke::rt::StreamHandle{});
  CHECK(!huge.ok());

  // Zero bytes is an argument error and must NOT be counted as an OOM — otherwise
  // this allocator's oom_events would not be comparable with another's.
  const std::uint64_t oom_before = a.stats().oom_events;
  auto zero = a.allocate(0, mcke::rt::StreamHandle{});
  CHECK(!zero.ok());
  CHECK_EQ(static_cast<int>(zero.status().code()),
           static_cast<int>(mcke::StatusCode::kInvalidArgument));
  CHECK_EQ(a.stats().oom_events, oom_before);

  // The allocator must still be fully usable after all of that. An alloc_node
  // that half-split the tree before discovering it could not finish would fail
  // exactly here.
  auto ok = a.allocate(1, mcke::rt::StreamHandle{});
  CHECK(ok.ok());
  CHECK(a.validate().ok());
  if (ok.ok()) {
    CHECK_EQ(ok->bytes, kTestMin);          // 1 byte rounds up to the min block
    CHECK(a.deallocate(*ok, mcke::rt::StreamHandle{}).ok());
  }
  CHECK(a.validate().ok());
}

void test_buddy_growth() {
  std::printf("test_buddy_growth\n");
  mcke::BuddyConfig cfg;
  cfg.initial_slab_bytes    = kTestSlab;          // 64 KiB
  cfg.max_total_bytes       = 3 * kTestSlab;      // 192 KiB
  cfg.min_block_bytes       = kTestMin;
  cfg.growth_factor         = 2.0;
  cfg.large_alloc_threshold = SIZE_MAX;
  mcke::BuddyAllocator a(cfg);
  CHECK(a.reserve(kTestSlab).ok());
  CHECK_EQ(a.stats().raw_malloc_calls, 1u);

  std::vector<mcke::Allocation> live;
  auto grab = [&](std::size_t n) {
    auto r = a.allocate(n, mcke::rt::StreamHandle{});
    if (r.ok()) live.push_back(*r);
    return r.ok();
  };

  CHECK(grab(kTestSlab));                       // fills slab 0 entirely
  CHECK_EQ(a.stats().raw_malloc_calls, 1u);
  CHECK(grab(kTestSlab));                       // must grow: 64 KiB -> 128 KiB
  CHECK_EQ(a.stats().raw_malloc_calls, 2u);
  CHECK_EQ(a.stats().bytes_reserved, 3 * kTestSlab);
  CHECK(grab(kTestSlab));                       // second half of slab 1: no growth
  CHECK_EQ(a.stats().raw_malloc_calls, 2u);
  CHECK(a.validate().ok());

  // Now the cap must bite: every candidate slab size is refused.
  CHECK(!grab(kTestSlab));
  CHECK_EQ(a.stats().raw_malloc_calls, 2u);
  CHECK_EQ(a.stats().oom_events, 1u);

  for (const auto& al : live) CHECK(a.deallocate(al, mcke::rt::StreamHandle{}).ok());
  CHECK(a.validate().ok());
}

void test_buddy_bypass() {
  std::printf("test_buddy_bypass\n");
  mcke::BuddyConfig cfg = fixed_config();
  cfg.large_alloc_threshold = 4096;
  mcke::BuddyAllocator a(cfg);
  CHECK(a.reserve(kTestSlab).ok());
  const std::uint64_t raw0 = a.stats().raw_malloc_calls;

  // Pin the boundary: exactly at the threshold stays in the pool, one byte over
  // bypasses. This is the classic > vs >= off-by-one.
  auto at = a.allocate(4096, mcke::rt::StreamHandle{});
  CHECK(at.ok());
  if (at.ok()) CHECK(at->slab_id != mcke::kBypassSlabId);

  auto over = a.allocate(4097, mcke::rt::StreamHandle{});
  CHECK(over.ok());
  if (over.ok()) {
    CHECK_EQ(over->slab_id, mcke::kBypassSlabId);
    // Bypass reports zero waste to stay directly comparable with RawDeviceAllocator.
    CHECK_EQ(over->bytes, over->requested_bytes);
    CHECK_EQ(a.stats().raw_malloc_calls, raw0 + 1);
    CHECK(a.validate().ok());
    CHECK(a.deallocate(*over, mcke::rt::StreamHandle{}).ok());
    CHECK_EQ(a.stats().raw_free_calls, 1u);
  }
  if (at.ok()) CHECK(a.deallocate(*at, mcke::rt::StreamHandle{}).ok());
  CHECK_EQ(a.stats().bytes_in_use, 0u);
  CHECK_EQ(a.stats().bytes_reserved, kTestSlab);      // bypassed bytes released
  CHECK(a.validate().ok());
}

void test_buddy_trim_preserves_slab_ids() {
  std::printf("test_buddy_trim_preserves_slab_ids\n");
  // This test exists to pin ONE decision: trim() must not erase from slabs_,
  // because erasing shifts indices and invalidates the slab_id inside every
  // Allocation the caller still holds.
  mcke::BuddyConfig cfg;
  cfg.initial_slab_bytes    = kTestSlab;
  cfg.max_total_bytes       = 3 * kTestSlab;
  cfg.min_block_bytes       = kTestMin;
  cfg.large_alloc_threshold = SIZE_MAX;
  mcke::BuddyAllocator a(cfg);
  CHECK(a.reserve(kTestSlab).ok());

  auto a0 = a.allocate(kTestSlab, mcke::rt::StreamHandle{});     // fills slab 0
  CHECK(a0.ok());
  auto a1 = a.allocate(kTestSlab, mcke::rt::StreamHandle{});     // grows slab 1
  CHECK(a1.ok());
  if (!a0.ok() || !a1.ok()) return;
  CHECK_EQ(a1->slab_id, 1u);
  CHECK_EQ(BuddyTestAccess::slab_count(a), 2u);

  CHECK(a.deallocate(*a0, mcke::rt::StreamHandle{}).ok());        // slab 0 now idle
  CHECK(a.trim().ok());
  CHECK(BuddyTestAccess::base(a, 0) == nullptr);                  // dead slot, not erased
  CHECK_EQ(BuddyTestAccess::slab_count(a), 2u);                   // NOT shrunk
  CHECK(BuddyTestAccess::base(a, 1) != nullptr);                  // still live
  CHECK_EQ(a.stats().raw_free_calls, 1u);
  CHECK(a.validate().ok());

  // The still-outstanding allocation must deallocate correctly. If trim() had
  // erased slab 0, a1->slab_id == 1 would now be out of range.
  CHECK(a.deallocate(*a1, mcke::rt::StreamHandle{}).ok());
  CHECK_EQ(a.stats().bytes_in_use, 0u);

  // A dead slot must be reused rather than leaked.
  CHECK(a.trim().ok());
  auto a2 = a.allocate(kTestMin, mcke::rt::StreamHandle{});
  CHECK(a2.ok());
  CHECK_EQ(BuddyTestAccess::slab_count(a), 2u);
  if (a2.ok()) CHECK(a.deallocate(*a2, mcke::rt::StreamHandle{}).ok());
  CHECK(a.validate().ok());
}

void test_buddy_bad_handles() {
  std::printf("test_buddy_bad_handles\n");
  mcke::BuddyAllocator a(fixed_config());
  CHECK(a.reserve(kTestSlab).ok());
  auto r = a.allocate(1024, mcke::rt::StreamHandle{});
  CHECK(r.ok());
  if (!r.ok()) return;

  CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
  // Double free must be a diagnosable Status, not silent tree corruption — a
  // Phase 4 executor bug will hit this path.
  const mcke::Status dbl = a.deallocate(*r, mcke::rt::StreamHandle{});
  CHECK(!dbl.ok());
  CHECK_EQ(static_cast<int>(dbl.code()),
           static_cast<int>(mcke::StatusCode::kFailedPrecondition));

  mcke::Allocation bad = *r;
  bad.slab_id = 99;
  CHECK_EQ(static_cast<int>(a.deallocate(bad, mcke::rt::StreamHandle{}).code()),
           static_cast<int>(mcke::StatusCode::kInvalidArgument));

  bad = *r;
  bad.block_id = 100000;
  CHECK_EQ(static_cast<int>(a.deallocate(bad, mcke::rt::StreamHandle{}).code()),
           static_cast<int>(mcke::StatusCode::kInvalidArgument));

  bad = *r;
  bad.ptr = nullptr;                                    // free(nullptr) convention
  CHECK(a.deallocate(bad, mcke::rt::StreamHandle{}).ok());

  CHECK(a.validate().ok());
}

void test_buddy_validate_detects_corruption() {
  std::printf("test_buddy_validate_detects_corruption\n");
  // A validate() that has never caught anything is decoration. Inject each class
  // of corruption behind the allocator's back and confirm it is reported, with the
  // offending node named in the message.
  auto message_names_node = [](const mcke::Status& st, std::size_t node) {
    return st.message().find(std::to_string(node)) != std::string::npos;
  };

  // NOTE: these cases deliberately abandon live allocations after corrupting the
  // tree (freeing a corrupted node is meaningless), so the allocator's
  // leak warning fires on stderr during this test. That is expected output.

  {  // a node marked kUsed while still sitting in a free list.
     //
     // Corrupt the BUDDY of the allocated block rather than the block itself.
     // Corrupting the block would flip it to kFree next to its already-free
     // buddy, which trips the parent's "both children kFree" check first — a
     // correct catch, but it names the PARENT, so it cannot pin the
     // "message identifies the offending node" property. Choosing a corruption
     // whose own check fires first is what makes that assertion meaningful.
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    auto r = a.allocate(kTestMin, mcke::rt::StreamHandle{});
    CHECK(r.ok());
    if (r.ok()) {
      const std::size_t buddy = mcke::buddy::buddy_of(r->block_id);
      BuddyTestAccess::set_node_state(a, 0, buddy, NodeState::kUsed);
      const mcke::Status st = a.validate();
      CHECK(!st.ok());
      CHECK_EQ(static_cast<int>(st.code()), static_cast<int>(mcke::StatusCode::kInternal));
      CHECK(message_names_node(st, buddy));
    }
  }
  {  // a kFree node missing from its free list, caught from the parent side
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    auto r = a.allocate(kTestMin, mcke::rt::StreamHandle{});
    CHECK(r.ok());
    if (r.ok()) {
      BuddyTestAccess::set_node_state(a, 0, r->block_id, NodeState::kFree);
      const mcke::Status st = a.validate();
      CHECK(!st.ok());
      CHECK_EQ(static_cast<int>(st.code()), static_cast<int>(mcke::StatusCode::kInternal));
      // Caught as a maximal-coalescing violation at the parent — two free buddies
      // that should have merged. Assert the diagnosis, not a specific node id.
      CHECK(st.message().find("coalescing not maximal") != std::string::npos ||
            st.message().find("should have coalesced") != std::string::npos);
    }
  }
  {  // a reachable node marked kDetached
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    BuddyTestAccess::set_node_state(a, 0, 0, NodeState::kDetached);
    const mcke::Status st = a.validate();
    CHECK(!st.ok());
    CHECK(message_names_node(st, 0));
  }
  {  // nonempty_mask disagreeing with the lists
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    BuddyTestAccess::clear_mask_bit(a, 0, 0);
    CHECK(!a.validate().ok());
  }
  {  // coalescing not maximal: two free buddies left unmerged
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    auto x = a.allocate(kTestSlab / 2, mcke::rt::StreamHandle{});
    auto y = a.allocate(kTestSlab / 2, mcke::rt::StreamHandle{});
    CHECK(x.ok() && y.ok());
    if (x.ok() && y.ok()) {
      CHECK(a.validate().ok());
      // Flip both halves to kFree without touching the free lists: exactly the
      // state a coalesce loop that exits one iteration early would leave.
      BuddyTestAccess::set_node_state(a, 0, x->block_id, NodeState::kFree);
      BuddyTestAccess::set_node_state(a, 0, y->block_id, NodeState::kFree);
      CHECK(!a.validate().ok());
    }
  }
}

void test_buddy_stats_thesis() {
  std::printf("test_buddy_stats_thesis\n");
  // The mirror image of the assertion at the end of test_raw_allocator: where the
  // baseline is FORCED into raw_malloc_calls == alloc_calls, the pool must reach
  // raw_malloc_calls << alloc_calls.
  mcke::BuddyAllocator a(fixed_config());
  CHECK(a.reserve(kTestSlab).ok());

  // Warm up, then confirm the driver is never touched again.
  auto warm = a.allocate(1024, mcke::rt::StreamHandle{});
  CHECK(warm.ok());
  if (warm.ok()) CHECK(a.deallocate(*warm, mcke::rt::StreamHandle{}).ok());
  const std::uint64_t raw_after_warmup = a.stats().raw_malloc_calls;

  for (int i = 0; i < 10000; ++i) {
    auto r = a.allocate(1024, mcke::rt::StreamHandle{});
    CHECK(r.ok());
    if (r.ok()) CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
  }
  const mcke::AllocatorStats st = a.stats();
  CHECK_EQ(st.raw_malloc_calls, raw_after_warmup);      // delta of EXACTLY zero
  CHECK_EQ(st.raw_malloc_calls, 1u);
  CHECK(st.alloc_calls > 10000u);
  CHECK_EQ(st.bytes_in_use, 0u);
  CHECK_EQ(st.blocking_drains, 0u);                     // never stalled the host
  CHECK(a.validate().ok());
  std::printf("  %llu allocate calls -> %llu driver allocations (delta after warm-up: 0)\n",
              (unsigned long long)st.alloc_calls, (unsigned long long)st.raw_malloc_calls);
}


// =============================================================================
//  FreeListAllocator
//
//  Test ladder, deliberately tiny so every class can be enumerated:
//      granularity 512 B, split at 4 KiB  ->  8 linear classes
//          class 0..7 = 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096
//      then powers of two ->  class 8..11 = 8 KiB, 16 KiB, 32 KiB, 64 KiB
//  Slab 64 KiB with a hard footprint cap, matching the buddy tests, so the two
//  allocators can be run head-to-head on identical requests.
// =============================================================================

mcke::FreeListConfig fl_config() {
  mcke::FreeListConfig c;
  c.slab_bytes              = kTestSlab;      // 64 KiB
  c.max_total_bytes         = kTestSlab;      // no growth
  c.small_class_granularity = 512;
  c.small_large_split       = 4096;
  c.large_alloc_threshold   = kTestSlab;
  return c;
}

void test_freelist_size_classes() {
  std::printf("test_freelist_size_classes\n");
  mcke::FreeListAllocator a(fl_config());
  CHECK_EQ(mcke::FreeListTestAccess::n_small(a), 8u);
  CHECK_EQ(a.num_classes(), 12u);

  // Exact class block sizes, both regimes and the boundary between them.
  const std::size_t want[] = {512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
                              8192, 16384, 32768, 65536};
  for (std::size_t c = 0; c < 12; ++c) CHECK_EQ(a.class_block_bytes(c), want[c]);

  // THE invariant that matters: a class must never hand out a block SMALLER than
  // the request. An off-by-one here is memory corruption, not mere waste — so
  // check every single byte count in the linear regime rather than sampling.
  int too_small = 0, not_minimal = 0;
  for (std::size_t n = 1; n <= 4096; ++n) {
    const std::size_t cls = a.size_class_of(n);
    const std::size_t bs  = a.class_block_bytes(cls);
    if (bs < n) ++too_small;
    // ...and it must be the SMALLEST such class, or we are wasting more than the
    // ladder's granularity requires.
    if (cls > 0 && a.class_block_bytes(cls - 1) >= n) ++not_minimal;
  }
  CHECK_EQ(too_small, 0);
  CHECK_EQ(not_minimal, 0);

  // Boundary: 4096 is the last linear class, 4097 jumps to the power-of-two ladder.
  CHECK_EQ(a.size_class_of(4096), 7u);
  CHECK_EQ(a.size_class_of(4097), 8u);
  CHECK_EQ(a.class_block_bytes(a.size_class_of(4097)), 8192u);
  CHECK_EQ(a.size_class_of(8192), 8u);
  CHECK_EQ(a.size_class_of(8193), 9u);

  // The predicted quirk: a 256 B request cannot be represented by a 512 B
  // granularity, so this design wastes 50% on its smallest class — exactly where
  // buddy (min block 256 B) is perfect.
  CHECK_EQ(a.class_block_bytes(a.size_class_of(256)), 512u);
  std::printf("  256 B request -> 512 B class block (50.0%% wasted; buddy: 0%%)\n");
}

void test_freelist_basic() {
  std::printf("test_freelist_basic\n");
  mcke::FreeListAllocator a(fl_config());
  CHECK(a.reserve(kTestSlab).ok());
  CHECK_EQ(a.stats().raw_malloc_calls, 1u);

  auto r = a.allocate(3000, mcke::rt::StreamHandle{});
  CHECK(r.ok());
  if (!r.ok()) return;
  CHECK_EQ(r->bytes, 3072u);                 // class 5
  CHECK_EQ(r->requested_bytes, 3000u);
  CHECK_EQ(r->internal_waste(), 72u);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(r->ptr) % mcke::kDeviceAlignment, 0u);
  CHECK(a.validate().ok());

  void* first = r->ptr;
  CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
  CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 5), 1u);
  CHECK_EQ(a.stats().bytes_in_use, 0u);

  // A same-class request must hit the cache, not carve new space.
  auto r2 = a.allocate(2600, mcke::rt::StreamHandle{});   // also class 5
  CHECK(r2.ok());
  if (r2.ok()) {
    CHECK(r2->ptr == first);
    CHECK_EQ(a.stats().raw_malloc_calls, 1u);
    CHECK(a.deallocate(*r2, mcke::rt::StreamHandle{}).ok());
  }

  // The thesis, same as buddy's: allocate calls grow, driver calls do not.
  for (int i = 0; i < 5000; ++i) {
    auto x = a.allocate(1500, mcke::rt::StreamHandle{});
    CHECK(x.ok());
    if (x.ok()) CHECK(a.deallocate(*x, mcke::rt::StreamHandle{}).ok());
  }
  CHECK_EQ(a.stats().raw_malloc_calls, 1u);
  CHECK(a.validate().ok());
}

void test_freelist_no_coalescing() {
  std::printf("test_freelist_no_coalescing\n");
  // The defining behavioural difference, run as a head-to-head on identical
  // requests: fill a 64 KiB arena with 512 B blocks, then free two ADJACENT ones.
  // Buddy merges them; this design cannot.
  const std::size_t n = kTestSlab / 512;    // 128 blocks

  std::size_t fl_largest = 0, bd_largest = 0;
  {
    mcke::FreeListAllocator a(fl_config());
    CHECK(a.reserve(kTestSlab).ok());
    std::vector<mcke::Allocation> live;
    for (std::size_t i = 0; i < n; ++i) {
      auto x = a.allocate(512, mcke::rt::StreamHandle{});
      CHECK(x.ok());
      if (x.ok()) live.push_back(*x);
    }
    CHECK_EQ(a.stats().largest_free_block, 0u);   // arena fully carved
    CHECK(a.deallocate(live[0], mcke::rt::StreamHandle{}).ok());
    CHECK(a.deallocate(live[1], mcke::rt::StreamHandle{}).ok());
    fl_largest = a.stats().largest_free_block;
    CHECK(a.validate().ok());
    for (std::size_t i = 2; i < live.size(); ++i)
      CHECK(a.deallocate(live[i], mcke::rt::StreamHandle{}).ok());
  }
  {
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    std::vector<mcke::Allocation> live;
    for (std::size_t i = 0; i < n; ++i) {
      auto x = a.allocate(512, mcke::rt::StreamHandle{});
      CHECK(x.ok());
      if (x.ok()) live.push_back(*x);
    }
    CHECK_EQ(a.stats().largest_free_block, 0u);
    // The first two 512 B blocks buddy hands out are siblings, so they merge.
    CHECK_EQ(mcke::buddy::buddy_of(live[0].block_id), live[1].block_id);
    CHECK(a.deallocate(live[0], mcke::rt::StreamHandle{}).ok());
    CHECK(a.deallocate(live[1], mcke::rt::StreamHandle{}).ok());
    bd_largest = a.stats().largest_free_block;
    CHECK(a.validate().ok());
    for (std::size_t i = 2; i < live.size(); ++i)
      CHECK(a.deallocate(live[i], mcke::rt::StreamHandle{}).ok());
  }

  CHECK_EQ(fl_largest, 512u);    // two 512 B holes, forever separate
  CHECK_EQ(bd_largest, 1024u);   // merged into one 1 KiB block
  std::printf("  freed 2 adjacent 512 B blocks: freelist largest_free=%zu B, "
              "buddy=%zu B\n", fl_largest, bd_largest);
}

void test_freelist_external_fragmentation() {
  std::printf("test_freelist_external_fragmentation\n");
  // The money test. Identical request sequence to both allocators:
  //   fill the arena with 512 B blocks, free ALL of them, then ask for 8 KiB.
  // Every byte is free in both cases. Only one of them can serve the request.
  const std::size_t n = kTestSlab / 512;

  bool fl_ok = false, bd_ok = false;
  std::size_t fl_free_bytes = 0, fl_largest = 0;
  {
    mcke::FreeListAllocator a(fl_config());
    CHECK(a.reserve(kTestSlab).ok());
    std::vector<mcke::Allocation> live;
    for (std::size_t i = 0; i < n; ++i) {
      auto x = a.allocate(512, mcke::rt::StreamHandle{});
      if (x.ok()) live.push_back(*x);
    }
    for (const auto& al : live) CHECK(a.deallocate(al, mcke::rt::StreamHandle{}).ok());
    CHECK_EQ(a.stats().bytes_in_use, 0u);
    fl_free_bytes = a.stats().bytes_reserved - a.stats().bytes_in_use;
    fl_largest    = a.stats().largest_free_block;

    auto big = a.allocate(8192, mcke::rt::StreamHandle{});
    fl_ok = big.ok();
    if (big.ok()) CHECK(a.deallocate(*big, mcke::rt::StreamHandle{}).ok());
    else CHECK_EQ(static_cast<int>(big.status().code()),
                  static_cast<int>(mcke::StatusCode::kOutOfMemory));
    CHECK(a.validate().ok());
  }
  {
    mcke::BuddyAllocator a(fixed_config());
    CHECK(a.reserve(kTestSlab).ok());
    std::vector<mcke::Allocation> live;
    for (std::size_t i = 0; i < n; ++i) {
      auto x = a.allocate(512, mcke::rt::StreamHandle{});
      if (x.ok()) live.push_back(*x);
    }
    for (const auto& al : live) CHECK(a.deallocate(al, mcke::rt::StreamHandle{}).ok());
    CHECK_EQ(a.stats().largest_free_block, kTestSlab);   // fully coalesced
    auto big = a.allocate(8192, mcke::rt::StreamHandle{});
    bd_ok = big.ok();
    if (big.ok()) CHECK(a.deallocate(*big, mcke::rt::StreamHandle{}).ok());
    CHECK(a.validate().ok());
  }

  // This is the whole Phase 2 tradeoff in two booleans.
  CHECK(!fl_ok);   // holds 64 KiB free, largest contiguous 512 B -> cannot serve 8 KiB
  CHECK(bd_ok);    // coalesced back to one 64 KiB block -> serves it trivially
  std::printf("  after freeing 128x512 B: freelist has %zu B free but largest block "
              "%zu B -> 8 KiB request %s; buddy -> %s\n",
              fl_free_bytes, fl_largest, fl_ok ? "OK" : "OOM", bd_ok ? "OK" : "OOM");
}

void test_freelist_beats_buddy_on_dl_shapes() {
  std::printf("test_freelist_beats_buddy_on_dl_shapes\n");
  // The other direction: where a finer-than-power-of-two ladder wins. These are
  // the literal sizes from the roadmap's DL trace.
  const std::size_t sizes[] = {768 * 4, 3072 * 4, 50257 * 4};   // 3072, 12288, 201028 B

  mcke::FreeListConfig fc;
  fc.slab_bytes              = std::size_t{4} << 20;
  fc.max_total_bytes         = std::size_t{4} << 20;
  fc.small_class_granularity = 512;
  fc.small_large_split       = std::size_t{1} << 20;
  fc.large_alloc_threshold   = std::size_t{4} << 20;

  mcke::BuddyConfig bc;
  bc.initial_slab_bytes    = std::size_t{4} << 20;
  bc.max_total_bytes       = std::size_t{4} << 20;
  bc.min_block_bytes       = 256;
  bc.large_alloc_threshold = SIZE_MAX;

  mcke::FreeListAllocator fl(fc);
  mcke::BuddyAllocator    bd(bc);
  CHECK(fl.reserve(fc.slab_bytes).ok());
  CHECK(bd.reserve(bc.initial_slab_bytes).ok());

  std::size_t requested = 0, fl_blocks = 0, bd_blocks = 0;
  for (std::size_t n : sizes) {
    auto f = fl.allocate(n, mcke::rt::StreamHandle{});
    auto b = bd.allocate(n, mcke::rt::StreamHandle{});
    CHECK(f.ok() && b.ok());
    if (!f.ok() || !b.ok()) continue;
    CHECK(f->bytes >= n);
    CHECK(b->bytes >= n);
    requested += n;
    fl_blocks += f->bytes;
    bd_blocks += b->bytes;
    CHECK(fl.deallocate(*f, mcke::rt::StreamHandle{}).ok());
    CHECK(bd.deallocate(*b, mcke::rt::StreamHandle{}).ok());
  }
  CHECK(fl_blocks < bd_blocks);        // the finer ladder wins here
  CHECK(fl_blocks >= requested);
  // 768 f32 = 3072 B is exactly a multiple of 512, so freelist is EXACT while
  // buddy must round to 4096.
  std::printf("  requested %zu B: freelist %zu B (%.1f%% eff), buddy %zu B (%.1f%% eff)\n",
              requested, fl_blocks, 100.0 * double(requested) / double(fl_blocks),
              bd_blocks, 100.0 * double(requested) / double(bd_blocks));
  CHECK(fl.validate().ok());
  CHECK(bd.validate().ok());
}

void test_freelist_split_large_blocks() {
  std::printf("test_freelist_split_large_blocks\n");
  // With splitting off, a cached 1 KiB block cannot serve a 512 B request and we
  // carve new space. With it on, the 1 KiB block is split and the remainder
  // re-cached -- lower footprint, at the cost of creating a small block that can
  // never merge back.
  std::size_t bump_off = 0, bump_on = 0;
  for (int on = 0; on < 2; ++on) {
    mcke::FreeListConfig c = fl_config();
    c.split_large_blocks = (on == 1);
    mcke::FreeListAllocator a(c);
    CHECK(a.reserve(kTestSlab).ok());

    auto big = a.allocate(1024, mcke::rt::StreamHandle{});   // class 1
    CHECK(big.ok());
    if (!big.ok()) continue;
    CHECK(a.deallocate(*big, mcke::rt::StreamHandle{}).ok());
    CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 1), 1u);

    auto small = a.allocate(512, mcke::rt::StreamHandle{});  // class 0
    CHECK(small.ok());
    if (!small.ok()) continue;
    if (on) {
      // Served by splitting the cached 1 KiB: its class is now empty and a 512 B
      // remainder appeared in class 0.
      CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 1), 0u);
      CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 0), 1u);
    } else {
      // The 1 KiB block just sits there, unusable for this request.
      CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 1), 1u);
      CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 0), 0u);
    }
    CHECK(a.validate().ok());
    const std::size_t bump = a.stats().bytes_in_use;
    if (on) bump_on = bump; else bump_off = bump;
    CHECK(a.deallocate(*small, mcke::rt::StreamHandle{}).ok());
  }
  CHECK_EQ(bump_off, 512u);
  CHECK_EQ(bump_on, 512u);
}

void test_freelist_bad_handles() {
  std::printf("test_freelist_bad_handles\n");
  mcke::FreeListAllocator a(fl_config());
  CHECK(a.reserve(kTestSlab).ok());
  auto r = a.allocate(1024, mcke::rt::StreamHandle{});
  CHECK(r.ok());
  if (!r.ok()) return;
  CHECK(a.deallocate(*r, mcke::rt::StreamHandle{}).ok());
  // The double-free check that live_ is actually paying for.
  const mcke::Status dbl = a.deallocate(*r, mcke::rt::StreamHandle{});
  CHECK(!dbl.ok());
  CHECK_EQ(static_cast<int>(dbl.code()),
           static_cast<int>(mcke::StatusCode::kFailedPrecondition));

  mcke::Allocation bogus = *r;
  bogus.ptr = reinterpret_cast<void*>(std::uintptr_t{0xDEAD000});
  CHECK(!a.deallocate(bogus, mcke::rt::StreamHandle{}).ok());
  CHECK(a.validate().ok());

  // Config rejection: granularity below the device alignment must be refused,
  // not clamped -- misaligned blocks are a silent performance bug later.
  mcke::FreeListConfig bad = fl_config();
  bad.small_class_granularity = 128;
  mcke::FreeListAllocator b(bad);
  CHECK(!b.reserve(kTestSlab).ok());
  CHECK(!b.allocate(512, mcke::rt::StreamHandle{}).ok());
}


// =============================================================================
//  LatencyStats — percentile edge cases.
//
//  This is the whole reason percentile math lives in a header rather than as a
//  static function inside bench/alloc_bench.cpp's main(): a static function in
//  a bench cannot be unit-tested, and off-by-one errors in nearest-rank
//  indexing are exactly the kind of bug that hides at n=1 or n=2 and then
//  silently mis-reports a p99 on a 100,000-sample run.
// =============================================================================
void test_latency_stats_edge_cases() {
  std::printf("test_latency_stats_edge_cases\n");

  { // n=1: every percentile must return the single sample.
    mcke::LatencyStats s(1);
    s.add(777);
    s.finalize();
    CHECK_EQ(s.percentile(0.0), 777u);
    CHECK_EQ(s.percentile(50.0), 777u);
    CHECK_EQ(s.percentile(99.9), 777u);
    CHECK_EQ(s.percentile(100.0), 777u);
    CHECK_EQ(s.min(), 777u);
    CHECK_EQ(s.max(), 777u);
  }
  { // n=2: nearest-rank must not go out of bounds in either direction.
    mcke::LatencyStats s(2);
    s.add(10);
    s.add(20);
    s.finalize();
    CHECK_EQ(s.percentile(0.0), 10u);
    CHECK_EQ(s.percentile(100.0), 20u);
    // p50 of 2 samples, nearest-rank: ceil(0.5*2)-1 = 0 -> the lower one.
    CHECK_EQ(s.median(), 10u);
  }
  { // all-equal: every percentile collapses to the one value, no divide-by-zero
    // anywhere in the mean either.
    mcke::LatencyStats s(5);
    for (int i = 0; i < 5; ++i) s.add(42);
    s.finalize();
    CHECK_EQ(s.median(), 42u);
    CHECK_EQ(s.p99(), 42u);
    CHECK_EQ(s.min(), 42u);
    CHECK_EQ(s.max(), 42u);
    CHECK(s.mean_ns() > 41.9 && s.mean_ns() < 42.1);
  }
  { // n=100: pins the exact nearest-rank definition against values 1..100.
    // p99 -> ceil(0.99*100)-1 = 98 (0-indexed) -> the 99th smallest value = 99.
    mcke::LatencyStats s(100);
    for (int i = 1; i <= 100; ++i) s.add(static_cast<std::uint64_t>(i));
    s.finalize();
    CHECK_EQ(s.percentile(1.0), 1u);
    CHECK_EQ(s.percentile(50.0), 50u);
    CHECK_EQ(s.percentile(99.0), 99u);
    CHECK_EQ(s.percentile(100.0), 100u);
  }
  { // empty: must not crash, and must report 0 rather than garbage.
    mcke::LatencyStats s(0);
    s.finalize();
    CHECK_EQ(s.count(), 0u);
    CHECK_EQ(s.median(), 0u);
    CHECK(s.empty());
  }

  // ClockCalibration must produce a sane, self-consistent measurement: nonzero
  // floor, and paired_median never below tick (the floor is a max of the two).
  const mcke::ClockCalibration cal = mcke::ClockCalibration::measure(20000);
  CHECK(cal.floor_ns > 0);
  CHECK(cal.floor_ns >= cal.tick_ns);
  CHECK(cal.floor_ns >= cal.paired_median_ns);
  CHECK(cal.is_below_floor(cal.floor_ns));           // at the floor: not trustworthy
  CHECK(!cal.is_below_floor(cal.floor_ns * 1000));   // far above it: trustworthy
  std::printf("  clock: %s\n", cal.describe().c_str());
}


// =============================================================================
//  The Phase-3 validation harness itself.
//
//  These test the TESTER. Every kernel correctness check in Phase 3 routes
//  through compare(), so a bug here would silently weaken all of them -- a
//  tolerance that is accidentally infinite passes everything, and one that
//  mishandles zero fails correct code until someone "fixes" it by loosening it.
//  Both failure modes end with a test nobody trusts.
// =============================================================================
void test_reference_compare() {
  std::printf("test_reference_compare\n");
  using namespace mcke::testing;

  {  // identical inputs match at any tolerance, including zero
    const float a[] = {1.0f, -2.5f, 0.0f, 1e8f, -1e-8f};
    const CompareResult r = compare(a, a, 5, 0.0, 0.0);
    CHECK(r.ok());
    CHECK_EQ(r.mismatches, 0u);
  }
  {  // THE near-zero case a pure relative test gets wrong.
     // want == 0 exactly (what ReLU produces for half its inputs). A relative
     // test divides by zero here; the mixed form falls back to abs_tol.
    const float want[] = {0.0f, 0.0f};
    const float got[]  = {1e-9f, 1e-3f};
    const CompareResult r = compare(got, want, 2, /*rel_tol=*/1e-5, /*abs_tol=*/1e-8);
    CHECK_EQ(r.mismatches, 1u);          // 1e-9 passes on abs_tol, 1e-3 does not
    CHECK_EQ(r.worst_index, 1u);
  }
  {  // ...and the large-magnitude case a pure ABSOLUTE test gets wrong: an error
     // of 1.0 next to 1e8 is 1e-8 relative and entirely acceptable.
    const float want[] = {1e8f};
    const float got[]  = {1e8f + 1.0f};
    CHECK(compare(got, want, 1, /*rel_tol=*/1e-5).ok());
  }
  {  // the tolerance boundary is respected in both directions
    const float want[] = {1.0f, 1.0f};
    const float just_under[] = {1.0f + 9e-6f, 1.0f};
    const float just_over[]  = {1.0f + 2e-5f, 1.0f};
    CHECK(compare(just_under, want, 2, 1e-5).ok());
    CHECK(!compare(just_over, want, 2, 1e-5).ok());
  }
  {  // NaN is reported SEPARATELY, not folded into the mismatch count. A
     // NaN compares false against everything including itself, so without the
     // explicit flag an all-NaN output and a slightly-off output are
     // indistinguishable -- and they are completely different bugs. NaN in a
     // softmax almost always means the max-subtraction was skipped.
    const float want[] = {1.0f, 2.0f};
    const float got[]  = {std::nanf(""), 2.0f};
    const CompareResult r = compare(got, want, 2, 1e-5);
    CHECK(!r.ok());
    CHECK(r.any_nan);
    CHECK_EQ(r.mismatches, 1u);
  }
  {  // an Inf where a finite value was expected must not silently pass
    const float want[] = {1.0f};
    const float got[]  = {std::numeric_limits<float>::infinity()};
    const CompareResult r = compare(got, want, 1, 1e-5);
    CHECK(!r.ok());
    CHECK(r.any_inf);
  }
  {  // fill_random must be deterministic and in range -- the whole
     // cross-machine reproducibility argument rests on this
    std::vector<float> a(64), b(64);
    fill_random(a.data(), 64, 12345);
    fill_random(b.data(), 64, 12345);
    int differing = 0, out_of_range = 0;
    for (int i = 0; i < 64; ++i) {
      if (a[i] != b[i]) ++differing;
      if (a[i] < -1.0f || a[i] > 1.0f) ++out_of_range;
    }
    CHECK_EQ(differing, 0);
    CHECK_EQ(out_of_range, 0);
    std::vector<float> c(64);
    fill_random(c.data(), 64, 999);      // a different seed must differ
    int same = 0;
    for (int i = 0; i < 64; ++i) if (a[i] == c[i]) ++same;
    CHECK(same < 64);
  }
}

void test_reference_kernels() {
  std::printf("test_reference_kernels\n");
  using namespace mcke::testing;
  namespace K = mcke::kernels;

  {  // bias+act against values computed by hand
    const float x[]    = {1.0f, -1.0f, 0.5f, -0.5f};
    const float bias[] = {0.0f, 0.0f};
    float y[4];
    reference_bias_act(x, bias, y, 2, 2, K::Activation::kRelu);
    CHECK(y[0] == 1.0f && y[1] == 0.0f && y[2] == 0.5f && y[3] == 0.0f);

    reference_bias_act(x, bias, y, 2, 2, K::Activation::kNone);
    CHECK(y[1] == -1.0f);

    // The two GELU forms must AGREE to ~1e-3 but not to 1e-7 -- that gap is the
    // entire reason "which GELU" is a real question. If they matched exactly,
    // one of them is not implementing what it claims.
    float ye[4], yt[4];
    reference_bias_act(x, bias, ye, 2, 2, K::Activation::kGeluErf);
    reference_bias_act(x, bias, yt, 2, 2, K::Activation::kGeluTanh);
    const CompareResult loose = compare(yt, ye, 4, 1e-2);
    const CompareResult tight = compare(yt, ye, 4, 1e-7);
    CHECK(loose.ok());
    CHECK(!tight.ok());
  }
  {  // row reduce: sum / mean / max, including an ALL-NEGATIVE row, which is
     // where a max-identity of 0 instead of -inf silently produces 0
     const float x[] = {1.0f, 2.0f, 3.0f, 4.0f,
                        -5.0f, -1.0f, -9.0f, -3.0f};
    float out[2];
    reference_row_reduce(x, out, 2, 4, K::ReduceKind::kSum);
    CHECK(out[0] == 10.0f && out[1] == -18.0f);
    reference_row_reduce(x, out, 2, 4, K::ReduceKind::kMean);
    CHECK(out[0] == 2.5f && out[1] == -4.5f);
    reference_row_reduce(x, out, 2, 4, K::ReduceKind::kMax);
    CHECK(out[0] == 4.0f);
    CHECK(out[1] == -1.0f);            // NOT 0 -- the all-negative trap
  }
  {  // softmax: rows sum to 1, order is preserved, and a huge logit does NOT
     // produce NaN (which it would without the max subtraction: expf(1000)=inf)
    // Ordering, on a MODERATE row where every entry stays representable.
    const float xm[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4];
    reference_row_softmax(xm, y, 1, 4);
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) sum += y[i];
    CHECK(std::fabs(sum - 1.0) < 1e-6);
    CHECK(y[3] > y[2] && y[2] > y[1] && y[1] > y[0]);

    // Overflow safety, on an EXTREME row. Note what is and is not asserted here:
    // the max subtraction guarantees no NaN/Inf, but it does NOT prevent the
    // small entries underflowing to EXACTLY 0.0 -- exp(1 - 1000) = e^-999 is
    // ~1e-434, far below the smallest representable double. So the three losing
    // entries are all exactly zero and are NOT strictly ordered. That is correct
    // softmax behaviour at this dynamic range, not a defect; asserting strict
    // ordering here (as an earlier version of this test did) fails on correct
    // code. Without the max subtraction this row would be expf(1000) = inf and
    // the whole result NaN, which is what the isnan checks pin.
    const float xe[] = {1.0f, 2.0f, 3.0f, 1000.0f};
    reference_row_softmax(xe, y, 1, 4);
    CHECK(!std::isnan(y[0]) && !std::isnan(y[3]));
    CHECK(std::fabs(y[3] - 1.0f) < 1e-6);   // the 1000 dominates completely
    CHECK(y[0] == 0.0f && y[1] == 0.0f);    // underflowed, as they must
    double esum = 0.0;
    for (int i = 0; i < 4; ++i) esum += y[i];
    CHECK(std::fabs(esum - 1.0) < 1e-6);    // still sums to 1

    // A uniform row must give exactly 1/n.
    const float u[] = {5.0f, 5.0f, 5.0f, 5.0f};
    reference_row_softmax(u, y, 1, 4);
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(y[i] - 0.25f) < 1e-6);
  }
  {  // GEMM against a hand-computed 2x2, then alpha/beta which are easy to drop
    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};      // [[1,2],[3,4]]
    const float b[] = {5.0f, 6.0f, 7.0f, 8.0f};      // [[5,6],[7,8]]
    float c[] = {0.0f, 0.0f, 0.0f, 0.0f};
    reference_gemm(a, b, c, 2, 2, 2, 1.0f, 0.0f);
    CHECK(c[0] == 19.0f && c[1] == 22.0f && c[2] == 43.0f && c[3] == 50.0f);

    // beta must accumulate onto the EXISTING C, not overwrite it.
    float c2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    reference_gemm(a, b, c2, 2, 2, 2, 2.0f, 3.0f);
    CHECK(c2[0] == 2.0f * 19.0f + 3.0f);
    CHECK(c2[3] == 2.0f * 50.0f + 3.0f);

    // A non-square, non-tile-multiple shape: this is where indexing bugs live.
    const float a3[] = {1, 2, 3, 4, 5, 6};            // 2x3
    const float b3[] = {1, 2, 3, 4, 5, 6, 7, 8};      // 3x... (use 3x2)
    float c3[4] = {0, 0, 0, 0};
    reference_gemm(a3, b3, c3, 2, 2, 3, 1.0f, 0.0f);
    CHECK(c3[0] == 1*1 + 2*3 + 3*5);                  // 22
    CHECK(c3[1] == 1*2 + 2*4 + 3*6);                  // 28
  }
}


// =============================================================================
//  The online-softmax rescaling recurrence.
//
//  This is the trickiest arithmetic in Phase 3 and it is testable here, on a
//  machine with no GPU, because the recurrence is MCKE_HOST_DEVICE. Verifying it
//  exhaustively on the Mac -- against the independent three-pass reference, on
//  random AND adversarial rows -- means that when softmax.cu later disagrees with
//  the reference on a GPU, the recurrence is already ruled out and the bug is in
//  the kernel's parallel decomposition. That is worth far more than the twenty
//  lines it costs.
// =============================================================================
void test_online_softmax_recurrence() {
  std::printf("test_online_softmax_recurrence\n");
  using mcke::kernels::OnlineState;
  using mcke::kernels::online_identity;
  using mcke::kernels::online_update;
  using mcke::kernels::online_combine;

  // Sequentially folding a whole row must reproduce the three-pass (max, sum).
  auto fold_all = [](const float* row, int n) {
    OnlineState s = online_identity();
    for (int i = 0; i < n; ++i) s = online_update(s, row[i]);
    return s;
  };
  // The three-pass answer, in double, as the thing to be judged against.
  auto three_pass = [](const float* row, int n, double& m_out, double& d_out) {
    double m = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) m = std::max(m, (double)row[i]);
    double d = 0.0;
    for (int i = 0; i < n; ++i) d += std::exp((double)row[i] - m);
    m_out = m; d_out = d;
  };

  {  // THE IDENTITY. Two empty partials must combine to an empty partial, NOT
     // to NaN. With -INFINITY as the identity this computes 0*exp(NaN) and the
     // whole block reduction is poisoned. This is the single highest-value
     // assertion in the file, and it is only reachable when cols < blockDim.
    const OnlineState e = online_identity();
    const OnlineState ee = online_combine(e, e);
    CHECK(!std::isnan(ee.m));
    CHECK(!std::isnan(ee.d));
    CHECK_EQ(ee.d, 0.0f);
    // An identity combined with real data must leave the data untouched.
    OnlineState s = online_update(online_identity(), 3.0f);
    const OnlineState se = online_combine(s, e);
    CHECK(std::fabs(se.m - s.m) < 1e-6f);
    CHECK(std::fabs(se.d - s.d) < 1e-6f);
  }

  int mismatches = 0, nans = 0, assoc_failures = 0;
  // Random rows of several lengths, including lengths below a warp.
  const int lens[] = {1, 2, 17, 31, 32, 33, 255, 256, 1000};
  for (int li = 0; li < 9; ++li) {
    const int n = lens[li];
    std::vector<float> row(static_cast<std::size_t>(n));
    mcke::testing::fill_random(row.data(), row.size(), 0xABCDEF00ull + li, -8.0f, 8.0f);

    const OnlineState got = fold_all(row.data(), n);
    double m_ref = 0.0, d_ref = 0.0;
    three_pass(row.data(), n, m_ref, d_ref);
    if (std::isnan(got.m) || std::isnan(got.d)) ++nans;
    if (std::fabs((double)got.m - m_ref) > 1e-6) ++mismatches;
    // Online accumulates rescaling error, so the sum gets a looser bound than
    // the max -- the max is a pure selection and must be exact.
    if (std::fabs((double)got.d - d_ref) / d_ref > 1e-5) ++mismatches;

    // ASSOCIATIVITY is the property the whole parallel decomposition rests on:
    // if folding [0,k) and [k,n) separately and merging does not equal folding
    // straight through, then no tree reduction of this operator is valid.
    for (int k = 1; k < n; k += (n / 4 + 1)) {
      const OnlineState a = fold_all(row.data(), k);
      const OnlineState b = fold_all(row.data() + k, n - k);
      const OnlineState merged = online_combine(a, b);
      if (std::fabs((double)merged.m - (double)got.m) > 1e-6) ++assoc_failures;
      if (std::fabs((double)merged.d - (double)got.d) / (double)got.d > 1e-5)
        ++assoc_failures;
      // ...and COMMUTATIVITY, since a warp shuffle merges in an arbitrary order.
      const OnlineState swapped = online_combine(b, a);
      if (std::fabs((double)swapped.d - (double)merged.d) / (double)merged.d > 1e-6)
        ++assoc_failures;
    }
  }
  CHECK_EQ(nans, 0);
  CHECK_EQ(mismatches, 0);
  CHECK_EQ(assoc_failures, 0);

  {  // ADVERSARIAL: a monotonically increasing row spanning 0..90 is the WORST
     // case for this algorithm -- the running max updates on every single
     // element, so every element pays a rescale, and the accumulated error is at
     // its maximum. It also proves the overflow guard: expf(90) is +inf, so a
     // formulation without the max subtraction would produce inf/inf = NaN here.
    const int n = 512;
    std::vector<float> row(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) row[i] = 90.0f * (float)i / (float)(n - 1);
    const OnlineState got = fold_all(row.data(), n);
    double m_ref = 0.0, d_ref = 0.0;
    three_pass(row.data(), n, m_ref, d_ref);
    CHECK(!std::isnan(got.m) && !std::isnan(got.d));
    CHECK(!std::isinf(got.d));
    CHECK(std::fabs((double)got.m - m_ref) < 1e-6);
    const double rel = std::fabs((double)got.d - d_ref) / d_ref;
    CHECK(rel < 1e-4);       // looser: this is the maximum-rescaling case
    std::printf("  worst case (monotone 0..90, rescale on every element): "
                "rel err %.2e\n", rel);

    // Reversed: the max arrives FIRST, so no rescale ever happens after it.
    //
    // MEASURED RESULT, which CONTRADICTED the prediction and is worth keeping:
    // the reversed row is *slightly LESS* accurate (2.50e-07 vs 1.73e-07), not
    // more. The prediction assumed "rescaling is the error source, so no
    // rescaling means less error." That reasoning is incomplete.
    //
    // What actually happens: rescaling does not only introduce error, it also
    // RENORMALISES. In the forward (monotone increasing) case every new element
    // IS the new max, so each step computes d = d*exp(m_old - m_new) + 1 -- the
    // accumulator is damped by a factor < 1 and then has 1.0 added, keeping d at
    // O(1) throughout and always summing quantities of comparable magnitude,
    // which is the numerically favourable regime.
    //
    // In the reversed case the max arrives first, so d starts at 1.0 and every
    // subsequent term is exp(very negative) -- a long tail of tiny values added
    // to a large accumulator. That is the classic ill-conditioned summation
    // pattern, and it costs more than the rescaling saved.
    //
    // So: rescaling error and summation conditioning pull in OPPOSITE directions
    // here. Both stay at ~2e-7, far inside any tolerance, so neither ordering
    // matters practically -- but asserting the predicted ordering would be
    // asserting something false. Assert what is actually guaranteed: both are
    // small.
    std::vector<float> rev(row.rbegin(), row.rend());
    const OnlineState got_rev = fold_all(rev.data(), n);
    const double rel_rev = std::fabs((double)got_rev.d - d_ref) / d_ref;
    std::printf("  same row reversed (max first, zero rescaling):        "
                "rel err %.2e  <- NOT better; see comment\n", rel_rev);
    CHECK(rel_rev < 1e-4);
    CHECK(!std::isnan(got_rev.d) && !std::isinf(got_rev.d));
  }
}

// =============================================================================
//  Stream-ordered reuse
//
//  These tests only exist because of two observations that together make the
//  safety-critical deferral path reachable with no GPU:
//
//   1. In a host-only build rt::stream_query and rt::event_query are
//      unconditionally true, so every free would be immediate and the parking
//      branch would be dead code. The protected `stream_completed` /
//      `event_completed` seams let a test subclass say "still running".
//
//   2. rt::StreamHandle is an opaque void* that the allocator only ever COMPARES
//      and passes along — it never dereferences it. So a test can fabricate
//      distinct fake handles and get N distinguishable streams for free.
//
//  Together, both rule 1 (same-stream) and rule 2 (cross-stream) run through real
//  production code paths on a laptop. Guarded off under CUDA, where handing the
//  driver a fake non-null stream handle would be a genuine crash.
// =============================================================================
#if !MCKE_WITH_CUDA

mcke::rt::StreamHandle fake_stream(std::uintptr_t id) {
  return reinterpret_cast<mcke::rt::StreamHandle>(id);
}

class TestBuddy : public mcke::BuddyAllocator {
 public:
  using mcke::BuddyAllocator::BuddyAllocator;
  std::set<mcke::rt::StreamHandle> busy_streams;   // "still has work in flight"
  std::set<std::uint32_t>          busy_events;    // per-slot, see event_completed

 protected:
  bool stream_completed(mcke::rt::StreamHandle h) const override {
    return busy_streams.find(h) == busy_streams.end();
  }
  bool event_completed(std::uint32_t slot) const override {
    return busy_events.find(slot) == busy_events.end();
  }
};

mcke::BuddyConfig policy_config(mcke::ReusePolicy p) {
  mcke::BuddyConfig c = fixed_config();
  c.reuse_policy = p;
  return c;
}

void test_buddy_rule1_same_stream() {
  std::printf("test_buddy_rule1_same_stream\n");
  // Rule 1 must hold under EVERY policy, and must cost no completion proof: a
  // block freed on stream S is immediately reusable by S because the stream is
  // in-order. Every stream is marked busy, so any policy that reached for a
  // completion probe would fail to reuse and we would see a different pointer.
  const mcke::ReusePolicy policies[] = {mcke::ReusePolicy::kSameStreamOnly,
                                        mcke::ReusePolicy::kCoarseStreamPoll,
                                        mcke::ReusePolicy::kPerFreeEvent};
  for (mcke::ReusePolicy pol : policies) {
    TestBuddy a(policy_config(pol));
    CHECK(a.reserve(kTestSlab).ok());
    const auto s1 = fake_stream(1);
    a.busy_streams.insert(s1);              // S1 is NOT idle
    a.busy_events.insert(0);                // nor is any event on it

    auto r1 = a.allocate(4096, s1);
    CHECK(r1.ok());
    if (!r1.ok()) continue;
    void* first = r1->ptr;
    CHECK(a.deallocate(*r1, s1).ok());

    auto r2 = a.allocate(4096, s1);         // same stream => reuse without proof
    CHECK(r2.ok());
    if (r2.ok()) {
      CHECK(r2->ptr == first);
      CHECK_EQ(a.stats().raw_malloc_calls, 1u);   // no growth was needed
      CHECK(a.deallocate(*r2, s1).ok());
    }
    CHECK(a.validate().ok());
  }
}

void test_buddy_cross_stream_defers() {
  std::printf("test_buddy_cross_stream_defers\n");
  // The safety property, driven deterministically with no GPU.
  TestBuddy a(policy_config(mcke::ReusePolicy::kCoarseStreamPoll));
  CHECK(a.reserve(kTestSlab).ok());
  const auto s1 = fake_stream(1), s2 = fake_stream(2);
  a.busy_streams.insert(s1);                       // S1 has work in flight

  auto r1 = a.allocate(4096, s1);
  CHECK(r1.ok());
  if (!r1.ok()) return;
  void* first = r1->ptr;

  CHECK(a.deallocate(*r1, s1).ok());
  CHECK_EQ(a.stats().deferred_reuses, 1u);         // parked, not freed
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);
  // The whole point: the block is still kUsed in the tree, so it cannot be
  // handed out or coalesced while S1 might still be reading it.
  CHECK(BuddyTestAccess::node_state(a, 0, r1->block_id) == NodeState::kUsed);
  CHECK(a.validate().ok());

  auto r2 = a.allocate(4096, s2);                  // different stream => must NOT reuse
  CHECK(r2.ok());
  if (r2.ok()) CHECK(r2->ptr != first);
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u); // still parked

  // Now S1 drains. The next allocate must reclaim it.
  a.busy_streams.erase(s1);
  auto r3 = a.allocate(4096, s2);
  CHECK(r3.ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 0u);
  CHECK(a.validate().ok());

  if (r2.ok()) CHECK(a.deallocate(*r2, s2).ok());
  if (r3.ok()) CHECK(a.deallocate(*r3, s2).ok());
}

void test_buddy_deferred_does_not_coalesce() {
  std::printf("test_buddy_deferred_does_not_coalesce\n");
  // The subtle invariant. Two buddies: free one normally, park the other. They
  // must NOT merge, because a merged block spans memory that a live kernel may
  // still be reading — and the merged block is exactly what a later allocate
  // would hand to some other stream.
  TestBuddy a(policy_config(mcke::ReusePolicy::kCoarseStreamPoll));
  CHECK(a.reserve(kTestSlab).ok());
  const auto s1 = fake_stream(1), s2 = fake_stream(2), s3 = fake_stream(3);

  auto x = a.allocate(kTestSlab / 2, s1);          // node 1, level 1
  auto y = a.allocate(kTestSlab / 2, s2);          // node 2, its buddy
  CHECK(x.ok() && y.ok());
  if (!x.ok() || !y.ok()) return;
  CHECK_EQ(mcke::buddy::buddy_of(x->block_id), y->block_id);

  CHECK(a.deallocate(*x, s1).ok());                // S1 idle => freed immediately
  CHECK_EQ(BuddyTestAccess::pending_count(a), 0u);

  a.busy_streams.insert(s2);
  CHECK(a.deallocate(*y, s2).ok());                // S2 busy => parked
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);

  // Both halves are logically free from the caller's point of view, but the tree
  // must still show only ONE free half and no merged root.
  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 1), 1u);
  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 0u);
  CHECK_EQ(a.stats().largest_free_block, kTestSlab / 2);
  CHECK(a.validate().ok());

  // Release S2. The next allocate reclaims the parked block, and only THEN may
  // the two halves merge back into the whole arena.
  a.busy_streams.erase(s2);
  auto probe = a.allocate(kTestMin, s3);
  CHECK(probe.ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 0u);
  if (probe.ok()) CHECK(a.deallocate(*probe, s3).ok());
  CHECK_EQ(a.stats().largest_free_block, kTestSlab);   // fully merged now
  CHECK_EQ(BuddyTestAccess::free_list_size(a, 0, 0), 1u);
  CHECK(a.validate().ok());
}

void test_buddy_policy_same_stream_only() {
  std::printf("test_buddy_policy_same_stream_only\n");
  // kSameStreamOnly never probes for completion, so capacity is stream-affine:
  // a block freed on S1 stays parked until S1 itself allocates again, EVEN IF S1
  // is provably idle. That is the policy's defining weakness, asserted rather
  // than described.
  TestBuddy a(policy_config(mcke::ReusePolicy::kSameStreamOnly));
  CHECK(a.reserve(kTestSlab).ok());
  const auto s1 = fake_stream(1), s2 = fake_stream(2);
  // Note: busy_streams is EMPTY — every stream is idle. A probing policy would
  // reclaim instantly. This one still will not.

  auto r1 = a.allocate(4096, s1);
  CHECK(r1.ok());
  if (!r1.ok()) return;
  CHECK(a.deallocate(*r1, s1).ok());
  CHECK_EQ(a.stats().deferred_reuses, 1u);        // parks even on an idle stream
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);

  auto r2 = a.allocate(4096, s2);                 // S2 cannot claim S1's block
  CHECK(r2.ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);
  if (r2.ok()) CHECK(a.deallocate(*r2, s2).ok());  // now two parked, one per stream
  CHECK_EQ(BuddyTestAccess::pending_count(a), 2u);

  // An allocate on S1 must reclaim ONLY S1's block, via rule 1.
  auto r3 = a.allocate(4096, s1);
  CHECK(r3.ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);  // S2's block still parked
  CHECK(a.validate().ok());

  // trim() drains unconditionally — the documented escape hatch for exactly the
  // capacity leak this policy creates.
  if (r3.ok()) CHECK(a.deallocate(*r3, s1).ok());
  CHECK(a.trim().ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 0u);
  CHECK(a.stats().blocking_drains > 0u);           // and it counted the stall
  CHECK(a.validate().ok());
}

void test_buddy_policy_per_free_event() {
  std::printf("test_buddy_policy_per_free_event\n");
  // Per-block precision: two blocks freed on the SAME stream get separate events,
  // so one can be reclaimed while the other is still in flight. Coarse polling
  // cannot express that — it would hold both or release both.
  TestBuddy a(policy_config(mcke::ReusePolicy::kPerFreeEvent));
  CHECK(a.reserve(kTestSlab).ok());
  const auto s1 = fake_stream(1), s2 = fake_stream(2);

  auto x = a.allocate(4096, s1);
  auto y = a.allocate(4096, s1);
  CHECK(x.ok() && y.ok());
  if (!x.ok() || !y.ok()) return;

  CHECK(a.deallocate(*x, s1).ok());        // -> event slot 0
  CHECK(a.deallocate(*y, s1).ok());        // -> event slot 1
  CHECK_EQ(a.stats().deferred_reuses, 2u); // records unconditionally, by design
  CHECK_EQ(BuddyTestAccess::pending_count(a), 2u);

  a.busy_events.insert(0);                 // x's work is still running; y's is done
  auto probe = a.allocate(kTestMin, s2);   // cross-stream: forces the event check
  CHECK(probe.ok());
  // Exactly one released. This is the granularity a stream-level probe cannot
  // reach, and the reason the seam takes a slot rather than a handle.
  CHECK_EQ(BuddyTestAccess::pending_count(a), 1u);
  CHECK(BuddyTestAccess::node_state(a, 0, x->block_id) == NodeState::kUsed);
  CHECK(a.validate().ok());

  a.busy_events.erase(0);
  auto probe2 = a.allocate(kTestMin, s2);
  CHECK(probe2.ok());
  CHECK_EQ(BuddyTestAccess::pending_count(a), 0u);

  if (probe.ok())  CHECK(a.deallocate(*probe, s2).ok());
  if (probe2.ok()) CHECK(a.deallocate(*probe2, s2).ok());
  CHECK(a.trim().ok());
  CHECK(a.validate().ok());
}


class TestFreeList : public mcke::FreeListAllocator {
 public:
  using mcke::FreeListAllocator::FreeListAllocator;
  std::set<mcke::rt::StreamHandle> busy_streams;
  std::set<std::uint32_t>          busy_events;
 protected:
  bool stream_completed(mcke::rt::StreamHandle h) const override {
    return busy_streams.find(h) == busy_streams.end();
  }
  bool event_completed(std::uint32_t slot) const override {
    return busy_events.find(slot) == busy_events.end();
  }
};

void test_freelist_reuse_policies() {
  std::printf("test_freelist_reuse_policies\n");
  // Same three policies, same shared decision function as buddy -- which is the
  // point: any divergence here would contaminate the Phase 2c comparison.
  const mcke::ReusePolicy pols[] = {mcke::ReusePolicy::kSameStreamOnly,
                                    mcke::ReusePolicy::kCoarseStreamPoll,
                                    mcke::ReusePolicy::kPerFreeEvent};
  for (mcke::ReusePolicy pol : pols) {
    mcke::FreeListConfig c = fl_config();
    c.reuse_policy = pol;
    TestFreeList a(c);
    CHECK(a.reserve(kTestSlab).ok());
    const auto s1 = fake_stream(1);
    a.busy_streams.insert(s1);
    a.busy_events.insert(0);

    auto r1 = a.allocate(1024, s1);
    CHECK(r1.ok());
    if (!r1.ok()) continue;
    void* first = r1->ptr;
    CHECK(a.deallocate(*r1, s1).ok());
    // Rule 1: same stream reuses with no completion proof, under every policy.
    auto r2 = a.allocate(1024, s1);
    CHECK(r2.ok());
    if (r2.ok()) {
      CHECK(r2->ptr == first);
      CHECK(a.deallocate(*r2, s1).ok());
    }
    CHECK(a.validate().ok());
  }

  // Cross-stream deferral, and the invariant that a parked block is in NEITHER
  // live_ nor a class list -- the analogue of buddy keeping a parked node kUsed.
  {
    mcke::FreeListConfig c = fl_config();
    c.reuse_policy = mcke::ReusePolicy::kCoarseStreamPoll;
    TestFreeList a(c);
    CHECK(a.reserve(kTestSlab).ok());
    const auto s1 = fake_stream(1), s2 = fake_stream(2);
    a.busy_streams.insert(s1);

    auto r = a.allocate(1024, s1);
    CHECK(r.ok());
    if (!r.ok()) return;
    void* first = r->ptr;
    CHECK(a.deallocate(*r, s1).ok());
    CHECK_EQ(mcke::FreeListTestAccess::pending_count(a), 1u);
    CHECK_EQ(mcke::FreeListTestAccess::cached_count(a, 1), 0u);  // NOT reusable yet
    CHECK_EQ(mcke::FreeListTestAccess::live_count(a), 0u);
    CHECK(a.validate().ok());

    auto other = a.allocate(1024, s2);      // must not get the parked block
    CHECK(other.ok());
    if (other.ok()) CHECK(other->ptr != first);

    a.busy_streams.erase(s1);
    auto after = a.allocate(1024, s2);      // now the reclaim releases it
    CHECK(after.ok());
    CHECK_EQ(mcke::FreeListTestAccess::pending_count(a), 0u);
    CHECK(a.validate().ok());
    if (other.ok()) CHECK(a.deallocate(*other, s2).ok());
    if (after.ok()) CHECK(a.deallocate(*after, s2).ok());
  }
}

#endif  // !MCKE_WITH_CUDA

// ---------------------------------------------------------------------------
// Phase 3d: the GEMM tile descriptor and the four-limiter occupancy calculator.
//
// These run on a machine with no GPU, which is the entire reason the logic was
// hoisted into mcke/kernels/gemm_tile.hpp rather than left in kernels/gemm.cu.
// docs/ROADMAP.md makes the hand-computed-vs-measured occupancy comparison the
// learning objective of Phase 3d; a hand calculation that lives in a comment
// cannot be wrong in any way a test notices.
//
// The assertions below are written to catch REAL errors rather than to restate
// the implementation. Concretely: smem figures are hard-coded byte counts, not
// `== (bm*bk + bk*bn)*4` (which would just re-run the code under test), and
// every rejection is asserted individually rather than as "returns false for
// garbage".
// ---------------------------------------------------------------------------
mcke::DeviceInfo t4_device_info() {
  // Hand-constructed rather than queried. query_device() errors out in a
  // host-only build (src/core/device.cpp), so tests MUST fabricate one -- which
  // is exactly the portability DeviceInfo was made a POD to enable.
  mcke::DeviceInfo d;
  d.name = "Tesla T4 (synthetic)";
  d.cc_major = 7; d.cc_minor = 5;
  d.sm_count = 40;
  d.max_threads_per_sm = 1024;      // Turing: 1024, NOT Volta's 2048
  d.max_threads_per_block = 1024;
  d.warps_per_sm = 32;
  d.regs_per_sm = 65536;
  d.regs_per_block = 65536;
  d.max_blocks_per_sm = 16;         // Turing: 16, NOT Volta/Ampere's 32
  d.shared_mem_per_block = 48 * 1024;
  d.shared_mem_per_block_optin = 64 * 1024;
  d.shared_mem_per_sm = 64 * 1024;
  return d;
}

void test_gemm_tile_geometry() {
  std::printf("test_gemm_tile_geometry\n");
  using namespace mcke::kernels;

  // Hard-coded expectations. Deriving them from the same formula the code uses
  // would make these tests pass for any consistent-but-wrong implementation.
  CHECK_EQ(threads_per_block(GemmTile{}), 256);              // (128/8)*(128/8)
  CHECK_EQ(threads_per_block(GemmTile{32, 32, 32, 1, 1}), 1024);
  CHECK_EQ(smem_bytes(GemmTile{}, false), 8192u);            // (128*8 + 8*128)*4
  CHECK_EQ(smem_bytes(GemmTile{}, true), 16384u);            // double-buffered
  CHECK_EQ(smem_bytes(GemmTile{32, 32, 32, 1, 1}, false), 8192u);

  // The DEFAULT tile must map to a compiled instantiation. Every caller that
  // omits a tile gets GemmTile{}, so if the default and the dispatch table ever
  // drift, the whole bench silently runs an unsupported configuration.
  CHECK(select_tile_config(GemmTile{}) == GemmTileConfig::k128x128x8_8x8);
  CHECK(select_tile_config(GemmTile{32, 32, 32, 1, 1}) == GemmTileConfig::k32x32x32_1x1);

  // The one that catches silent-fallback-to-nearest-tile: a tile that is
  // perfectly self-consistent and simply was never instantiated. There is no
  // compile error for this case, so only an explicit test finds it.
  const GemmTile plausible{64, 64, 16, 4, 4};
  CHECK(tile_is_self_consistent(plausible));
  CHECK(select_tile_config(plausible) == GemmTileConfig::kUnsupported);

  // Self-consistency: each rejection reason individually.
  CHECK(tile_is_self_consistent(GemmTile{}));
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 8, 7, 8}));   // bm % tm != 0
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 8, 8, 7}));   // bn % tn != 0
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 0, 8, 8}));   // bk == 0
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 8, 1, 1}));   // 16384 threads
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 8, 8, 128})); // 16 threads, not
                                                                 // warp-aligned...
  // ...and warp alignment specifically: 8x8 output tile with 4x4 threads = 4
  // threads/block. A partial warp per block wastes issue slots silently and
  // skews every occupancy figure computed from it.
  CHECK(!tile_is_self_consistent(GemmTile{8, 8, 8, 4, 4}));
  // The staging loop must tile evenly across threads, or the tail of the shared
  // tile is never written and the inner loop multiplies garbage.
  CHECK(!tile_is_self_consistent(GemmTile{128, 128, 3, 8, 8}));   // 128*3 % 256 != 0

  for (const GemmTile& t : {GemmTile{}, GemmTile{32, 32, 32, 1, 1}}) {
    CHECK(threads_per_block(t) % 32 == 0);
    CHECK(threads_per_block(t) <= 1024);
  }
}

void test_gemm_occupancy_limiters() {
  std::printf("test_gemm_occupancy_limiters\n");
  using namespace mcke::kernels;
  const mcke::DeviceInfo d = t4_device_info();

  // --- Each of the four limiters must bind in at least one case. This is the
  //     only way to prove the "which limiter" answer is computed rather than
  //     guessed, and it is why the function returns a limiter at all.

  // 1. REGISTERS. 256 threads at 128 regs/thread: 65536 / (128*32 rounded to a
  //    256-register warp granule = 4096) = 16 warps -> 2 blocks of 8 warps.
  //    Shared memory would allow 8, the threads cap 4, the blocks cap 16.
  {
    const auto e = occupancy_blocks_per_sm(d, 256, 128, 8192);
    CHECK_EQ(e.blocks_per_sm, 2);
    CHECK(e.limiter == OccupancyLimiter::kRegisters);
    CHECK(!e.tied);
    CHECK(e.occupancy > 0.499 && e.occupancy < 0.501);        // 2*256/1024 = 50%
    CHECK_EQ(e.by_smem, 8);
    CHECK_EQ(e.by_threads, 4);
  }

  // 2. SHARED MEMORY, uniquely binding: 128-thread blocks using 32 KiB each.
  {
    const auto e = occupancy_blocks_per_sm(d, 128, 32, 32 * 1024);
    CHECK_EQ(e.blocks_per_sm, 2);                              // 64 KiB / 32 KiB
    CHECK(e.limiter == OccupancyLimiter::kSharedMemory);
    CHECK(!e.tied);
  }

  // 3. THREADS-PER-SM cap, uniquely binding.
  {
    const auto e = occupancy_blocks_per_sm(d, 256, 32, 1024);
    CHECK_EQ(e.blocks_per_sm, 4);                              // 1024 / 256
    CHECK(e.limiter == OccupancyLimiter::kThreadsPerSm);
  }

  // 4. BLOCKS-PER-SM cap. This is the case that fails loudly if 32 (the Volta /
  //    Ampere figure) was hardcoded instead of read from DeviceInfo -- and every
  //    authoritative number in this project comes from a Turing T4, where it
  //    is 16.
  {
    const auto e = occupancy_blocks_per_sm(d, 32, 16, 0);
    CHECK_EQ(e.blocks_per_sm, 16);
    CHECK(e.limiter == OccupancyLimiter::kBlocksPerSm);
  }

  // --- Pins the claim documented on GemmTile in gemm_tile.hpp: the register-
  //     blocked tile is REGISTER-bound at 2 blocks / 50%, and shared memory --
  //     the obvious-looking constraint -- is not binding. If someone edits the
  //     formula, this doc-claim test fails.
  {
    const auto e = occupancy_blocks_per_sm(d, 256, 128, 16384);   // double-buffered
    CHECK_EQ(e.blocks_per_sm, 2);
    CHECK(e.limiter == OccupancyLimiter::kRegisters);
    CHECK_EQ(e.by_smem, 4);            // smem allows 4; registers allow 2
  }

  // --- tiled_smem: 1024 threads/block. Registers and the threads cap BOTH give
  //     1 block, so this is a genuine tie -- and the tie must be reported,
  //     because "registers" alone would invite cutting register pressure, which
  //     cannot possibly help: two 1024-thread blocks would need 2048 threads on
  //     a 1024-thread SM. Ties resolve toward the architectural cap for exactly
  //     this reason.
  {
    const auto e = occupancy_blocks_per_sm(d, 1024, 64, 8192);
    CHECK_EQ(e.blocks_per_sm, 1);
    CHECK(e.tied);
    CHECK(e.limiter == OccupancyLimiter::kThreadsPerSm);
    CHECK(e.occupancy > 0.999);        // 1 block x 1024 threads = 100% occupancy
  }

  // --- THE REGISTER CLIFF, and it is a launch failure rather than a slowdown.
  //     At 1024 threads, 32 warps x ceil(R*32/256) x 256 <= 65536 forces R <= 64.
  //     R = 65 does not run slowly; it returns cudaErrorLaunchOutOfResources.
  //     __launch_bounds__(1024, 1) on gemm_tiled_smem_kernel exists to make
  //     ptxas responsible for staying under it.
  {
    CHECK_EQ(occupancy_blocks_per_sm(d, 1024, 64, 8192).blocks_per_sm, 1);
    const auto over = occupancy_blocks_per_sm(d, 1024, 65, 8192);
    CHECK_EQ(over.blocks_per_sm, 0);
    CHECK(over.limiter == OccupancyLimiter::kInvalid);
  }
  // Same cliff one tile up: the register-blocked variants need R <= 128 for two
  // blocks. Double buffering adds ~8 registers of staged operands, so this
  // boundary is the reason __launch_bounds__(256, 2) is not optional there.
  CHECK_EQ(occupancy_blocks_per_sm(d, 256, 128, 8192).blocks_per_sm, 2);
  CHECK_EQ(occupancy_blocks_per_sm(d, 256, 129, 8192).blocks_per_sm, 1);

  // --- Register allocation granularity actually applied. A naive
  //     regs_per_sm / (R * threads) would give 7 here; the per-warp 256-register
  //     granule plus the 4-warp rounding gives 5. If this ever reads 7, the
  //     granularity modelling was dropped and the whole three-way comparison in
  //     the bench becomes uninformative.
  {
    const auto e = occupancy_blocks_per_sm(d, 512, 17, 0);
    CHECK_EQ(e.by_registers, 5);
    CHECK(65536 / (17 * 512) == 7);    // what the naive formula would have said
  }

  // --- Degenerate inputs return "will not launch" rather than crashing or
  //     inventing a 1. A default-constructed DeviceInfo is all zeros and is
  //     reachable in a host-only build, so this is a normal path.
  {
    const mcke::DeviceInfo empty;
    CHECK_EQ(occupancy_blocks_per_sm(empty, 256, 32, 1024).blocks_per_sm, 0);
    CHECK_EQ(occupancy_blocks_per_sm(d, 0, 32, 1024).blocks_per_sm, 0);
    CHECK_EQ(occupancy_blocks_per_sm(d, 2048, 32, 1024).blocks_per_sm, 0);   // > 1024
    CHECK_EQ(occupancy_blocks_per_sm(d, 256, 256, 1024).blocks_per_sm, 0);   // > 255 regs
    CHECK_EQ(occupancy_blocks_per_sm(d, 256, 32, 128 * 1024).blocks_per_sm, 0);
  }

  // --- Invariants that must hold everywhere. Monotonicity is the useful one:
  //     asking for more registers can never buy more resident blocks, and a
  //     violation would mean a rounding bug in the granularity arithmetic.
  {
    int prev = 1 << 30;
    for (int r = 1; r <= 255; ++r) {
      const auto e = occupancy_blocks_per_sm(d, 256, r, 8192);
      CHECK(e.blocks_per_sm <= prev);
      prev = e.blocks_per_sm;
      CHECK(e.blocks_per_sm <= d.max_blocks_per_sm);
      if (e.blocks_per_sm > 0) {
        CHECK(e.occupancy > 0.0 && e.occupancy <= 1.0);
        CHECK_EQ(e.warps_per_sm, e.blocks_per_sm * 8);
      }
    }
  }
}

void test_gemm_dbuf_schedule() {
  std::printf("test_gemm_dbuf_schedule\n");
  using namespace mcke::kernels;

  // Replays the double-buffered k-loop from kernels/gemm.cu against a model of
  // shared memory, for every tile count from 0 to 200, and asserts the two
  // properties the kernel's correctness rests on:
  //
  //   (a) EVERY k-tile is computed EXACTLY ONCE. A dropped tile means the last
  //       BK columns of K silently contribute nothing -- a ~0.2% error at
  //       K=4096 that no tolerance would flag as a bug and that the K=256
  //       validation shape (32 tiles, even) would never expose at all.
  //   (b) Each compute reads the buffer that was last PUBLISHED with that tile.
  //       This is the ping-pong parity, and getting it wrong reads a stale tile
  //       rather than no tile -- a wrong answer that still looks plausible.
  //
  // The loop below mirrors gemm.cu line for line. It shares dbuf_schedule() with
  // the kernel, so the parity and the tail cannot drift; what it re-states is
  // only the publish/compute ordering around them.
  for (std::int64_t kt = 0; kt <= 200; ++kt) {
    const DbufSchedule sch = dbuf_schedule(kt);

    std::vector<int> computed(static_cast<std::size_t>(kt > 0 ? kt : 1), 0);
    // buf[i] = which tile that shared buffer currently holds (-1 = none/OOB).
    std::int64_t buf[2] = {-1, -1};
    bool parity_ok = true;

    auto publish = [&](int b, std::int64_t tile) { buf[b] = (tile < kt) ? tile : -1; };
    auto compute = [&](int b) {
      if (buf[b] < 0 || buf[b] >= kt) { parity_ok = false; return; }
      ++computed[static_cast<std::size_t>(buf[b])];
    };

    publish(0, 0);                                    // pre-loop publish
    for (std::int64_t it = 0; it < sch.pair_iters; ++it) {
      const std::int64_t t = it * 2;
      compute(0); publish(1, t + 1);
      compute(1); publish(0, t + 2);
    }
    if (sch.has_tail) compute(0);

    CHECK(parity_ok);
    // Exactly once, for every tile.
    for (std::int64_t t = 0; t < kt; ++t)
      CHECK_EQ(computed[static_cast<std::size_t>(t)], 1);
  }

  // The specific boundaries, spelled out so a regression names itself.
  CHECK_EQ(dbuf_schedule(0).pair_iters, 0);  CHECK(!dbuf_schedule(0).has_tail);
  CHECK_EQ(dbuf_schedule(1).pair_iters, 0);  CHECK(dbuf_schedule(1).has_tail);
  CHECK_EQ(dbuf_schedule(2).pair_iters, 1);  CHECK(!dbuf_schedule(2).has_tail);
  CHECK_EQ(dbuf_schedule(3).pair_iters, 1);  CHECK(dbuf_schedule(3).has_tail);
  // K=4096 with BK=8 is 512 tiles -- EVEN, so the benchmark shape never
  // exercises the tail. K=257 (a validation shape) gives 33 tiles, which does.
  CHECK_EQ(dbuf_schedule(512).pair_iters, 256);  CHECK(!dbuf_schedule(512).has_tail);
  CHECK_EQ(dbuf_schedule(33).pair_iters, 16);    CHECK(dbuf_schedule(33).has_tail);
}

void test_gemm_bank_conflict_math() {
  std::printf("test_gemm_bank_conflict_math\n");
  // The bank arithmetic that decides kGemmAPad, checked rather than asserted in
  // a comment. 32 banks of 4 B, so bank(addr_in_floats) = addr % 32, and a warp
  // pays one phase per DISTINCT ADDRESS per bank (equal addresses broadcast).
  //
  // This is a pure integer property of the layout, so it is fully checkable
  // without a GPU -- and it is the difference between an 8-way conflict paid
  // K/BK = 512 times per block and no conflict at all.
  constexpr int kBM = 128, kBK = 8;

  // Distinct addresses landing in the same bank, over the 32 lanes of a warp,
  // for the SCALAR A-store decomposition: arow = tid/BK, acol = tid%BK.
  auto scalar_store_conflict = [](int pad) {
    std::map<int, std::set<int>> per_bank;   // bank -> distinct addresses
    for (int tid = 0; tid < 32; ++tid) {
      const int arow = tid / kBK, acol = tid % kBK;
      const int addr = acol * (kBM + pad) + arow;
      per_bank[addr % 32].insert(addr);
    }
    std::size_t worst = 0;
    for (const auto& kv : per_bank) worst = std::max(worst, kv.second.size());
    return worst;
  };

  CHECK_EQ(scalar_store_conflict(0), 8u);   // the bug: lanes 0-7 all in bank 0
  CHECK_EQ(scalar_store_conflict(1), 4u);   // the reflexive "+1" fix: still 4-way
  CHECK_EQ(scalar_store_conflict(4), 1u);   // kGemmAPad: conflict-free
  // Why 4 works: stride 132 % 32 == 4, so bank = (4*acol + arow) % 32 walks
  // 0,4,8,...,28 with a 0..3 offset and covers all 32 banks exactly once.
  CHECK_EQ((kBM + 4) % 32, 4);

  // The VECTORIZED A-store decomposition (arow = tid/2, acol = 4*(tid%2)) must
  // ALSO be conflict-free under the same pad -- otherwise warptile_vec4 would
  // improve the conflict degree as a side effect and its row would carry two
  // causes instead of one.
  for (int i = 0; i < 4; ++i) {
    std::map<int, std::set<int>> per_bank;
    for (int tid = 0; tid < 32; ++tid) {
      const int arow = tid / 2, acol = (tid % 2) * 4 + i;
      const int addr = acol * (kBM + 4) + arow;
      per_bank[addr % 32].insert(addr);
    }
    std::size_t worst = 0;
    for (const auto& kv : per_bank) worst = std::max(worst, kv.second.size());
    CHECK_EQ(worst, 1u);
  }

  // The B-FRAGMENT READ is the conflict warp tiling addresses, and the point is
  // that it CANNOT be eliminated by any lane permutation -- only reduced.
  // bank = (TN*thread_col + i) % 32 has period 4 in thread_col when TN = 8, so
  // at most 4 distinct banks are reachable however lanes are assigned.
  auto b_read_conflict = [](bool warp_tile) {
    std::map<int, std::set<int>> per_bank;
    for (int tid = 0; tid < 32; ++tid) {
      int thread_col;
      if (!warp_tile) {
        thread_col = tid % 16;                          // kRowMajor: 2x16 lanes
      } else {
        const int lane = tid % 32;
        thread_col = (0 /*warpCol*/) * 8 + (lane % 8);   // kWarp4x8: 4x8 lanes
      }
      const int addr = thread_col * 8;                   // i = 0
      per_bank[addr % 32].insert(addr);
    }
    std::size_t worst = 0;
    for (const auto& kv : per_bank) worst = std::max(worst, kv.second.size());
    return worst;
  };
  CHECK_EQ(b_read_conflict(false), 4u);   // 4-way -> 4 phases (+1 for A) = 5
  CHECK_EQ(b_read_conflict(true),  2u);   // 2-way -> 2 phases (+1 for A) = 3
  // 5 -> 3 phases is a 1.67x cut, NOT elimination. The Phase-0 header claimed
  // the conflict was "removed"; RESULTS.md sec 3d records the retraction and the
  // revised 5-12% prediction. This assertion is what stops the claim coming back.

  // The A-fragment read is conflict-free under BOTH lane maps, and by BROADCAST
  // (few distinct addresses), not by the transpose. Conflating those two is how
  // the wrong intuition gets carried into the next kernel.
  for (bool warp_tile : {false, true}) {
    std::set<int> distinct_rows;
    for (int tid = 0; tid < 32; ++tid) {
      const int thread_row = warp_tile ? (0 * 4 + (tid % 32) / 8) : (tid / 16);
      distinct_rows.insert(thread_row);
    }
    CHECK(distinct_rows.size() <= 4u);   // <=4 distinct addresses, all broadcast
  }
}

void test_gemm_tolerances() {
  std::printf("test_gemm_tolerances\n");
  using namespace mcke::testing;

  // The tolerance must SCALE WITH K. A single constant is wrong at both ends:
  // at K=1 a GEMM is one multiply, at K=4096 the accumulation is 64x deeper.
  CHECK(tol_gemm(1) < tol_gemm(256));
  CHECK(tol_gemm(256) < tol_gemm(4096));
  // Recovers the previously hand-picked kTolGemmK256 = 1e-5 at K=256, so the new
  // function is consistent with the old constant rather than contradicting it.
  CHECK(tol_gemm(256) < 1e-5 && tol_gemm(256) > 1e-6);
  // A K=1 GEMM is a single multiply: near machine epsilon, not 1e-5.
  CHECK(tol_gemm(1) < 1e-6);
  // Guard against a zero/negative K dividing or sqrt-ing into nonsense.
  CHECK(tol_gemm(0) > 0.0);

  // The absolute floor scales with K and with the input magnitude, because the
  // rounding error is set by the size of the TERMS being summed, not by the size
  // of the result -- the same trap that produced kAbsTolReduceSum4096.
  CHECK(abs_tol_gemm(4096, 1.0) > abs_tol_gemm(256, 1.0));
  CHECK(abs_tol_gemm(4096, 2.0) > abs_tol_gemm(4096, 1.0));
  CHECK(abs_tol_gemm(4096, 1.0) > 1e-5);   // big enough to cover near-zero outputs
  CHECK(abs_tol_gemm(1, 1.0) < 1e-6);      // but not so big it hides a K=1 bug

  // spot_check_gemm must agree with reference_gemm where they overlap -- it is a
  // cheap oracle for the benchmark shape, so it had better be the SAME oracle.
  {
    const std::int64_t m = 17, n = 13, k = 29;
    std::vector<float> a(m * k), b(k * n), c(m * n, 0.0f);
    fill_random(a.data(), a.size(), 0x11AAull, -1.0f, 1.0f);
    fill_random(b.data(), b.size(), 0x22BBull, -1.0f, 1.0f);
    reference_gemm(a.data(), b.data(), c.data(), m, n, k, 1.0f, 0.0f);
    const auto ok = spot_check_gemm(a.data(), b.data(), c.data(), m, n, k, 1.0f,
                                    200, 0x33CCull, tol_gemm(k), abs_tol_gemm(k, 1.0));
    CHECK_EQ(ok.mismatches, 0u);
    CHECK_EQ(ok.checked, 200u);

    // And it must actually FAIL on corrupted output -- a spot check that never
    // fires is worse than none, because it reads as evidence.
    c[(m / 2) * n + (n / 2)] += 1.0f;
    const auto bad = spot_check_gemm(a.data(), b.data(), c.data(), m, n, k, 1.0f,
                                     4000, 0x33CCull, tol_gemm(k), abs_tol_gemm(k, 1.0));
    CHECK(bad.mismatches > 0u);
  }
}

// ---------------------------------------------------------------------------
// The independent-oracle cross-check.
//
// Reads tests/data/reference_vectors.txt (generated by tools/gen_reference.py,
// committed, never a build step) and replays every case through
// tests/reference.hpp.
//
// WHAT THIS CATCHES THAT NOTHING ELSE CAN. reference.hpp is the oracle every
// CUDA kernel is judged against, so nothing in the project judges IT. If the C++
// reference and a kernel share a misunderstanding -- both treating B as
// column-major, say -- they agree perfectly and validation passes while the
// answer is wrong. Only an oracle derived from a different source breaks that,
// which is why the Python side re-derives each operation from its definition
// instead of porting the C++.
//
// For `exact` cases the inputs are multiples of 0.25, so every product and
// partial sum is exact in f32 and agreement must be BIT-FOR-BIT -- no tolerance,
// no room to explain a mismatch away as rounding.
// ---------------------------------------------------------------------------
struct RefCase {
  std::string name, kind, mode;
  std::map<std::string, long long> ints;
  std::map<std::string, float> floats;
  std::map<std::string, std::vector<float>> arrays;
};

float bits_to_f32(const std::string& hex) {
  const std::uint32_t u = static_cast<std::uint32_t>(std::strtoul(hex.c_str(), nullptr, 16));
  float f;
  std::memcpy(&f, &u, sizeof(f));   // memcpy, not a union or a reinterpret_cast:
  return f;                         // the only strictly-conforming type pun
}

std::vector<RefCase> load_reference_vectors(std::string* where) {
  // Tests may run from the repo root (the documented one-command clang++ line)
  // or from a build directory (ctest), so try both rather than assuming.
  const char* candidates[] = {"tests/data/reference_vectors.txt",
                              "../tests/data/reference_vectors.txt",
                              "../../tests/data/reference_vectors.txt"};
  std::ifstream in;
  for (const char* p : candidates) {
    in.open(p);
    if (in.is_open()) { *where = p; break; }
    in.clear();
  }
  std::vector<RefCase> cases;
  if (!in.is_open()) return cases;

  std::string tok;
  while (in >> tok) {
    if (tok == "#") { std::getline(in, tok); continue; }
    if (tok[0] == '#') { std::getline(in, tok); continue; }
    if (tok != "case") continue;
    RefCase c;
    in >> c.name;
    while (in >> tok && tok != "end") {
      if (tok == "kind") in >> c.kind;
      else if (tok == "mode") in >> c.mode;
      else if (tok == "int") { std::string k; long long v; in >> k >> v; c.ints[k] = v; }
      else if (tok == "float") { std::string k, h; in >> k >> h; c.floats[k] = bits_to_f32(h); }
      else if (tok == "array") {
        std::string k; std::size_t n; in >> k >> n;
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) { std::string h; in >> h; v[i] = bits_to_f32(h); }
        c.arrays[k] = std::move(v);
      }
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

void test_reference_vectors() {
  std::printf("test_reference_vectors\n");
  std::string path;
  const std::vector<RefCase> cases = load_reference_vectors(&path);

  // FAIL, do not skip. A silently-skipped oracle check reads exactly like a
  // passing one in the summary line, which is the worst of both worlds.
  CHECK(!cases.empty());
  if (cases.empty()) {
    std::printf("  FAIL could not open tests/data/reference_vectors.txt "
                "(regenerate: python3 tools/gen_reference.py > tests/data/reference_vectors.txt)\n");
    return;
  }
  std::printf("  loaded %zu cases from %s\n", cases.size(), path.c_str());

  std::size_t exact_cases = 0;
  for (const RefCase& c : cases) {
    const auto exp_it = c.arrays.find("expect");
    CHECK(exp_it != c.arrays.end());
    if (exp_it == c.arrays.end()) continue;
    const std::vector<float>& want = exp_it->second;
    std::vector<float> got(want.size(), 0.0f);

    if (c.kind == "gemm") {
      const auto m = c.ints.at("m"), n = c.ints.at("n"), k = c.ints.at("k");
      got = c.arrays.at("c");     // reference_gemm reads C for the beta term
      mcke::testing::reference_gemm(c.arrays.at("a").data(), c.arrays.at("b").data(),
                                    got.data(), m, n, k,
                                    c.floats.at("alpha"), c.floats.at("beta"));
    } else if (c.kind == "bias_act") {
      const auto rows = c.ints.at("rows"), cols = c.ints.at("cols");
      mcke::testing::reference_bias_act(
          c.arrays.at("x").data(), c.arrays.at("bias").data(), got.data(), rows, cols,
          static_cast<mcke::kernels::Activation>(c.ints.at("act")));
    } else if (c.kind == "row_reduce") {
      const auto rows = c.ints.at("rows"), cols = c.ints.at("cols");
      mcke::testing::reference_row_reduce(
          c.arrays.at("x").data(), got.data(), rows, cols,
          static_cast<mcke::kernels::ReduceKind>(c.ints.at("kind")));
    } else if (c.kind == "row_softmax") {
      const auto rows = c.ints.at("rows"), cols = c.ints.at("cols");
      mcke::testing::reference_row_softmax(c.arrays.at("x").data(), got.data(), rows, cols);
    } else {
      CHECK(false);   // an unknown kind means the generator and reader drifted
      continue;
    }

    if (c.mode == "exact") {
      ++exact_cases;
      // Bit-for-bit. Comparing the bit patterns rather than the values also
      // catches a -0.0 / +0.0 disagreement, which `==` would call equal.
      std::size_t bad = 0;
      for (std::size_t i = 0; i < want.size(); ++i) {
        std::uint32_t g, w;
        std::memcpy(&g, &got[i], 4);
        std::memcpy(&w, &want[i], 4);
        if (g != w) ++bad;
      }
      CHECK_EQ(bad, 0u);
      if (bad) std::printf("  FAIL %s (%s): %zu/%zu elements not bit-exact\n",
                           c.name.c_str(), c.kind.c_str(), bad, want.size());
    } else {
      // exp/erf/tanh cases: the formula is under test, not libm. A few ULP of
      // libm disagreement between Python and libc++ is expected and uninteresting.
      const auto r = mcke::testing::compare(got.data(), want.data(), want.size(),
                                            1e-6, 1e-7);
      CHECK(r.ok());
      if (!r.ok()) std::printf("  FAIL %s (%s): %s\n", c.name.c_str(), c.kind.c_str(),
                               r.to_string().c_str());
    }
  }
  // If the exact cases ever vanish, the file has been regenerated by something
  // that dropped the quarter-integer inputs, and the strongest check in this
  // test silently became a tolerance check.
  CHECK(exact_cases >= 10);
}

}  // namespace

int main() {
  std::printf("=== MCKE host-core tests (MCKE_WITH_CUDA=%d) ===\n", MCKE_WITH_CUDA);
  test_buddy_math();
  test_shape();
  test_raw_allocator();
  test_buddy_geometry();
  test_buddy_exact_levels();
  test_buddy_split_signature();
  test_buddy_exhaust_and_coalesce();
  test_buddy_property_no_overlap();
  test_buddy_internal_waste();
  test_buddy_too_large_and_edges();
  test_buddy_growth();
  test_buddy_bypass();
  test_buddy_trim_preserves_slab_ids();
  test_buddy_bad_handles();
  test_buddy_validate_detects_corruption();
  test_buddy_stats_thesis();
  test_freelist_size_classes();
  test_freelist_basic();
  test_freelist_no_coalescing();
  test_freelist_external_fragmentation();
  test_freelist_beats_buddy_on_dl_shapes();
  test_freelist_split_large_blocks();
  test_freelist_bad_handles();
  test_latency_stats_edge_cases();
  test_reference_compare();
  test_reference_kernels();
  test_online_softmax_recurrence();
  test_gemm_tile_geometry();
  test_gemm_occupancy_limiters();
  test_gemm_dbuf_schedule();
  test_gemm_bank_conflict_math();
  test_gemm_tolerances();
  test_reference_vectors();
#if !MCKE_WITH_CUDA
  // Stream-ordered reuse: needs fake stream handles, which are only safe to
  // fabricate when no driver will ever see them.
  test_buddy_rule1_same_stream();
  test_buddy_cross_stream_defers();
  test_buddy_deferred_does_not_coalesce();
  test_buddy_policy_same_stream_only();
  test_buddy_policy_per_free_event();
  test_freelist_reuse_policies();
#endif
  std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
