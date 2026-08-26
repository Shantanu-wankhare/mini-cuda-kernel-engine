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
//        -o /tmp/mcke_tests && /tmp/mcke_tests
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <set>
#include <vector>

#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_math.hpp"
#include "mcke/tensor/shape.hpp"

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

}  // namespace

int main() {
  std::printf("=== MCKE host-core tests (MCKE_WITH_CUDA=%d) ===\n", MCKE_WITH_CUDA);
  test_buddy_math();
  test_shape();
  test_raw_allocator();
  std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
