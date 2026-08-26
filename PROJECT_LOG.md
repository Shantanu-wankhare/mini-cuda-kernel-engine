# PROJECT_LOG.md

Chronological engineering log for MCKE. Newest entries at the bottom.
Every session appends one dated entry. See `CLAUDE.md` §7 for the required
contents. This is the project's record — the owner's personal learning notes live
in `LEARNING_LOG.md`.

---

## 2026-08-24 — Session 1: Architecture and scaffold

**Environment:** MacBook Air (Apple Silicon, M-series), Apple clang 21.0.0,
no CUDA. Host-only build. No GPU was involved in this session.

### What was built

Directory scaffold plus the complete interface layer. Files created:

**Build**
- `CMakeLists.txt` — CMake 3.24+, CUDA as an *optional* language. Key decision:
  `MCKE_ENABLE_CUDA=OFF` must produce a working, testable library, so the
  MacBook is a real development environment rather than an editor.
  `CMAKE_CUDA_ARCHITECTURES=native` so one command line works on sm_75 through
  sm_120. Default build type `RelWithDebInfo` (not `Release`) because Nsight
  Compute needs line tables to map SASS back to source.

**Core (`include/mcke/core/`)**
- `config.hpp` — portability macros. Separates `MCKE_WITH_CUDA` (build-wide,
  from CMake) from `__CUDACC__` (per translation unit, set by nvcc). Hardware
  constants: `kWarpSize=32`, `kDeviceAlignment=256`, `kCacheLineBytes=128`.
- `status.hpp` — `Status` / `StatusOr<T>` + `MCKE_RETURN_IF_ERROR`. Split
  policy: status codes for expected failures (OOM), exceptions for programmer
  errors. No `std::expected` — that is C++23, we target C++20.
- `dtype.hpp` — runtime `DType` tag + compile-time `DTypeOf<T>` + `DeviceScalar`
  concept.
- `device.hpp` — `DeviceInfo`, a POD snapshot of the ~20 `cudaDeviceProp` fields
  we actually use, plus `peak_dram_gb_s()`. Exists so scheduler/tile-selection
  code has no CUDA type dependency and stays compilable on macOS.

**Runtime boundary (`include/mcke/runtime/`)**
- `cuda_check.hpp` — `MCKE_CUDA_RETURN_IF_ERROR` / `MCKE_CUDA_CHECK` /
  `MCKE_CUDA_CHECK_LAUNCH`. Documents the sync-vs-async error distinction and
  the fact that async errors are *sticky* (unrecoverable for the context).
- `stream.hpp` — RAII `Stream` and `Event`. Non-copyable, movable. Streams
  created with `cudaStreamNonBlocking` so they do not implicitly serialise
  against the legacy NULL stream. Events default to `cudaEventDisableTiming`
  unless explicitly created for measurement.

**Memory (`include/mcke/memory/`)**
- `buddy_math.hpp` — the full constexpr buddy-tree arithmetic (heap-order
  indexing, `buddy_of` via the XOR trick, size→level mapping), with
  `static_assert` self-checks that run at compile time.
- `allocator.hpp` — `DeviceAllocator` interface, `Allocation`, `AllocatorStats`.
  Every `allocate`/`deallocate` takes a stream: stream-ordered semantics are in
  the type signature, not in a comment.
- `buddy_allocator.hpp`, `freelist_allocator.hpp` — declarations for Phase 2a/2b.

**Tensor (`include/mcke/tensor/`)**
- `shape.hpp` — fixed-capacity (rank ≤ 5) POD shape with strides,
  `rows()`/`cols()` collapse. Uses raw C arrays, not `std::array`, so the type is
  unambiguously device-safe.
- `tensor.hpp` — `Storage` (owning, `shared_ptr`, records `last_use_stream`) /
  `Tensor` (view) / `TensorRef<T>` (kernel-argument POD).

**Graph (`include/mcke/graph/`)**
- `op.hpp` — polymorphic `Op` with mandatory `cost()` returning ideal FLOPs and
  bytes; `GemmOp`, `BiasActOp`, `ReduceOp`, `SoftmaxOp` with their param structs
  and kernel-variant enums.
- `graph.hpp` — SSA-style DAG. Edges are *derived* from tensor def/use, never
  declared by the user. Kahn's algorithm (not DFS) because its wave structure is
  exactly the parallelism information the scheduler needs.
- `executor.hpp` — `ExecutionPlan` / `GraphExecutor`, three schedule policies
  (`kSequential`, `kLevelParallel`, `kChainGreedy`), two memory policies.

**Kernels + profiling**
- `kernels/kernels.hpp` — launcher declarations; documents why tile sizes must be
  template parameters (unrolling, register-resident accumulators).
- `kernels/elementwise.cu` — Phase-0 vector add, grid-stride, `__restrict__`,
  64-bit indexing, fully commented.
- `profiling/profiler.hpp` + `src/core/profiler.cpp` — `NvtxRange`,
  `KernelRecord`, `Roofline`, `Profiler::time_op` (warmup → per-iteration event
  pairs → single sync → median), CSV export.
- `src/core/device.cpp`, `src/memory/allocator.cpp` — CUDA and host backends;
  `RawDeviceAllocator` as the benchmark control group.
- `tools/device_query.cpp`, `tools/smoke_vector_add.cpp`,
  `tests/test_host_core.cpp`.

### Design decisions made (and alternatives rejected)

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Host/device split | CUDA types confined to `mcke::rt`; everything above is plain C++20 | CUDA types throughout | Makes ~60% of the codebase testable on a Mac with no GPU |
| Allocator strategy | Buddy first, then size-class free-list, then A/B them | Pick one | The comparison is the deliverable; both have a workload where they win |
| Allocator dispatch | Virtual `DeviceAllocator` | CRTP / templates | ~2 ns vtable vs. µs-scale launches; enables runtime `--allocator=` A/B in one process |
| Op dispatch | Virtual `Op` | `enum` + switch, `std::variant` | Dispatch happens once per launch; per-op file locality is worth far more |
| Graph edges | Derived from tensor def/use (SSA) | User-declared edge list | A declared edge can disagree with data flow → silent races |
| Topological sort | Kahn | DFS post-order | Kahn's waves *are* the parallelism structure; DFS discards it |
| Cross-node deps | `cudaEventRecord` + `cudaStreamWaitEvent` | `cudaStreamSynchronize` | Stream-sync is a host barrier and destroys all overlap |
| Strides | Contiguous row-major only in Phases 0-4 | General strides now | General indexing costs a MAD per dim per element and hides coalescing behaviour |
| Error handling | `Status` for OOM/shape errors, throw for bugs | One or the other | OOM is recoverable policy; a rank mismatch is a bug |
| Shape storage | Raw C arrays, rank ≤ 5 | `std::vector`, `std::array` | Must be a device-copyable plain aggregate; `std::array::operator[]` is host-constexpr |
| Test framework | 30-line macro harness | GoogleTest | Must build offline in one command on any machine; GBench comes in Phase 5 where its statistics are actually needed |

### Benchmarks run

None — no GPU. One measurement worth recording anyway, from the buddy-math test:

| Measurement | Value | Note |
|---|---|---|
| Worst-case buddy internal fragmentation | **50.0%** (4097 B request → 8192 B block) | Confirms the theoretical ~2x bound. This is the number `FreeListAllocator` must beat in Phase 2c. |

### Verification performed (actually run, on macOS)

```
clang++ -std=c++20 -Wall -Wextra -I include -DMCKE_WITH_CUDA=0 \
  tests/test_host_core.cpp src/core/device.cpp src/memory/allocator.cpp \
  -o /tmp/mcke_tests && /tmp/mcke_tests
→ === 4410 checks, 0 failures ===
```

- All 17 headers compile clean under `-Wall -Wextra` in host-only mode.
- `mcke_device_query` builds and correctly reports "no CUDA device" on macOS.
- Buddy-math property tests: buddy involution, buddy/parent consistency,
  per-level arena tiling, offset uniqueness and alignment, offset↔node inverse
  mapping, size→level rounding, sentinel for oversized requests.

### Not verified — read this before trusting anything CUDA

`kernels/elementwise.cu`, `tools/smoke_vector_add.cpp`, and the
`MCKE_WITH_CUDA=1` branches of `device.cpp` / `stream.hpp` / `cuda_check.hpp`
**have never been compiled by nvcc**. They are written from knowledge of the
APIs, not from a passing build. The first GPU session's job #1 is to compile them
and fix whatever falls out.

### What's next

**Phase 1 (Colab, ~30 min):** compile with CUDA on, run `mcke_device_query` and
paste output into this log, run `mcke_smoke`, record achieved vector-add
bandwidth vs. `peak_dram_gb_s()`. Expect 70-85% of peak for a 3-stream
elementwise kernel; anything under 50% means something is wrong (pageable
staging, too-small grid, or clocks).

Then Phase 2a: implement `BuddyAllocator` (host logic → testable on the Mac
first, GPU only for the final numbers).
