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

---

## 2026-08-26 — Session 2: First light on Colab T4 (Phase 1)

**Environment:** Google Colab Pro, T4 GPU runtime, Standard RAM. Driver
580.82.07 (reports CUDA 13.0 capability), CUDA toolkit/nvcc 12.8.93. Repo
cloned over HTTPS with a short-lived, repo-scoped fine-grained PAT.

### What was built

- `bench/stream_triad.cu` — classic STREAM triad
  (`a[i] = b[i] + scalar*c[i]`), measures *achieved* DRAM bandwidth.
- `bench/fma_peak.cu` — register-only f32 FMA microbenchmark: 8 independent
  accumulator chains per thread (the ILP width), each running the contraction
  recurrence `acc = acc*0.999 + 1` for 100,000 iterations — `|m|<1` keeps the
  value bounded near its fixed point (1000) for the whole run, so there is no
  overflow/denormal risk that could silently corrupt the timing. Measures
  *achieved* f32 FMA throughput.
- Both wired into `CMakeLists.txt` under the existing `MCKE_BUILD_BENCH`
  option. Landed via PR #1 (`phase1/measured-peak-benchmarks` -> `main`),
  merged only after the Colab run confirmed everything built and ran.
- **Bug fix, committed directly to `main` (`8a4117d`), no branch/PR:**
  `DeviceInfo::peak_dram_gb_s()` was missing a factor of 2.

### What was learned — including something that turned out wrong

The first Colab run reported `mcke_smoke` and `mcke_stream_triad` achieving
**~150% of "peak" DRAM bandwidth** — a physical impossibility. That is exactly
the useful kind of signal: it means the "peak" being compared against was
wrong, not that the kernel exceeded physics.

Root cause: `peak_dram_gb_s()`'s original comment asserted
`cudaDeviceProp::memoryClockRate` "is already the effective data rate, so no
extra x2 for DDR." That was wrong. The field reports **one edge** of a
double-data-rate clock — the same convention NVIDIA's own `deviceQuery` sample
uses (`2.0 * memoryClockRate * busWidth/8`). Confirmed against T4's published
spec (320 GB/s): our un-doubled formula gave 160.032 GB/s, exactly half.
Doubled, it gives 320.064 GB/s, matching spec, and both `vector_add` (75.2%)
and `stream_triad` (73.6%) now land at a believable fraction of it — the two
kernels agree with each other to within ~2%, which is what gives confidence
the *measurement* methodology was sound the whole time, even while the
*formula* was broken.

This is the concrete case the project's own docs warned about only in the
abstract ("never trust a spec-sheet number, measure it") — it turns out the
theoretical *formula* itself needed the same skepticism, not just the
achieved-vs-formula comparison built on top of it.

Secondary finding, acted on later the same session: `RESULTS.md`'s row format
calls for both **median and min** per its own rule 3, but `Profiler::time_op`
(`include/mcke/profiling/profiler.hpp`) computed and stored only the
**median** — there was no min in `KernelRecord` at all. Fixed before the
Colab rerun rather than deferred to Phase 5: `KernelRecord::ms` was split into
`median_ms` (unchanged semantics — every derived metric, `tflops()`/
`gb_per_s()`, is still computed from it) and a new `min_ms`, computed in
`time_op` via `std::min_element` on the *same* set of timed iterations, before
`std::nth_element` partitions the vector for the median (order doesn't
actually matter here — `nth_element` only reorders, never removes, elements —
but computing min first keeps the two computations obviously independent on
inspection rather than relying on that fact). `summary_table()` and
`write_csv()` in `src/core/profiler.cpp` were updated to print/export both
columns. Verified by recompiling `device_query` + `test_host_core` on the Mac
in host-only mode after the change: still 4410 checks, 0 failures, and the
new binaries link and run correctly — this header is included by every
benchmark from here on, so it was worth confirming before handing it back for
the Colab rerun rather than finding a break there.

Also learned in passing: authenticating to a private GitHub repo from Colab
via `Authorization: Bearer <PAT>` in a `git -c http.extraHeader` does **not**
work against GitHub's git-over-HTTPS endpoint (fails with "could not read
Username", i.e. the header was silently ignored and git fell back to an
interactive prompt). `Authorization: Basic <base64(x-access-token:PAT)>` does
work — the same convention GitHub Actions uses internally. Also: `base64`
wraps long output at 76 columns by default, which would have corrupted the
header if not passed `-w0`.

### Benchmarks run (Colab, Tesla T4, driver 580.82.07, CUDA toolkit 12.8.93)

| Program | Result |
|---|---|
| `mcke_device_query` | sm_75, 40 SMs, 64 KiB smem/SM, peak DRAM BW (corrected formula) = 320.064 GB/s |
| `mcke_smoke` (vector_add) | 240.6 GB/s achieved (75.2% of corrected peak); verification OK, 0/67,108,864 mismatches |
| `mcke_stream_triad` | **235.5 GB/s achieved** (73.6% of corrected peak) — canonical *measured bandwidth* denominator for T4 from here on |
| `mcke_fma_peak` | **8.059 TFLOP/s** measured sustained f32 FMA — canonical *measured compute peak* denominator for T4 from here on |
| `ctest` / `mcke_test_host_core` | 4410 checks, 0 failures — first time this host logic has compiled and run under gcc + nvcc's host compiler rather than Apple clang; confirms nothing clang-specific crept in |

Derived: T4 roofline ridge point (measured/measured) =
8.059e12 / 235.5e9 approx **34.2 FLOP/byte**. Kernels below that AI are
memory-bound on this GPU; our planned GEMM (AI approx 682) sits deep in the
compute-bound region, row-reduce (AI approx 0.25) deep in memory-bound —
structurally as `docs/PROFILING.md` sec 2 predicted, now anchored to a real
number for this specific chip.

### Design decisions taken (and alternatives rejected)

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Colab auth for a private repo | HTTPS + fine-grained PAT, `http.extraHeader="Authorization: Basic ..."`, `x-access-token` convention | `Authorization: Bearer` header (tried first) | Bearer isn't honored by GitHub's git-smart-HTTP endpoint; Basic with `x-access-token` is what GitHub Actions itself uses |
| GPU/RAM tier for Phase 1 | T4, Standard RAM | L4 / A100 | Phase 1 needs no compute headroom (scoped in `docs/ROADMAP.md` as ~2 credits); save bigger GPUs for Phase 3 GEMM sweeps |
| Bandwidth-formula bug fix | Committed directly to `main`, no branch/PR | Branch + PR (as done for the new benchmark files) | Small, single-reasoning-chain correction to existing code, not new functionality |

### What's next

Rerun the four Phase 1 programs on Colab once more to get real min-ms numbers
for the `RESULTS.md` §1 rows currently marked "pending rerun", then move to
Phase 2: `BuddyAllocator` — design and unit-test the split/merge and
stream-ordered pending-free logic on the Mac (the logic needs no GPU at all),
then bring real `raw_malloc_calls`-vs-`alloc_calls` and fragmentation numbers
back from a GPU session.
