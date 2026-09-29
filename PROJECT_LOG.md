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

**Phase 1 closed out**, same session: reran all four programs on Colab T4 after
the min-tracking fix. Final numbers: `vector_add` 240.5 GB/s (median 3.348 ms /
min 3.344 ms, 75.1% of corrected peak), `stream_triad` 235.4 GB/s (median
3.421 ms / min 3.417 ms, 73.6%), `fma_peak` 8.130 TFLOP/s (vs. 8.050 and 8.059
TFLOP/s on the two earlier runs — about 1% run-to-run spread, unremarkable).
Median and min agree to within 0.1-0.2% on both bandwidth kernels, which is
itself informative: it says this T4 session was clean (idle, 41 degC, no other
tenants) rather than something we'd need to caveat. `RESULTS.md` sec 0 and
sec 1 now hold these as the final Phase 1 figures; `docs/ROADMAP.md`'s Phase 1
exit criteria are met.

Next: Phase 2 — `BuddyAllocator`. Design and unit-test the split/merge and
stream-ordered pending-free logic on the Mac (the logic needs no GPU at all),
then bring real `raw_malloc_calls`-vs-`alloc_calls` and fragmentation numbers
back from a GPU session.

---

## 2026-08-26 to 2026-08-29 — Session 3: Phase 2 (2a-2d) — allocators, bench, race test

**Environment:** MacBook Air (Apple Silicon), Apple clang 21.0.0, host-only, for
all design and implementation. Colab Tesla T4 (driver 580.82.07, nvcc 12.8.93)
for two verification runs (2026-08-28 and 2026-08-29) via a separate forked
session, per the owner's request to keep GPU fix-compile-run loops out of the
main design thread.

### What was built

**Memory (`include/mcke/memory/`, `src/memory/`)**
- `buddy_allocator.hpp/.cpp` — full implementation: power-of-two slab rounding
  with OOM halving-retry, split-downward `alloc_node` (search toward the root
  via `nonempty_mask` + `countl_zero`), coalesce-upward `free_node`, an
  iterative (non-recursive) `validate()` checking nine invariants, `dump_free_map()`.
- `freelist_allocator.hpp/.cpp` — linear (512 B granularity) + power-of-two
  size-class ladder, bump-pointer slab carving, no coalescing by design,
  optional `split_large_blocks`.
- `reuse_policy.hpp` — the three cross-stream reuse policies
  (`kSameStreamOnly`, `kCoarseStreamPoll`, `kPerFreeEvent`) and the single
  `pending_reusable()` decision function BOTH allocators route through.
  Extracted into its own header specifically so the Phase 2c buddy-vs-freelist
  comparison could not be contaminated by the two pools disagreeing about what
  "safe to reuse" means.
- `settle_pending()` added to `DeviceAllocator` (2026-08-29, post-Colab fix,
  see below) — reclaim parked blocks without releasing slabs, distinct from
  `trim()` which also releases idle slabs to the driver.

**Profiling (`include/mcke/profiling/host_timer.hpp`, `src/core/host_timer.cpp`)**
- `ClockCalibration`, `LatencyStats` (retained-sample nearest-rank percentiles,
  log2 histogram), `HostTimer` — kept as a sibling to `profiler.hpp` rather than
  merged into it, because that file's own banner declares itself GPU-roofline
  scoped and conditionally includes NVTX; a host-malloc latency type there would
  make the banner false.

**Benchmark (`bench/alloc_bench.cpp`)**
- Deterministic trace generator: `uniform_pow2` (13 power-of-two classes,
  proportional-control live-set targeting, LIFO warmup → random-victim churn →
  FIFO+size-ratchet adversarial → full drain) and `dl_transformer` (GPT-2-small
  shapes, long-lived weights + short-lived per-layer/per-decode-step
  activations), both seeded from a fixed `std::mt19937_64` using **raw engine
  output only** (never `<random>` distributions — those are not portably
  specified, engines are).
- `dl_transformer_bypass`: the DL trace plus a 147 MiB embedding table, kept as
  its own trace so its one extra permanent driver allocation never muddies the
  clean traces' `raw_malloc_calls` flatline. Plus a fourth, isolated bypass
  probe (allocate/assert/free, no churn).
- Reports three named fragmentation ratios (`block_eff`, `reserv_eff`,
  `utilisation`) rather than the header's single `utilisation()`, because that
  one number conflates internal fragmentation with slab over-provisioning.
- All allocators run under all three reuse policies (7 configs: raw + buddy×3 +
  freelist×3), so the deallocate-latency comparison decomposes into "pure
  bookkeeping" vs. "+ one `cudaStreamQuery`" vs. "+ one `cudaEventRecord`".

**Test (`tests/test_stream_safety.cu`, `tests/test_access.hpp`)**
- The repo's first `.cu` test target. Constructs a real cross-stream reuse race
  on hardware: a naive one-block pool with no stream tracking (expected to
  corrupt), each allocator × policy (expected clean), a same-stream control
  (reuses the identical pointer, still must be clean), and a positive check that
  `RawDeviceAllocator` is safe because `cudaFree` synchronises.
- Determinism by construction, not a timed spin: the reader blocks on a flag in
  mapped pinned host memory; the host only releases it after
  `cudaStreamSynchronize` proves the corrupting write already landed. A timed
  spin was in the original design and was replaced after a dedicated red-team
  pass found it only probabilistically correct (tuned to one GPU's clock,
  silently degrading elsewhere) — see rejected alternatives below.
- `test_access.hpp` extracted from `test_host_core.cpp` as a shared friend-access
  header so both the host test and the `.cu` test assert on the identical
  `pending_count()` definition rather than risking two copies drifting apart.

### Bugs found and fixed (four, in order of how they were found)

1. **A real bug in Phase 0 header claims**, found by design review before any
   GPU touched the code: `trim()` as originally declared would have erased from
   `slabs_`, invalidating the `slab_id` in every outstanding `Allocation`;
   `Allocation::slab_id` defaulted to 0, colliding with real slab 0; and
   `reclaim_completed` was literally unwritable — no way to query a bare
   `StreamHandle`. Fixed with `kBypassSlabId` sentinel, "mark dead, never erase"
   slabs, and a new `rt::stream_query()` free function.
2. **`free_node` was O(n), not O(log n), as declared.** Removing a coalesced
   buddy from the middle of `std::vector<size_t>` free_lists is a linear scan,
   and a 256 MiB slab's deepest level can hold 524,288 entries — genuinely
   reachable state. Fixed by adding `pos[]` (node → its index in its own free
   list) for O(1) swap-and-pop removal.
3. **A real crash, found by a pre-Colab code audit**: `test_buddy_property_no_overlap`
   did a host `std::memset`/byte-read on `r->ptr`, which is a `cudaMalloc`
   pointer in a CUDA build — an immediate segfault, in the flagship
   20,000-iteration test, registered unconditionally. Guarded to
   `#if !MCKE_WITH_CUDA`, since the allocator's address computation is pure host
   arithmetic and identical in both backends — host coverage is sufficient, a
   device-side version would only re-prove something backend-independent.
4. **`settle_pending()` — found by the actual Colab run, not by review.** The
   first GPU run of `test_stream_safety` failed two arms
   (`buddy/same_stream_only`, `freelist/same_stream_only`) on a secondary
   diagnostic ("did not park + refuse cross-stream reclaim"), even though the
   actual safety property held (both CLEAN 20/20). Root cause: under
   `kSameStreamOnly`, a parked block only settles via a *same-stream* reclaim;
   the per-trial warm-up round-trip's own frees parked unconditionally with no
   later same-stream allocate *within the warm-up* to reclaim them, so residue
   from warm-up polluted the trial's own pending-count delta. The identical
   mechanism explained `alloc_bench`'s "silently overshot!" lines on
   `uniform_pow2`: a trace's final drain-phase free has no later same-trace
   allocate to settle it, so `largest_free_block` understated capacity until the
   probe's own allocate reclaimed (and for buddy, coalesced) the leftover as an
   unrelated side effect. Fixed by adding a **public** `DeviceAllocator::settle_pending()`
   — deliberately not `trim()`, which also releases idle slabs and would have
   undone the warm-up's whole purpose (keeping a driver call out of the measured
   window) or invalidated an already-captured fragmentation snapshot.

### Design decisions taken (and alternatives rejected)

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Free-list node removal | `pos[]` index array, O(1) swap-and-pop | Linear scan (as originally declared) | A 256 MiB slab's level can hold 524,288 free entries; the "O(log n) free" claim needs this to be true, not aspirational |
| Cross-stream reuse policy | All three (`kSameStreamOnly`/`kCoarseStreamPoll`/`kPerFreeEvent`) implemented and benchmarked, sharing one decision function | Pick one up front | Owner's explicit call; also the only way the Colab deallocate-latency table decomposes into "bookkeeping" vs. "probe cost" |
| Race-test determinism | Host-released mapped-pinned gate flag | A timed spin (~100 ms, tuned per-GPU) | Red-team review: a timed spin is only probabilistically correct and silently degrades on a different GPU; a false CLEAN would read as proof of safety |
| Race-test corruption check | Integer mismatch count via one atomicAdd per thread | Warp-shuffle block reduction | The repo has no `__shfl_down_sync` anywhere yet (that's Phase 3b's teaching content); a test whose job is to be unimpeachable shouldn't add a correctness dependency on unproven-in-this-project device code |
| Settling parked blocks before a measurement | New public `DeviceAllocator::settle_pending()` | Reuse the existing `trim()` | `trim()` also releases idle slabs to the driver — exactly the side effect that would undo a warm-up or invalidate an already-captured stat |
| Stream-safety unsafe control | A ~25-line `naive_pool` in the test's own anonymous namespace | A `stream_ordered=false` flag on `BuddyConfig` | The flag would be a public header, letting anyone disable the safety property in shipped code forever; a toy pool in a test TU can't escape its translation unit |

### Benchmarks run

**Host (MacBook, `MCKE_WITH_CUDA=0`, 2026-08-26/27):** 37,354 checks, 0
failures (up from Phase 1's 4,410) — the full buddy/freelist test suite
including the 20,000-op property test, the stream-ordered reuse policy tests
(driven via fabricated stream handles, since a host build's real handles are
all `nullptr`), and the head-to-head allocator comparisons
(`test_freelist_no_coalescing`, `test_freelist_external_fragmentation`,
`test_freelist_beats_buddy_on_dl_shapes`). Fragmentation figures from
`alloc_bench` on this build are **authoritative** (pure host bookkeeping) and
match the Colab run byte-for-byte.

**Colab T4 (`MCKE_WITH_CUDA=1`, 2026-08-29, after the `settle_pending()` fix):**
- `ctest`: `host_core` and `stream_safety` both `Passed`. `stream_safety` full
  breakdown: `naive_pool` corrupted 524,288/524,288 elements on all 20 trials;
  all six allocator×policy arms plus the same-stream control were CLEAN 20/20
  with correct mechanism (parked + refused cross-stream reclaim, or reclaimed
  via rule 1 for the control); `raw(cudaMalloc)` confirmed stream-idle
  immediately after `deallocate`.
- `stream_triad` 235.3 GB/s, `fma_peak` 8.126 TFLOP/s — both within 0.1% of the
  Phase 1 session's numbers, confirming this machine is comparable.
- `alloc_bench` full latency table across 3 traces × 7 allocator configs (see
  `RESULTS.md` sec 2a for all figures). Headline: `raw` allocate median
  1.8-2.9 µs vs. pooled medians 56-182 ns; `raw_malloc_calls` stays at 2-6 total
  across 26k-99k logical `allocate()` calls for every pooled configuration, vs.
  being forced equal to `alloc_calls` for raw by construction.

### What was learned — including things that turned out to be wrong

- **The roadmap's freelist prediction was wrong, and the reason is more useful
  than the number.** Predicted 85-95% `block_eff` on DL shapes; measured 64.0%
  — an exact tie with buddy. `FreeListConfig::small_large_split` (1 MiB) puts
  the ladder into power-of-two mode above that point, and multi-MiB tensors are
  ~all the bytes in a transformer, so the two designs must round identically
  there. The free-list's real advantage (99.9% vs. 76.6%, pinned in
  `test_freelist_beats_buddy_on_dl_shapes`) only shows up on sub-1-MiB
  non-power-of-two shapes — the decode-step sizes, not the weights.
- **The free-path cost story flips depending on whether you look at median or
  tail.** Host build and initial reasoning suggested "buddy pays a coalesce
  cascade on free, freelist doesn't" as a clean tradeoff. Real GPU numbers:
  buddy's `same_stream` deallocate median is *lower* than freelist's on
  `dl_transformer` (56 ns vs. 123 ns) — freelist's `unordered_map` insert/erase
  costs more in the common case than buddy's usually-short coalesce check. The
  cascade is real but shows up in the tail (p999/max), not the median. A
  median-only comparison would have said the opposite of what's true.
  (Owner check-back pending: why do median and tail disagree here, and which
  one should a caller planning for worst-case latency actually read?)
- **A safety property can hold while its diagnostic is wrong** — the
  `settle_pending()` bug (#4 above) is the clean illustration: both failing
  arms were CLEAN 20/20 in the run that reported `FAIL`. Worth being able to
  tell these apart under pressure: "the test is red" is not the same claim as
  "the thing under test is broken."

### What's next

Phase 2 is closed: all 4 sub-parts (buddy, freelist, bench, race test) built,
tested on the Mac, and verified on real Colab T4 hardware, with
`docs/ROADMAP.md`'s Phase 2 exit criteria satisfied and `RESULTS.md` sec 2
holding final numbers.

Next: Phase 3 — kernels (GEMM, reductions, softmax, fused bias+GELU). Per the
owner's mode/model/effort mapping, expect Plan mode for each kernel variant's
tiling strategy, Opus, high effort for the warp-tiling/double-buffering GEMM
work specifically; correctness-critical but simpler fusion ops can run at
lower effort. `docs/ROADMAP.md` Phase 3 section has the variant-by-variant
plan already; start with `naive` GEMM to get a correctness and roofline-position
baseline before tiling.

## 2026-08-29 — Session 4: Phase 3 (3a-3c) — fusion, reduce, softmax on Colab T4

**Hardware:** macOS host-only (design, implementation, host-suite verification,
fake-CUDA type-checking) + Colab T4 (sm_75), driver 580.82.07, CUDA toolkit
12.8.93, two trips (first trip hit a runtime disconnect mid-verification;
second trip re-cloned fresh and re-ran everything in one consolidated cell).

### What was built

- `tests/reference.hpp`: CPU reference implementations for bias+activation,
  row reduction, row softmax, and small-shape GEMM, all accumulating in
  `double` so any disagreement is attributable to the GPU; `compare()` with
  mixed absolute+relative tolerance (`numpy.allclose` form); deterministic
  `fill_random` via raw `std::mt19937_64` output (not a `std::uniform_real_distribution`,
  which isn't bit-specified across libc++/libstdc++).
- `include/mcke/kernels/softmax_online.hpp`: `OnlineState{m,d}` running
  max/sum pair implementing the Milakov & Gimelshein online-softmax
  recurrence, host-testable (`MCKE_HOST_DEVICE`) without a GPU.
- `kernels/bias_act.cu`, `kernels/reduce_ops.cuh`, `kernels/reduce.cu`,
  `kernels/softmax.cu`: fused bias+{none,relu,gelu_tanh,gelu_erf} with
  vector-width 1/2/4 variants; row reduction via smem-tree, warp-shuffle, and
  two-pass (workspace-based, for row-starved shapes); three-pass and online
  one-pass row softmax.
- `bench/bias_act_bench.cpp`, `bench/reduce_bench.cpp`, `bench/softmax_bench.cpp`
  + shared `bench/bench_common.hpp` (`make_roofline()` forces both
  `peak_gb_s` and `peak_tflops` to be set explicitly — the roofline
  `peak_tflops=0` silent-zero trap from Phase 1/2 planning is now
  structurally hard to hit again).
- `scripts/fakecuda/{cuda_runtime_api.h,cuda_lang_prelude.h}` +
  `scripts/typecheck_cuda.sh`: stub CUDA runtime/cublas signatures and CUDA
  language extensions so `MCKE_WITH_CUDA=1` code can be syntax-checked with
  plain `clang++` on the Mac before ever touching a GPU. Confirmed effective —
  the actual Colab build compiled clean on the first try across all new
  kernels and benches.
- Extended `tests/test_host_core.cpp`: `test_reference_compare()`,
  `test_reference_kernels()`, `test_online_softmax_recurrence()`.
- `RESULTS.md` §3a/3b/3c filled with real measured numbers (see below);
  `docs/ROADMAP.md` Phase 3 env line updated for the two-trip Colab batching.

### Bugs found and fixed

1. **GELU validation false failures** (`bias_act_bench`, 3 configs):
   `max_abs_err≈4.77e-7` (exactly 4 ULP at magnitude ~2) against the default
   `abs_tol=1e-8`. Root cause: GELU's `(1+tanh(z))`/`(1+erf(z))` intermediate
   is O(1), so a routine few-ULP device-vs-host libm disagreement becomes an
   absolute error independent of the tiny output magnitude near the curve's
   knee — not a kernel bug. Fixed with a derived (not guessed) constant,
   `testing::kAbsTolGeluCancellation = 1e-6` (≥2× the observed error).
2. **Row-sum validation false failures** (`reduce_bench`, `kSum` only):
   `max_abs_err≈1.5e-5` at a near-zero-mean row (`want≈0.2`, routine for
   `[-1,1]` random fill, not adversarial). Root cause: summation rounding
   error scales with the magnitude of the terms being summed (~1), not the
   final sum's magnitude, so a near-cancelling row fails a pure-relative test
   even though the kernel is correct. `kMean` is unaffected because dividing
   by `cols` shrinks the value and the error floor together. Fixed with
   `testing::kAbsTolReduceSum4096 = 5e-5` (≥3× the observed error), applied
   only at the `kSum` call site — `kMean` already passed with margin.
3. **A real diagnostic bug in `compare()`'s "worst offender" tracker**, found
   while investigating #2 (didn't affect pass/fail, only the reported worst
   element): it compared `abs_err >= r.max_abs_err` after `max_abs_err` had
   already been unconditionally updated in the same loop iteration, so it
   reported "the last element to set a new global max that also happened to
   fail" rather than the true worst mismatch. Fixed with a dedicated
   `worst_mismatch_rel` variable scoped to failing elements only.
4. **A wrong prediction in my own test, caught before spending GPU time**:
   predicted the reversed-monotone-ramp softmax row (max arrives first, no
   rescaling) would be more accurate than the forward ramp (max updates every
   element, maximal rescaling) for the online recurrence. Measured on the Mac:
   the reverse case was actually *less* accurate (2.50e-07 vs 1.73e-07).
   Mechanism: rescaling both introduces error and renormalizes, keeping the
   accumulator at O(1); the reversed case instead sums a large accumulator
   against tiny tail terms, an ill-conditioned pattern that costs more than
   the rescaling saved. Both remain far inside tolerance either way — fixed
   the assertion to check what's actually guaranteed (small error, no
   NaN/Inf) instead of asserting the wrong ordering.

### Benchmarks run (Colab, Tesla T4, driver 580.82.07, CUDA toolkit 12.8.93)

Sanity re-check against the Phase 1/2 baseline before trusting any new number:
`stream_triad` 240.9 GB/s (baseline 235.4, +2.3%), `fma_peak` 8.184 TFLOP/s
(baseline 8.130, +0.7%) — same class of machine, frozen denominators kept.

Full results tables are in `RESULTS.md` §3a (fusion), §3b (row reduction),
§3c (row softmax), each with a "results vs. predictions" writeup. Headline
numbers:
- **3a fusion**: relu/gelu_tanh/gelu_erf fused-vs-unfused speedup landed at
  2.01×/2.05×/2.02×, matching the 2× prediction almost exactly. The L2-control
  experiment (512×512) did **not** collapse to the predicted ~1.0–1.2× — it
  measured 1.38×, a real partial miss worth an `ncu` L2 hit-rate follow-up.
  Vector width was flat at full occupancy (238.8–250.5 GB/s across vw1/2/4)
  and showed a real ~5–8% width-dependent gap once artificially starved to 40
  blocks (208.7–226.9 GB/s), confirming MLP only matters when occupancy is
  scarce.
- **3b row reduction**: warp-shuffle did **not** beat the smem tree by the
  predicted 10–30% — it landed within ±1.3% (sometimes fractionally slower).
  Both variants already sit at ~108–110% of the 235.4 GB/s denominator, so
  DRAM bandwidth, not the 9-vs-1 barrier count, is the limiter — an honest
  negative result. `kTwoPass` was 2.7% slower at the saturated shape
  (predicted 1–3% slower — confirmed) and 1.74× faster at the row-starved
  shape (predicted 3–10× faster — real but smaller than predicted;
  warp-shuffle's 60.8%-of-peak efficiency even when "starved" suggests
  partial multi-block-per-SM overlap the naive one-block-per-row model
  didn't account for).
- **3c row softmax**: three-pass vs. online speedup landed at 1.16×, inside
  the predicted 1.0–1.25× band — "one-pass" names the statistics passes, not
  the memory passes. Neither variant showed the predicted L2 masking of
  three-pass's extra reads (both land at 55.5%/64.5% of the bandwidth
  denominator, well below the "beats its own traffic model" outcome) —
  worth an `ncu` DRAM-bytes check before treating as settled. Online is 1.28×
  less accurate by the reference-free `Σrow−1` check, both ~2e-7, as
  predicted qualitatively though the ratio was smaller than the 2–5× guess.

### Design decisions taken (and alternatives rejected)

- Combined all Colab commands for the fresh-runtime re-verification into one
  consolidated cell with `echo "=====SECTION====="` separators, rather than
  the original one-cell-per-step flow, specifically because Colab sessions
  die mid-work and a single paste-back-everything cell is cheaper to re-run
  from scratch than re-issuing a dozen small cells after a disconnect.
- Added a stream_triad/fma_peak re-run as an explicit sanity gate before
  trusting any §3 number, rather than assuming the denominators frozen from
  Phase 1/2 still apply — this is now the standing practice for every new
  Colab session that reports numbers against those denominators.

### What was learned — including things that turned out to be wrong

- Tolerance bugs in this session were all in the *test*, not the kernel — and
  both had the same shape: a tolerance derived from "typical" magnitude
  reasoning broke down at routine near-zero outputs (GELU's knee, zero-mean
  row sums), not at some adversarial edge case. The fix in both cases was to
  measure the actual observed error and derive a named constant at ≥2-3×
  margin, not to guess a bigger number.
- Two of three "beats the prediction ceiling" arguments (§3b barrier count,
  §3c L2 masking) turned out to be already-saturated-bandwidth situations
  where the mechanism being tested had no room to show up — the absolute
  GB/s numbers, which were predicted alongside the ratios specifically to
  catch this, did their job.
- The L2-control experiment in §3a is the one result that didn't cleanly
  confirm its own hypothesis (1.38× instead of ~1.0–1.2×) — flagged rather
  than smoothed over, pending an `ncu` L2 hit-rate check.

### What's next

Phase 3 stages 1-4 (fusion, row reduction, row softmax, validation harness)
are verified on real Colab T4 hardware with all correctness checks passing
and `RESULTS.md` §3a/3b/3c holding final numbers. Not yet started: Stage 5/6
GEMM ladder (`kernels/gemm.cu`: naive → tiled_smem → tiled_regblock →
kWarpTileNoDbuf → kWarpTile double-buffered → cuBLAS reference) and
`bench/gemm_bench.cpp`, per the detailed tile/occupancy design already
produced; `tools/gen_reference.py` NumPy cross-check (the "both" validation
method the owner chose); Colab Trip 2 to verify + measure the GEMM ladder and
run Nsight Compute for occupancy hand-calc vs. measured comparison, filling
`RESULTS.md` §3d. `LEARNING_LOG.md` end-of-Phase-3 Q&A entries are also due
once Phase 3 closes, per the owner's convention (not filled without being
asked).

## 2026-08-30 — Session 5: Phase 3d (3d stages 5-6) — the GEMM ladder, Colab trip 2

**Hardware: Colab Tesla T4 (sm_75), driver 580.82.07, CUDA 13.0 runtime /
nvcc 12.8.93.** Clocks not locked (no root on hosted Colab); idle at session
start was 36°C / 9W (P8).

### What was built (Mac-side, before this Colab session)

- `include/mcke/kernels/gemm_tile.hpp`: `GemmTile`, the runtime-tile →
  compile-time-instantiation dispatch, and a four-limiter occupancy calculator
  (registers / shared memory / threads-per-SM / blocks-per-SM), all pure host
  code, unit-tested exhaustively on macOS against hand-worked cases — including
  the double-buffered k-loop's ping-pong schedule, checked for every tile count
  0–200 (a dropped-tail bug at odd tile counts was confirmed live: forcing it
  produced 103 failures).
- `kernels/gemm.cu`: the 8-row ladder — `naive_uncoalesced`, `naive`,
  `tiled_smem`, `tiled_regblock`, `warptile_nodbuf`, `warptile_dbuf`,
  `warptile_vec4`, `cublas`. Rows 4–7 are one template differing by exactly one
  argument each (`LaneMap`, `DBUF`, `VW`), enforcing the one-variable-per-row
  rule via the type system rather than discipline.
- `tools/gen_reference.py` + `tests/data/reference_vectors.txt`: an independent
  Python oracle, bit-exact on 11 of 19 cases (exactly-representable inputs), so
  a shared misunderstanding between `reference.hpp` and a kernel cannot validate
  clean.
- `bench/gemm_bench.cpp`: validates every variant against the CPU reference at
  awkward shapes, then against cuBLAS (itself validated non-square first) as a
  full-shape oracle at 4096³, plus a 1024-point random spot-check; brackets the
  timed run with `cublas` first and last as a thermal-drift check; uses the
  operation's compulsory bytes as the roofline denominator for all eight rows.
- Design review (two independent passes) found six issues before any kernel
  code ran: a described double-buffering scheme that was actually a race
  (needs two *buffers*, not two barriers); warp tiling cannot remove the
  B-fragment bank conflict, only cut it 5 phases → 3 (prediction revised down
  from "comparable to double buffering" to +5–12%); the transposed-A store
  needs a stride-**4** pad, not the reflexive stride-1 (stride-1 is still
  4-way conflicting); sm_75's blocks-per-SM cap is 16, not 32 (Volta/Ampere);
  `cublasSetStream` is mandatory given `cudaStreamNonBlocking` streams; the
  fake cuBLAS header had to move out of `cuda_runtime_api.h` into its own file
  to avoid a redeclaration error.

Host suite: 58,856 checks, 0 failures. 20 translation units clean under
`scripts/typecheck_cuda.sh`.

### The Colab run

`./build/bin/mcke_gemm_bench 4096` — correctness first (all awkward shapes,
the β≠0 read-modify-write case, and the full 4096³ shape against the validated
cuBLAS oracle plus a 1024-point spot-check) all passed with zero mismatches.
Occupancy hand-calc agreed with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
on every row, including both edge cases the design specifically predicted:
`tiled_smem` landed at 43 regs/thread, a genuine **tie** between registers and
the threads/SM cap (both give exactly 1 block); `warptile_dbuf` landed at
**exactly** 128 registers, the boundary for staying at 2 blocks rather than
falling to 1 — one register over and occupancy would have halved. Neither
register-blocked kernel spilled. `tiled_regblock`'s measured shared-memory
footprint (8320 B) matched the pad prediction exactly: 8192 + 128 B, where
128 B is `kGemmAPad=4`'s modelled cost.

**Every performance prediction missed, in the same direction, and one was
contradicted outright:**

| Row | Predicted | Actual | |
|---|---|---|---|
| naive_uncoalesced | 0.2–0.6% | 1.48% | miss, above range |
| naive | 2–4% | 4.92% | miss, just above |
| tiled_smem | 15–25% | 10.36% | miss, below range |
| tiled_regblock | 45–65% | 40.39% | miss, below range |
| warptile_nodbuf | +5–12% over regblock | **−1.8%** | **contradicted** |
| warptile_dbuf | 60–80% | 40.28% | miss, well below |
| warptile_vec4 | +10–20% over dbuf | +5.7% | miss, below range |
| cuBLAS | 75–85% | 51.05% (first) / 45.61% (last) | miss, well below |

A real, measured cause for part of this: `cublas` bracketed the run at 33.12 ms
first and 37.07 ms last — **+11.9%**, past this project's 3% drift threshold.
The ladder runs slow-rows-first, so the fast rows near the end were measured on
a warmer chip than the frozen 8.130 TFLOP/s denominator (from a cool Phase-1
session) assumes; `warptile_vec4` against the **hot** `cublas_last` figure
gives 93.4%, not 42.6%. This does not explain `tiled_smem`'s miss (its own
banner already flagged the risk: 1024 threads/block occupies the entire SM
with one resident block, so there is no second block to hide the two
`__syncthreads` stalls per k-tile — a third limiter the 15–25% roofline
argument never modelled) or, most importantly, the `warptile_nodbuf`
regression: the lane permutation's bank-conflict cut (4-way → 2-way) is
verified as a pure integer property (`test_gemm_bank_conflict_math`), and
register count, shared memory, and occupancy are all identical to
`tiled_regblock` — so either the extra lane-index arithmetic costs more than
the conflict reduction saves, or the kernel was never actually
shared-memory-bandwidth-bound at this occupancy and cutting conflicts bought
nothing. **Open, and the top thing to check with `ncu`'s stall-reason
breakdown on Explorer or the 5060** (Colab does not expose profiling counters
— confirmed this session, not just assumed from `docs/ENVIRONMENTS.md`).

### Design decisions taken this session

- Recorded every prediction *before* the run (`RESULTS.md` rule 6) rather than
  writing the ladder's expected bands after seeing the numbers, specifically so
  the systematic miss-in-one-direction pattern above would be visible rather
  than rationalized row by row after the fact.
- Kept the Colab numbers in `RESULTS.md` §3d despite the confirmed thermal
  drift, rather than discarding the run — correctness and the occupancy
  three-way comparison are unaffected by clock drift, and the drift itself
  (measured, not assumed) is a legitimate finding. The performance figures are
  explicitly marked Colab-indicative, not authoritative, per the project's
  existing Colab-vs-Explorer convention.

### What's next

An `ncu` pass on Explorer or the RTX 5060 (not Colab — no profiling
permissions) is now the single most valuable next step, specifically to
resolve the `warptile_nodbuf` regression and to check whether `tiled_smem`'s
shortfall is barrier stalls as hypothesized. `RESULTS.md` §5a is scaffolded and
waiting for exactly these runs. `docs/ROADMAP.md`'s Phase-3 exit criterion (a
written explanation of the remaining gap to cuBLAS) cannot be finished
honestly until that pass happens — the current gap is real but its causes are
only partially diagnosed. `LEARNING_LOG.md` end-of-Phase-3 Q&A remains due once
Phase 3 actually closes.

## 2026-08-31 — Session 6: Phase 3d cross-architecture run on Explorer (Tesla V100)

**Hardware: Northeastern Explorer, Tesla V100-SXM2-32GB (sm_70), `gpu-interactive`
partition (`--gres=gpu:v100-sxm2:1`), driver 545.23.08, nvcc 12.3** (downgraded
from the initially-loaded `cuda/12.8.0` module after `nvidia-smi` reported the
driver's max supported CUDA version as 12.3 — mismatched toolkit/driver versions
risk a `CUDA driver version is insufficient` failure at run time rather than at
compile time, so matched them before building). Home directory confirmed shared
across login and compute nodes (the shell's cwd carried over into the `srun`
allocation unchanged), so no re-clone was needed.

Ahead of any GEMM numbers, re-ran `stream_triad`/`fma_peak` on this chip per
this project's own rule (never reuse another machine's denominators): measured
`peak_gb_s = 636.3`, `peak_tflops = 15.601`, ridge point 24.5 FLOP/byte — a
different roofline from the T4's 34.5, recorded in `RESULTS.md` §0.

### The GEMM ladder, same binary, same source, different architecture

`./build/bin/mcke_gemm_bench 4096 --peak-gb-s=636.3 --peak-tflops=15.601`.
Correctness: identical outcome to Colab — every variant OK against the CPU
reference at the awkward shapes, the β≠0 case, and cuBLAS at 4096³. One
diagnostic worth recording so it isn't mistaken for a regression later: the
printed `max_rel_err` against cuBLAS at the benchmark shape was **0 on the T4**
and **~4.15 on the V100**, for every kernel, uniformly. Not a bug — `compare()`
tracks the worst relative error over every element regardless of pass/fail, and
a single near-zero-true output (routine with K=4096 random ±1 data) can show a
large relative error while still passing the absolute floor. The uniformity
across all seven structurally different kernels, and the fact that it was
exactly 0 on the T4, points at cuBLAS choosing a different reduction kernel per
architecture (Volta vs. Turing) rather than at anything in our own code —
bit-exact agreement with cuBLAS on the T4 run was luck, not a guarantee.

**Timing is a much cleaner dataset than Colab's**: the cuBLAS first-vs-last
drift check measured **+0.03%**, against the Colab run's confirmed +11.9%. This
is the first run in Phase 3d with no thermal caveat attached to any number.

**Every architecture-comparable finding, T4 vs. V100 (both %-of-that-chip's-own-
measured-peak):**

| Row | T4 | V100 |
|---|---|---|
| naive_uncoalesced | 1.48% | 2.90% |
| naive | 4.92% | 11.16% |
| tiled_smem | 10.36% | 18.02% |
| tiled_regblock | 40.39% | 75.20% |
| warptile_nodbuf | 39.66% (**−0.73pp**, contradicted the prediction) | 75.06% (−0.10pp, flat/noise) |
| warptile_dbuf | 40.28% (+0.62pp) | 80.96% (**+5.90pp**) |
| warptile_vec4 | 42.58% (+2.30pp) | 81.66% (+0.70pp) |
| cuBLAS | 51.05% / 45.61% (drifting) | 89.60% / 89.57% (stable) |

**The `warptile_nodbuf` regression from the Colab trip did not reproduce.**
Registers, shared memory, tile shape, and occupancy are identical to
`tiled_regblock` on both chips, and the lane permutation's bank-conflict cut is
a verified, architecture-independent integer property
(`test_gemm_bank_conflict_math`). Flat-not-negative on the V100 is evidence
*against* a bug in the kernel and evidence *for* the standing hypothesis that
the extra lane-index arithmetic cost and the conflict saving were roughly
cancelling specifically on Turing — still an open question for `ncu`, but "the
kernel is subtly broken" is no longer a live explanation.

**Occupancy told a genuinely different story per architecture from identical
source code and identical launch configs** — not a discrepancy, an expected
consequence of a different SM shape:
- `naive`/`naive_uncoalesced` (32 regs/thread): a threads-cap-unique bind on the
  T4 (4/4) became a registers/threads **tie** on the V100 (8/8), because 32
  regs/thread is *exactly* this chip's own "regs per thread at 100% occupancy"
  figure (`mcke_device_query` prints this before any kernel runs) — the V100
  has half the register headroom per thread that the T4 has, for the same
  65536-register file, because it has twice the threads/SM to spread across.
- `tiled_smem`: a 100%-occupancy tie on the T4 became a uniquely
  register-bound 50% on the V100 — the V100's 2048 threads/SM means one
  1024-thread block fills only half the SM, so the thread cap stops being
  competitive with the register limit at all.
- `tiled_regblock`: 50% on the T4, 25% on the V100 — same absolute 16 active
  warps, smaller fraction of a bigger SM — and yet 75.2% of peak here versus
  40.4% there. Occupancy *percentage* explained none of that gap; whether 16
  warps is enough to keep an ALU-heavy kernel fed evidently depends on more
  than the fraction of the SM's cap it represents.

**Double buffering is a real, substantial win here (+5.9pp) where it was
marginal on the T4 (+0.6pp)** — consistent with hiding DRAM latency behind
compute mattering more at lower relative occupancy (25% vs. 50%, same absolute
warp count). **`warptile_vec4` buys less here** (+0.7pp vs. +2.3pp) — cutting
global-load instruction count matters most when issue rate is the binding
constraint, and this kernel is evidently not issue-bound on the V100 the way it
may have been on the T4.

**cuBLAS's own efficiency, at rock-stable clocks, is the headline number of
this session: 89.6%**, against the T4's throttled 51.0%/45.6% — the number to
treat as authoritative for "how close does hand-written CUDA get to a vendor
library." `warptile_vec4` sits within 8 percentage points of it (1.10× gap,
versus the T4's 1.20×) with tile sizes never tuned for this architecture.

### Design decisions taken this session

- Kept both architectures' rows in the same `RESULTS.md` §3d table (the
  `Machine` column already exists for this) rather than a separate section per
  chip, since the side-by-side comparison IS the finding — but never computed a
  cross-architecture ratio or "%peak" against the wrong chip's denominator, per
  this file's own "don't mix V100 and A100 in one comparison" rule.
- Did not chase the `warptile_nodbuf` or `tiled_smem` shortfalls with a code
  change this session, per the explicit reasoning from the pre-trip discussion:
  two unconfirmed causes (thermal drift, the regression itself) are still live,
  and coding against an unconfirmed guess risks tuning against noise. `ncu` is
  next.

### `ncu` attempted, blocked — an honest negative result, not a dead end

`ncu` is present (`/shared/EL9/explorer/cuda/12.3.0/bin/ncu`, version
2023.3.0.0, ships inside the `cuda/12.3.0` module — no separate Nsight Compute
module exists on this cluster). Running it against a single trivial launch
(`--set basic --launch-count 1`) failed:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

A driver-level restriction, not a job-configuration mistake — Explorer's own
documentation (`HPC docs/source/`) never mentions Nsight Compute or this
permission model at all, only Nsight Systems, so this was genuinely unverified
until tried rather than something the docs should have warned about. Fixing it
needs `rchelp@northeastern.edu` or the ServiceNow ticket link from Explorer's
own login banner, to either grant the account counter access or have RC set
`NVreg_RestrictProfilingToAdminUsers=0` cluster-wide.

**Decision (logged in `DECISIONS.md`): hold on `ncu`, write up what two
architectures' worth of correctness/occupancy/timing data already supports,
rather than block Phase 3's exit criterion on a ticket with unknown turnaround.**
`RESULTS.md` §5a now records the block explicitly rather than sitting as an
unexplained empty table, and §5b is a new section: the Phase-3 exit writeup
("remaining gap to cuBLAS") done honestly from the evidence in hand — most of
the story is confirmed and cross-validated across the T4 and V100 (the
attribution order, the `warptile_nodbuf` non-bug conclusion, the trend in
dbuf-vs-vec4 importance), with the specific *mechanism* behind that trend and
the exact size of an L2-swizzle's contribution left as open questions rather
than guesses.

### RC ticket filed, and Phase 3 is closed

An email was sent to `rchelp@northeastern.edu` requesting GPU performance-counter
access on the `gpu-interactive` partition (account `wankhare.s`), citing the
exact `ERR_NVGPUCTRPERM` error, job details (V100 node `d1007`,
`--gres=gpu:v100-sxm2:1`, `cuda/12.3.0`, `ncu` 2023.3.0.0), and asking for
either per-account counter access or `NVreg_RestrictProfilingToAdminUsers=0`
cluster-wide, per RC's standard practice. Open, no reply yet — not blocking
further work per the decision above.

**Phase 3 is closed** on that basis: `RESULTS.md` §3a–§3d are filled with real
measured numbers across three sessions and two GPU architectures; every
speedup is attributed to one change (two disclosed multi-cause exceptions,
stated rather than hidden); §5b gives the honest current state of "the
remaining gap to cuBLAS" exit criterion — mostly explained, with the exact
stall-reason mechanism and the L2-swizzle's specific contribution left open
pending `ncu`. `CLAUDE.md` §8 updated to reflect Phases 0–3 complete (previously
five phases stale, still reading "Phase 0 complete"). `LEARNING_LOG.md`
end-of-Phase-3 Q&A remains available whenever asked for, per the owner's
standing convention — not written unprompted.

### What's next

**Phase 4: the computation graph engine and async scheduling**
(`docs/ROADMAP.md`) — `src/graph/graph.cpp` (Kahn sort, levels, live ranges),
`src/graph/executor.cpp` (three schedule policies, event insertion, the
liveness-based memory planner), the four `Op` subclasses wired to Phase 3's
kernels, and `bench/graph_bench.cpp`. Design work (topology, scheduling
policy semantics, the memory planner) is Mac-side and host-testable; the
overlap numbers and `nsys` timelines need Explorer. The numerics gate
(`kLevelParallel`/`kChainGreedy` must be bit-identical to `kSequential`, any
difference is a race not rounding) should be automated from the start, not
added after a bug is found.

## 2026-09-07 — Session 7: Phase 4 numerics-gate failure, two real bugs, closed on Colab T4

### What was built / changed

- `CMakeLists.txt`: fixed two related link-order bugs surfaced by
  `mcke_test_graph_host` (then `mcke_graph_bench`) failing to link on a fresh
  Colab CUDA build: `undefined reference` to kernel launchers. Root cause 1
  (test target only linked `mcke_core`, not the `mcke` umbrella — Itanium ABI
  emits a class's vtable in the TU with its first out-of-line virtual method,
  so constructing an `Op` subclass in a host test pulls in that kernel's `.o`
  at link time even though the virtual `launch()` is never called). Root
  cause 2, found when the fix reappeared on `mcke_graph_bench` (which already
  linked `mcke`): the `mcke` INTERFACE library only declared
  `mcke_kernels PUBLIC mcke_core` (for header/define propagation), so CMake
  was free to place `mcke_kernels` before `mcke_core` on the link line — the
  opposite of what `mcke_core`'s `src/graph/ops_*.cpp` actually needs. Fixed
  with `$<LINK_GROUP:RESCAN,mcke_core,mcke_kernels>` (CMake ≥ 3.24), which
  wraps both in a `--start-group/--end-group` idiom so link order stops
  mattering, rather than reordering by hand (a real fix, not a workaround).
- `include/mcke/graph/executor.hpp` / `src/graph/executor.cpp`: two real bugs
  found while diagnosing a genuine numerics-gate failure on `fanout4x4`
  (Colab T4): `sequential x alloc_per_tensor differs from golden at tensor 5
  word 0: golden -0.168422 vs -0`.
  1. **Use-after-free in `GraphExecutor::set_input()`.** The numerics gate's
     replay loop calls `set_input(id, cache[id].data(), cache[id].size())` —
     so `host_data` routinely points *into* `input_cache_`'s own buffer for
     that tensor. The old code updated the cache first
     (`kv.second = std::move(copy)`), which frees the old backing buffer —
     the exact buffer `host_data` still points into — then read through that
     now-dangling pointer to do the device copy. The "wrong" value read back
     wasn't garbage, it was real (stale/reused) heap content, which is why it
     looked like a plausible logic bug rather than an obvious crash. Fixed by
     doing the device copy *before* touching the cache.
  2. **Mismatched-index comparison**, found immediately after fixing bug 1 and
     re-testing (same host-only repro, different failure). The new
     "check every bound tensor, report every mismatch" diagnostic compared
     tensors by raw loop position `k`, but `kAllocPerTensor` enumerates *every*
     tensor with a buffer while any reuse policy only captures the *declared
     outputs* — different lists, different lengths, different order. Fixed by
     mapping both sides through `TensorId` (`golden_by_id`, `tensors_to_check`)
     instead of position. Added `GraphExecutor::any_bound_tensor()` (reads any
     bound tensor, not just declared outputs; documented as unsafe to expose
     publicly, used only by this diagnostic under `kAllocPerTensor`).
  - Removed a stray uncommitted backup file, `src/graph/executor.cpp.bak`
    (the pre-fix, still-buggy version of the file, left over from live
    debugging) — not tracked, not needed once the real fix was committed.

### Debugging technique that mattered

`std::printf` (stdout) is fully buffered when piped, `std::fprintf(stderr,
...)` is not; combining `2>&1` in one pipe scrambled the apparent order of
interleaved debug prints across `plan()`/`set_input()`/the replay loop badly
enough to suggest a race that wasn't there. Capturing stdout and stderr to
*separate* files (`>out.txt 2>err.txt`) immediately clarified the true
sequence and was what actually exposed the use-after-free.

### Verification

- Host-only (`MCKE_WITH_CUDA=0`, no GPU): `test_numerics_gate_on_host` went
  FAIL (1/9 configs) → FAIL (2/9 configs, the second bug) → **PASS (9/9
  configs, 40,960 elements)**. Full host suite: 86,809 checks, 0 failures.
- Real hardware, Colab Tesla T4, driver 580.82.07, after `git pull` to
  `4e60fe5` and incremental rebuild (~20s): `mcke_test_graph_host` — 86,798
  checks, 0 failures. `mcke_graph_bench --streams=4` — all five gated graphs
  (`fanout4x4`, the originally-failing one; `diamond_starved`;
  `transformer_block`; `diamond_gemm_2048`; `chain16`) numerics gate **PASS**,
  9 configs × 20 repeats each. Full numbers in `RESULTS.md` §4. Wave-sweep
  section also completed and matches the predicted shape (speedup falls
  monotonically, crosses ~1.5× near one full SM wave). Notebook disconnected
  after (idle-timed-out on its own while unattended, not explicitly clicked —
  confirmed "Not connected to runtime" / 0 active sessions on return).

### Design decisions taken (and alternatives rejected)

- `LINK_GROUP:RESCAN` over manually reordering `target_link_libraries` calls:
  the real dependency is bidirectional (core calls kernel launchers; kernels
  target needs core's headers/defines), so any single fixed order is fragile
  against the next new `Op`. The group wrapper is the property that's
  actually true, not just a link line that happens to work today.
- Fixed the use-after-free by reordering (copy-then-cache) rather than by
  deep-copying `host_data` defensively at the top of `set_input()`: the
  caller's contract for a `copy_from_host`-shaped function is that the
  pointer is valid for the duration of the call, same as every other
  `copy_from_host` call site — the bug was this function violating its own
  contract internally, not the contract being insufficient.

### What was learned — including things that turned out to be wrong

- Initial hypotheses ruled out in order before finding the real bug: kernel
  grid-stride coverage gaps (`bias_act.cu`'s `grid_2d()` guarantees ≥1 block
  each axis — not it), uninitialized shared memory (kernel uses zero shared
  memory — not it), poisoning clobbering live inputs (a separate bug, already
  fixed in a prior session — not it), and a real cross-stream race (already
  proven impossible by the host-side happens-before checker — not it). The
  actual bug was in the *test harness's own caching code*, not in graph
  execution, scheduling, or any kernel — worth remembering that a "numerics
  gate" bug can be a bug in the gate itself.
- Colab's UI can show stale cached cell output from a previous runtime after
  a silent reset; only real markers (`%cd` failing with `[Errno 2]`, or the
  Resources panel's "Not connected to runtime" / active-session count) are
  trustworthy state, not what's displayed.

### What's next

- ~~`diamond_gemm_2048` and `chain16`'s exact per-policy numbers~~ — resolved
  same day: found intact in the notebook's own cached cell output (no rerun
  needed) and transcribed verbatim into `RESULTS.md` §4.
- `reports/nsys_phase4.nsys-rep` exists on a since-disconnected Colab
  instance's ephemeral disk, generated by profiling `fanout4x4` against the
  **pre-fix** binary (explicitly flagged `*** NON-COMPLIANT WITH RESULTS.md
  RULE 3 ***`, profiling-only, not for `RESULTS.md`) — so even if it could be
  recovered it wouldn't be usable as-is. A clean Nsight Systems timeline on
  the fixed binary, meeting the ≥5 warmup/≥20 timed rule, would need a fresh
  `nsys profile` run and a real download step (`files.download(...)` or Drive)
  — not yet done, not urgent since it's a deep-dive/visualization aid, not a
  correctness or timing question (both already answered by the gate + bench
  numbers already in hand).
- Phase 4 exit write-up (design tradeoffs already explained live to the
  owner during earlier sessions: SSA-DAG vs. flat list, event vs.
  stream-sync, the three schedule policies, liveness-based reuse) — not yet
  consolidated into a `RESULTS.md` narrative section the way Phase 3 got §5b.

## 2026-09-10 — Session 8: wave-sweep anomaly investigated, not just reworded

### What triggered this session

A review of `RESULTS.md` §4 flagged two problems with the wave-sweep table
recorded in Session 7: the prose claimed "falls monotonically" but the data
had a dip at N=512 and a peak of 2.00× at N=1024; and 2.00× is structurally
impossible for `diamond_gemm(n)` under the obvious model (two independent
GEMMs B, C feeding one dependent GEMM D — sequential = 3t, best case
`max(t_B,t_C)+t_D = 2t`, a 1.5× ceiling if per-kernel time `t` is constant).
The instruction was explicit: investigate which of three hypotheses explains
it, do not reword the prose to fit the data.

### What was built

`bench/graph_bench.cpp`:
- `--gemm-n=N` flag plus two new graphs, `diamond_gemm_custom`/
  `single_gemm_custom`, reachable through a file-scope `g_gemm_n` (`run_graph`
  takes a plain function pointer, no captures) — run the pinned diamond or an
  isolated single GEMM at an arbitrary N instead of only N=2048.
- `--profile`'s output now prints each node's own median/min time per policy
  (`ex.node_timings()`), not just the graph-level concurrency factor — the
  direct test for "did concurrency change a kernel's OWN time" vs. "did the
  graph total just get faster for some other reason."
- The wave-sweep table now measures `blocks/SM` itself
  (`cudaFuncGetAttributes` + `kernels::occupancy_blocks_per_sm` on this
  build's actual `tiled_regblock` kernel) instead of asserting the Phase 3d
  prose figure, prints `waves` per N directly, and adds N=768/1536 between
  the reported dip and peak.

Doc fixes made while in these files: `CLAUDE.md` claimed 86,809 checks passing
in "`test_host_core` + `test_graph_host` combined" — 86,809 is
`test_graph_host` ALONE; combined with `test_host_core`'s 58,856 it's
145,665. `RESULTS.md` §4 referenced `kReuseByLiveness`, which no longer
exists (renamed to `kReuseHappensBefore` when the five memory policies
landed).

### The investigation, on Colab T4 (fresh clone + rebuild at commit `d4e6826`)

**Per-node timing at N=1024** (`--gemm-n=1024 --profile --only=diamond_gemm_custom`):
B_gemm/C_gemm/D_gemm each take ~0.89 ms (min) under `kSequential`, dropping to
~0.75 ms under `kChainGreedy` — a genuine ~16% per-kernel speedup.
**Mechanism:** one GEMM at N=1024 launches 64 blocks; this T4 measures 2
blocks/SM × 40 SMs = 80 blocks/full wave, so 64 blocks alone (0.8 waves)
can't fill the machine — running two together (128 blocks) gives the
scheduler more independent work to hide latency behind, so the per-kernel
time itself drops. Real, but not enough alone to explain a reported 2.00×.

**Isolated single-GEMM cross-check** (`--gemm-n=1024 --only=single_gemm_custom`):
one GEMM alone at N=1024, sequential: 0.927 ms. ×3 = 2.78 ms, matching this
session's `diamond_gemm_custom` sequential number (2.69 ms) to within 3%.
**Hypothesis (b) is refuted for this session's data** — the diamond's own
sequential number is not an anomaly. But the ORIGINAL 2026-09-07 run's
sequential number at N=1024 was 3.704 ms, ~30-37% higher than every
measurement taken this session — that specific historical run's sequential
number really was the outlier.

**Reproducibility — four runs, two different Colab VM instances** (one set
of three back-to-back runs, then the notebook was disconnected mid-session
per the owner's credit-conscious workflow, and a fourth run happened after a
completely fresh reconnect: new clone, new build, new VM). Every N except
768 and 1024 is stable to ~2% across all four runs (e.g. N=256: 1.42–1.49×,
N=1536: 1.01–1.03×, N=4096: 0.99× on every run). N=768 (0.45 waves) ranges
1.06×–1.87×; N=1024 (0.80 waves) ranges 1.05×–1.66× (plus the original run's
2.00×) — genuinely unstable, not measurement error. **Localized mechanism:**
at 768/1024 the two concurrent GEMMs' combined block count (72/128) is close
enough to the 80-block full-wave capacity that which kernel's blocks land in
the first available SM slots is sensitive to launch-order jitter, changing
how much real overlap happens run to run. At N=512 there's enough headroom
that residency order doesn't matter (32 combined blocks); at N≥1536 each
GEMM alone already dominates a wave, so the marginal overlap is small and
stable regardless of order.

**Conclusion, written into `RESULTS.md` §4 verbatim (not smoothed over):**
"falls monotonically" was wrong — the true shape is a smooth decay from
~1.45× (near-zero waves) to ~1.0× (≥1.8 waves) with a genuine, reproducible
**noisy bump** localized to 0.45–0.8 waves, whose value should be reported as
a range, not a single number. Hypothesis (a) is real but small; hypothesis
(b) explains the *original* run's specific number; the dominant honest
explanation is (c), a real scheduling-jitter sensitivity in a narrow
occupancy band — stated as such rather than picked as "the" answer to make
the writeup tidier.

### Debugging note: don't trust a Colab cell's cached bracket/checkmark

Mid-session, the notebook was reconnected after a disconnect and several
cells appeared to show fresh green checkmarks and output — but `ls /content/`
showed no `mcke` directory, and a cell's own hover tooltip read "cell has not
been executed in this session" alongside a checkmark from 52 minutes earlier.
Colab's cell UI can display a stale previous execution's checkmark/output
indefinitely until the cell is actually re-run in the current runtime;
the reliable signals were the tooltip's own "started at HH:MM (N minutes
ago)" text and the bottom status bar's live "Executing (Ns)" counter, not the
cell's bracket number or checkmark color.

### Verification

- Full 5-graph suite (`mcke_graph_bench --streams=4`, no filter) re-run
  clean on the fresh VM: all five numerics gates PASS (same 9×20 matrix as
  Session 7), confirming the bug fixes from Session 7 hold on a completely
  independent rebuild, not just the runtime they were originally verified on.
  `RESULTS.md` §4's main table now reflects this fresh run's exact numbers
  (previous numbers were sourced from a stale cached cell and are superseded,
  per the owner's explicit "don't use old results, run again").

### What's next

- `reports/nsys_phase4.nsys-rep` still not regenerated on the fixed binary
  (see Session 7) — still a deep-dive/visualization nice-to-have, not
  blocking.
- Phase 4 exit write-up (Phase 3 got one in §5b) — the wave-sweep anomaly
  section above is most of the substance Phase 4's write-up would need;
  remaining is consolidating the design-tradeoff narrative already discussed
  live with the owner (SSA-DAG vs flat list, event vs stream-sync, the three
  schedule policies, liveness-based reuse).

---

## 2026-09-10 — Session 9: Phase 4 exit write-up, and two corrections to my own prose

**Environment:** macOS host-only (no GPU). No benchmarks run; this session was
documentation and review of numbers already measured in Sessions 7–8.

### What was built

- **`RESULTS.md` §5c — the Phase 4 exit write-up**, matching §5b's shape for
  Phase 3. Covers: speedup per policy per graph and why four of five graphs
  measured ~1.00×; the corrected event-count claim; the memory-vs-parallelism
  tension; the design decisions with their rejected alternatives (SSA-DAG vs
  declared edge list, events vs stream-synchronize, plan-time arena vs runtime
  allocate/deallocate, three schedule policies); the topological-liveness
  correctness result; and the open items.
- **`LEARNING_LOG.md` — Phase 4 end-of-phase Q&A** (5 questions). Noted in it
  that Phases 1–3 still have none.
- **`CLAUDE.md` §8** updated: the exit write-up is no longer outstanding, so the
  only item left before Phase 4 closes is the `nsys` regeneration.

### What was learned — two errors in my own earlier prose, both found by re-reading against the data

1. **The wave-sweep bump is in 3 of 4 runs, not all four.** §4 claimed it was
   "visible across all four runs, even though the exact value varies." The table
   does not support that. Interpolating the smooth decay between 0.20 waves
   (1.35×) and 1.80 waves (1.02×) predicts ~1.22× at 0.45 waves and ~1.10× at
   0.80; run 1's 1.13× / 1.09× sit at or just below that line, i.e. **run 1 is
   the monotone curve the original prediction described** — no bump at all. The
   bump appears in runs 2 and 3 (at N=1024) and run 4 (at N=768), always at
   exactly one of the two adjacent points, never both.

   This is a *weaker* claim about universality and a **stronger** one about
   mechanism: a bump that migrates between adjacent sample points and sometimes
   fails to appear is much better evidence for launch-order/residency
   sensitivity than one reproducing at a fixed N would be. A fixed bump would
   point at something structural about that block count; a mobile, intermittent
   one is what jitter looks like.

2. **`diamond_starved`'s ceiling estimate used the wrong row of §3a.**
   `bench/graph_bench.cpp`'s banner computed ~1.13× from 235.4/208.7 — but
   208.7 GB/s is §3a's **vw1** starved figure, and `BiasActOp` is constructed
   with `vector_width = 0`, meaning "pick the widest legal width", which at 4096
   columns resolves to **vw4**. The vw4 starved figure is 224.6 GB/s, so the
   correct estimate is **235.4/224.6 = ~1.048×**. The measured 1.01× therefore
   sits within 4% of the ceiling rather than 12% below a looser bound.

   Lesson worth keeping: using a kernel's *measured* bandwidth as a denominator
   means using the measurement for **the configuration that actually runs**. The
   prediction was right in kind and loose by ~2.6× in the headroom it claimed.
   Corrected in both `RESULTS.md` §5c and the bench banner, with the wrong
   version named.

The framing that made this recoverable: D3's prediction was deliberately
recorded as a **ceiling estimate and not a floor**, because 235.4/224.6 assumes
two concurrent kernels share DRAM cleanly and additively — the very assumption
under test. Had it been written as a lower bound, the correct measured result
(1.01×) would have read as a failure.

### Design decisions taken

None new — this session wrote up decisions already taken and discussed live in
Sessions 5–8. The one judgement call: the future-dated Session 8 entry
(2026-09-10, written while the clock read 2026-09-09) was **left alone** rather
than "fixed" to 09-09, since the date has now rolled over to match and changing
it would substitute one guess for another about when a Colab session spanning
timezones actually ran.

### Benchmarks run

None. Both host suites re-verified unchanged on this machine:
`test_host_core` 58,856 checks / 0 failures, `test_graph_host` 86,809 / 0
failures (145,665 combined). `scripts/typecheck_cuda.sh` clean on all TUs
including the NVTX pass.

### What's next

1. **Regenerate `reports/nsys_phase4.nsys-rep` on the fixed binary** — the only
   thing still outstanding for Phase 4. Colab or Explorer; build with
   `-DMCKE_USE_NVTX=ON`, then
   `nsys profile --trace=cuda,nvtx --only=fanout4x4 --iters=5 --warmup=2`.
   Delete the stale pre-fix report so it cannot be mistaken for current.
   `fanout4x4` is the graph to profile: it is the only one with real overlap to
   show (1.94×).
2. Optionally, Phase 1–3 end-of-phase Q&A entries in `LEARNING_LOG.md`, which
   were skipped at the time.
3. Then Phase 5 (`docs/ROADMAP.md`) — Google Benchmark integration and the
   profiling/telemetry deliverable.

## 2026-09-11 — Session 10: the nsys timeline, Phase 4's last exit criterion

**Environment:** Colab Tesla T4, driver 580.82.07, fresh clone + rebuild at
commit `13ef269` (`-DMCKE_USE_NVTX=ON`).

### What was built / done

- Regenerated the Nsight Systems timeline the Session 7 attempt failed to
  preserve, this time avoiding all three of that attempt's mistakes: built
  from current HEAD (post-fix, not the use-after-free build), accepted the
  `*** NON-COMPLIANT WITH RESULTS.md RULE 3 ***` banner as correct for a
  profiling-only run (and kept its timings out of `RESULTS.md`/`§4` entirely),
  and downloaded/committed the artifact in the same working session instead of
  leaving it on ephemeral `/content`.
- `nsys` was not on `PATH`; located at
  `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys` (bundled
  with the installed `nsight-compute` package — no separate `nsight-systems`
  package exists on this image, contrary to what the hardcoded Session 7 path
  might have suggested was a guess; it was actually correct).
  `nsys profile --trace=cuda,nvtx,osrt --stats=true -o
  reports/nsys_phase4_fanout4x4 ./build/bin/mcke_graph_bench
  --only=fanout4x4 --iters=5 --warmup=2` produced
  `reports/nsys_phase4_fanout4x4.nsys-rep` (998 KB) and a `.sqlite` export.
- `reports/` is `.gitignore`d (correctly, for CSV/log noise); force-added just
  this one file (`git add -f`) as a deliberate, narrow exception — the exit
  criterion explicitly wants it committed. Committed and pushed directly from
  the Colab instance (reusing the `GITHUB_TOKEN` auth already in the kernel's
  Python state from the clone cell, via `git -c http.extraHeader=...`), rather
  than relying on `google.colab.files.download()`, which triggered no
  visible error but also produced no evidence of a completed browser download
  in this automated environment — committing directly is the more reliable
  path when driving Colab non-interactively.
- No GUI is available to open the `.nsys-rep` visually here, so the trace was
  queried directly instead: `nsys stats --force-export=true --report
  cuda_gpu_trace --format csv` gives exact per-kernel start/end timestamps and
  stream IDs, which were swept in pandas for cross-stream overlapping
  windows. This is arguably *better* evidence than a screenshot of timeline
  bars, since it's an exact, reproducible number rather than an eyeballed
  visual — 4,404 overlapping kernel pairs, a measured max of 4 concurrently
  resident streams (matching `fanout4x4`'s own reported `streams 4/4`
  exactly), and one worked example: four `bias_act` kernels on streams
  14/15/16/17 spanning a combined wall-clock window of 569,046 ns while their
  own durations sum to 1,969,885 ns — a **3.46× local concurrency factor**.
  Full writeup and the table are in `RESULTS.md` §5c.
- Fixed the two stale docs named in the assignment: `docs/PROFILING.md`'s
  nsys example command used a `--policy=chain_greedy` flag that does not
  exist (graph_bench runs all three policies per graph; `--only=` selects the
  graph) — replaced with the actual working command. `docs/ROADMAP.md`
  assigned "the overlap numbers and nsys timelines" to Explorer, which was
  simply wrong for `nsys` (it needs no GPU performance counters, unlike `ncu`)
  and was the actual reason this task looked blocked on the `ERR_NVGPUCTRPERM`
  ticket for two sessions running — split the line: `nsys` → Colab (the
  machine `RESULTS.md` §4's numbers already come from), `ncu` → Explorer,
  pending the RC ticket.

### What was learned

- **Colab's "Run all" is one click away from a real accident.** Reaching for
  the Resources panel toggle at the end of this session, a misclick landed on
  "Run all" instead, and it began re-executing the entire notebook top to
  bottom — including the clone, build, and git-commit-and-push cells already
  used earlier. Interrupted via Runtime → Interrupt execution (took two tries;
  the first interrupt let the in-flight cell finish before stopping). Checked
  local git history afterward: no duplicate or corrupted commits, because
  every cell in this notebook happens to be idempotent by construction (clone
  no-ops if the dir exists, `git add -f` on an unchanged file produces "nothing
  to commit", `git push` on a fresh remote says "Everything up-to-date").
  That idempotence was luck from how the notebook was built, not a designed
  safety property — worth being more careful about which cell has focus before
  invoking a whole-notebook command.
- **Cell UI state (bracket number, checkmark, tooltip text) can lag the
  runtime's real state by an entire disconnect/reconnect cycle.** Confirmed
  again this session: a cell showing a green check and old output can still
  say, on hover, "cell has not been executed in this session." The only
  trustworthy live signals are the tooltip's own "started at HH:MM (N minutes
  ago)" and the bottom status bar's "Executing (Ns)" counter — both were
  needed repeatedly this session to tell a genuinely-running cell apart from
  a stale display.
- Programmatically driving Colab's Monaco-based cell editor by clicking
  coordinates is fragile in one specific way: clicking anywhere on an EMPTY
  cell's "Start coding or **generate** with AI" placeholder routes typed text
  into a Gemini side-panel chat box instead of the cell, silently. Clicking on
  non-empty text is safe; for empty cells, either click clear of the
  "generate" hyperlink specifically, or (far more reliable) set the cell's
  content directly via `monaco.editor.getModels()[i].setValue(...)` through
  the browser's JS console — sidesteps ghost-autocomplete-suggestion overlays
  and placeholder-link hijacking entirely. Used this for the rest of the
  session once discovered.

### Verification

- `mcke_test_graph_host` run fresh on this same build before profiling:
  86,798 checks, 0 failures (must pass before a profiling run counts, per the
  task's own instructions).
- `reports/nsys_phase4_fanout4x4.nsys-rep` confirmed present in the pushed
  commit (`9e38af0`) and pulled successfully back to the Mac working copy.

### What's next

**Phase 4 is closed.** All four `docs/ROADMAP.md` exit criteria are met with
measured numbers on real hardware: speedup per policy per graph, event counts,
peak memory with/without liveness reuse, and now the `nsys` timeline
confirming actual overlap on the one graph (`fanout4x4`) the wall-clock numbers
show (1.94×).

1. Optionally, Phase 1–3 end-of-phase Q&A entries in `LEARNING_LOG.md`, which
   were skipped at the time.
2. Then Phase 5 (`docs/ROADMAP.md`) — Google Benchmark integration and the
   profiling/telemetry deliverable.

## 2026-09-11 — Session 11: fixed a wrong mechanism in the Session 10 write-up

**Environment:** macOS host-only, no GPU. Documentation-only fix — no code
changes except one clarifying comment; nothing was re-run.

### What was wrong

`RESULTS.md` §5c's nsys-timeline subsection explained the gap between the
measured 3.46× local concurrency factor and `fanout4x4`'s overall 1.94×
speedup as Amdahl's-law amortization against "the initial input distribution
and the final join/reduce that must run after all four branches complete."
**That mechanism doesn't exist.** `bench/graph_bench.cpp`'s `fanout()` builds
one input feeding four independent chains, each ending in its own
`mark_output()` — four separate graph outputs, no join node, nothing that
runs after them. `set_input()`'s one H2D copy happens once before the warmup
loop, not inside the timed region. The actual per-iteration serial cost is
microseconds of fork/join events against a 2.2 ms run.

### What was found instead, and how it was checked

The real mechanism is **per-kernel slowdown under DRAM contention**, computed
from numbers already recorded in the same section and in §3a — nothing new
was measured:

- Sequential per-node time: 4.222 ms / 16 nodes = 263.9 µs. Contended
  per-node time (mean of the four durations already in the nsys table):
  492.5 µs. **Per-node slowdown under contention: 1.87×.**
- 4 depth-levels × 569.0 µs/level (one "wave" of 4 concurrent kernels per
  level) predicts 2.276 ms; measured `chain_greedy` median is 2.175 ms —
  close.
- Ideal parallelism (4×) divided by the measured 1.87× slowdown predicts
  2.14×; measured speedup is 1.94× — close. The 1.87× slowdown, not a serial
  fraction, is what turns 4× of available parallelism into 1.94×.
- This is the exact mechanism §3a already diagnosed for `diamond_starved`:
  `fanout(4, 4, 2048, 2048, 10)` gives 10 blocks/branch, so four concurrent
  branches occupy 40 blocks total — the same block count §3a measured this
  kernel hitting 224.6 GB/s / 95.4% of the 235.4 GB/s DRAM ceiling at (a
  different shape, 8192×4096, so that exact GB/s figure doesn't transfer
  directly, only the mechanism does). Recomputing §3a's own ideal-bytes
  formula directly for `fanout4x4`'s shape and the measured 569,046 ns
  window gives an achieved 235.9 GB/s — matching the ceiling to within 0.2%,
  a shape-matched, directly-computed confirmation stronger than citing the
  §3a row by analogy. `diamond_starved`'s lesson — "SMs are idle" is not "the
  machine is idle" — reappears here quantitatively, on the one graph where
  overlap paid off.
- Secondary correction: 3.46× and 1.94× were being read as directly
  comparable, and they aren't. The concurrency factor's numerator is
  *contended* durations (inflated by the 1.87× slowdown itself); the
  speedup's numerator is *uncontended* sequential time. 3.46×/4 = 87%
  correctly measures **time-packing** (how tightly the four launches land in
  the wall-clock window, limited by ~65 µs of launch skew visible in the
  table's own start timestamps) — not 87% of useful parallelism. Recorded
  both in `RESULTS.md` (next to the 3.46× figure) and at the metric's actual
  definition in `bench/graph_bench.cpp` (the `r.concurrency` computation),
  so a future reader hits the caveat at the source, not just in the prose.

All arithmetic above was checked against the repo's own numbers before
writing anything (the §4 fanout4x4 row, the nsys table's four durations and
569,046 ns span, §3a's starved-bandwidth rows and ideal-bytes formula) rather
than taken on faith from the request that flagged the error.

### What was learned

A plausible, tidy-sounding explanation (Amdahl / join-node overhead) can be
wrong in a way that only shows up when someone actually reads the graph
builder it claims to describe. The corrected version is less tidy (it needed
three separate numbers pinned down, not one clean ratio) but ties back to a
result already in the document (§3a's DRAM-saturation finding) instead of
inventing a new one — which is itself a sign it's more likely right.

### What's next

Same as end of Session 10: Phase 4 remains closed. Optionally, Phase 1–3
end-of-phase Q&A entries in `LEARNING_LOG.md` (skipped at the time). Then
Phase 5 (`docs/ROADMAP.md`) — Google Benchmark integration and the
profiling/telemetry deliverable.

## 2026-09-16 — Session 12: Phase 5 opens — Google Benchmark build wiring (stage 5a)

**Environment:** MacBook Air (Apple Silicon), host-only. AppleClang 21.0.0,
**CMake 4.4.3 — newly installed via Homebrew this session** (see below). No GPU
work; nothing in `RESULTS.md` changed.

### Environment change: CMake is now installed on the Mac

Worth its own heading because it invalidates a standing assumption. Until today
this machine had no `cmake` at all, and `CLAUDE.md` §4 documented a one-line
`clang++` fallback as the way to run host tests. Phase 5 stage 5a is *entirely*
CMake work, and its central property — "an offline machine with no Google
Benchmark configures cleanly and just skips the `_gb` targets" — is a
**Mac-specific** property that Colab structurally cannot test, because Colab
always has network. So the choice was install CMake or ship the tri-state
design unverified.

`CLAUDE.md` §4 is updated to say so. The `clang++` fallback is **kept**, not
deleted — it is still the right answer for a fresh container or a login node
with no CMake module — and was re-run verbatim this session to confirm it has
not rotted: compiles warning-free, **58,856 checks, 0 failures**. Two
limitations of it are now written down that were not before: it builds
`test_host_core` only (~40% of the host suite; `test_graph_host`'s 86,809
checks are not in the one-liner), and it must be run from the repo root.

### What was built

`CMakeLists.txt` only. Two blocks:

1. **`MCKE_GOOGLE_BENCHMARK` — a tri-state cache variable, not an `option()`.**
   `AUTO` (default) / `ON` / `OFF`. A bool cannot express what three machines
   need. `AUTO` uses Google Benchmark if it is *already* available and otherwise
   **silently skips the `_gb` targets without ever touching the network**.
2. **The acquisition block** — `find_package(benchmark QUIET CONFIG)` first,
   FetchContent only on explicit `ON` or when `FETCHCONTENT_SOURCE_DIR_BENCHMARK`
   points at a local checkout.

### What was learned

**`FetchContent`'s `FIND_PACKAGE_ARGS` is the wrong tool here, despite being
exactly what the roadmap bullet implies.** It exists at our 3.24 floor, so the
obvious move is `FetchContent_Declare(benchmark ... FIND_PACKAGE_ARGS)`. But
`FetchContent_MakeAvailable` **falls through to the clone** when `find_package`
misses, and on a machine with no route to github.com that is a configure-time
`FATAL_ERROR` for the *whole project* — not just the bench targets. Two of our
five environments have that property routinely: the MacBook offline, and
Explorer's compute nodes. The zero-dependency "configures on a laptop with no
network" property has held since Phase 0 and is worth more than the convenience,
so absence degrades to *skip*, not to *fail*.

**Every claim about Google Benchmark's CMake was verified against the actual
v1.9.5 source rather than assumed.** This mattered — the first draft of the
comment block got the GTest gating structure wrong. Verified facts:

- `BENCHMARK_ENABLE_TESTING` defaults **ON** and is the **outer** gate
  (`CMakeLists.txt:348`). `BENCHMARK_ENABLE_GTEST_TESTS` (also ON) is *nested
  inside* it at `:350`, and `find_package(GTest CONFIG REQUIRED)` at `:356` is
  nested inside that. The original comment described GTEST_TESTS as the
  independently dangerous one; it isn't, it's unreachable with TESTING off.
  Corrected in the file.
- `BENCHMARK_ENABLE_WERROR` defaults **ON** — the genuinely dangerous default.
  Google Benchmark builds *its own* sources with `-Werror`, so a future compiler
  emitting one new warning inside a dependency we don't control breaks *our*
  build. Forced OFF.
- `CMakeLists.txt:144` does an **unguarded `set(CMAKE_CXX_STANDARD 17)`** and GB
  declares **no `target_compile_features` anywhere**. Two consequences, and they
  point opposite ways: (good) `benchmark::benchmark` imposes nothing on our
  C++20; (bad) `libbenchmark.a` compiles as C++17 while *our* TUs compile
  `benchmark.h` as C++20 — the header's inline functions and templates compiled
  twice under two standards, an **ODR violation with no diagnostic**.
  `-DCMAKE_CXX_STANDARD=20` on the command line does *not* fix it (GB's
  unguarded `set()` clobbers it); `set_property(TARGET benchmark PROPERTY
  CXX_STANDARD 20)` afterwards does, and was **confirmed by reading the actual
  compile flags**: `-std=c++20`.
- The `SYSTEM` keyword on `FetchContent_Declare` is CMake **3.25**; our floor is
  3.24. So the hand-rolled `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` is *required*,
  not defensive — without it our `-Wall -Wextra -Wpedantic` apply to GB's headers.

**`find build -name '*gtest*'` is the wrong acceptance test**, and it failed on
the first try for the wrong reason. GB's clone *always* contains
`test/*_gtest.cc` source files; their presence proves nothing. The real checks
are that no `googletest-*` directory appears in `_deps` and that zero gtest
targets exist — both verified.

### Verification — five configurations, all passing

| # | Configuration | Result |
|---|---|---|
| 1 | `AUTO`, no GB, **network poisoned**, **fresh** build dir | configures clean, `_gb` skipped, **no `_deps` created** |
| 2 | `MCKE_GOOGLE_BENCHMARK=ON`, network, fresh dir | fetched at pinned SHA `192ef100`; **no googletest in `_deps`**, 0 gtest targets; `libbenchmark.a` built `-std=c++20` |
| 3a | `AUTO` + `FETCHCONTENT_SOURCE_DIR_BENCHMARK`, **network poisoned** | used local checkout (the Explorer recipe) |
| 3b | `AUTO` + installed package, **network poisoned** | `find_package` found 1.9.5, **no `_deps`** |
| 4 | existing host suite, in-repo `build-host/` | **145,665 checks, 0 failures, 0 warnings** (58,856 + 86,809) |

Config 1 and 3a/3b were run with `HTTPS_PROXY=http://127.0.0.1:1` and
`GIT_TERMINAL_PROMPT=0` so that any network attempt *fails fast* rather than
silently succeeding — asserting the offline property instead of assuming it.
All from **fresh** build directories: once FetchContent populates `_deps` it
will not re-fetch, so an offline configure over a warm build dir passes for the
wrong reason and proves nothing.

### A non-bug worth recording, because it looked like a bug

The first `ctest` run reported **58,817 checks and 1 failure** where the binary
run directly reported **58,856 and 0**. Deterministic, not flaky. Cause:
`test_reference_vectors` locates `tests/data/reference_vectors.txt` by probing
three relative paths (`tests/data/`, `../tests/data/`, `../../tests/data/`), and
the build dir was `/tmp/mcke_cfg1` — outside the repo, so none resolved and 39
reference-vector checks silently vanished. With the documented in-repo
`build-host/`, `../tests/data/` resolves and it passes. So: my artifact, not a
regression, and **not** caused by the CMake change. Now documented in
`CLAUDE.md` §4 rather than left to be rediscovered.

### Design decisions

Logged live to `DECISIONS.md` as **Q7/Q8/Q9** before any code was written.
Q7 (Google Benchmark's role) is the one that shapes the phase: **wrap, don't
replace** — GB owns repetitions and reporting, `Profiler::time_op` keeps owning
the measurement. The reasoning is that GB computes its statistics **over
repetitions, of per-repetition means**, so the per-launch median and min that
`RESULTS.md` rule 3 publishes are averaged away *inside* each repetition before
GB ever sees them. GB is therefore strictly *less* informative about kernel
timing than what we already have. Adopting it as a measurement tool would have
invalidated every §1–§4 number for a downgrade.

Q9 went **against** the recommendation: the owner chose to keep `ncu` on the
"write the script, validate on Colab, leave §5a blocked" path rather than open
the RTX 5060 as a new environment. Recorded as such.

### What's next

Stage 5b — `bench/gb_adapter.hpp` and `bench/alloc_bench_gb.cpp`. The adapter is
the correctness seam: `SetIterationTime` takes **seconds** while
`rt::Event::elapsed_ms` returns **milliseconds**, and forgetting
`->UseManualTime()` silently discards the manual timing and reports host
wall-clock launch latency instead. `alloc_bench_gb` is host-only, so the whole
thing is verifiable on this machine with no GPU — deliberately, so the `* 1e-3`
bug cannot survive to burn a Colab session. Stage 5g is the Colab run.

## 2026-09-22 — Session 13: Phase 5 stage 5b — the Google Benchmark adapter

**Environment:** MacBook Air (Apple Silicon), host-only, AppleClang 21.0.0,
CMake 4.4.3, Google Benchmark v1.9.5 (`192ef100`). No GPU. Nothing in
`RESULTS.md` changed — these are host allocator microbenchmarks, not a
replacement for anything published.

### What was built

- **`bench/gb_adapter.hpp`** (new) — the seam between Google Benchmark's
  iteration loop and CUDA-event timing. Two named modes: `kBurst` (one GB
  iteration = one `Profiler::time_op` burst, identical code path to every
  existing `RESULTS.md` row) and `kPerIter` (GB's native model, one host sync
  per iteration — present *on purpose* as the instrument for stage 5g's
  cross-check, never for a published timing).
- **`bench/alloc_bench_gb.cpp`** (new) — host-only allocator microbenchmarks
  plus the unit-conversion self-check. `bench/alloc_bench.cpp` was **not**
  touched; its stdout is transcribed into §2a/2b/2c.
- **`CMakeLists.txt`** — `mcke_alloc_bench_gb` target, gated on
  `MCKE_HAVE_GOOGLE_BENCHMARK`, linking `mcke_core` (not the `mcke` umbrella)
  and `benchmark::benchmark` (not `benchmark_main`).
- **`scripts/typecheck_cuda.sh`** — Google Benchmark header discovery.

### Benchmarks run — actual numbers (macOS host, 5 repetitions each)

| Allocator | 4 KiB | 1 MiB | 8 MiB |
|---|---|---|---|
| raw (host `malloc`, **not** `cudaMalloc`) | 34.5 ns | 158 ns | 95.2 ns |
| buddy (coarse poll) | 84.6 ns | 47.4 ns | 31.3 ns |
| freelist (coarse poll) | 48.8 ns | 49.1 ns | — |

Policy sweep at 4 KiB: buddy `same_stream_only` 101 ns / `coarse_poll` 84.6 ns
/ `per_free_event` 103 ns; freelist 54.9 / 48.8 / 56.3 ns. All `_cv` under
3.6%, most under 1%.

### What was learned

**Two of those numbers look like they refute Phase 2, and neither does.**

*"raw" beats both pools at 4 KiB.* Only because this is a host-only build: with
`MCKE_WITH_CUDA=0`, `RawDeviceAllocator` never calls `cudaMalloc` — it calls the
system allocator, a fast cached user-space free list. The pools exist to
amortise a ~10–100 µs driver round trip that does not occur here, so this is a
different experiment wearing the same name. Documented at the top of the file so
nobody reads it as a pool-vs-driver result.

*Buddy gets **faster** as blocks get **bigger*** (84.6 → 47.4 → 31.3 ns). This
one is real and is buddy structure showing through: cost tracks **levels
traversed**, not bytes. The slab is 16 MiB = 2²⁴, so a 2^k request costs 24−k
splits down and the same number of merges back: 4 KiB → 12 levels, 1 MiB → 4,
8 MiB → 1. **Tested rather than asserted:** fitting a line through *only* the
two extremes gives `cost ≈ 26.5 ns + 4.85 ns/level`, which predicts the
held-out middle point (1 MiB) at **45.8 ns against a measured 47.4 ns — 3.3%
error**. A bytes-based model cannot even produce that ordering. Freelist is flat
(48.8 vs 49.1 ns), exactly as a segregated size-class design predicts. So the
two allocators' cost *curves differ in shape*, not just height, and cross near
1 MiB — which is a sharper statement than Phase 2c's aggregate comparison made.

**Both silent failure modes were deliberately triggered, not just guarded
against.** This is why `alloc_bench_gb` is host-only: it is the one GB target
that runs without a GPU, so the guards get tested here in a second rather than
on Colab, where a wrong number costs a session and may not look wrong.

- Deleting the `* 1e-3`: the 2 ms self-check reported **2,000,000 µs instead of
  2,000 µs** — exactly 1000×.
- Dropping `->UseManualTime()`: **0.292 µs instead of 2,000 µs** — a ~6,850×
  *understatement*, i.e. it makes a kernel look spectacular rather than broken,
  which is the dangerous direction.

That second experiment turned up **the one detection tell**, now recorded in the
adapter: GB appends `/manual_time` to the reported benchmark *name* when manual
timing is active. `.../iterations:16/manual_time` is wired correctly;
`.../iterations:16` is not. It is the sole visible difference.

**Three bugs of my own, all found by verification rather than by reading.**

1. **`DEFINED` vs non-empty in CMake.** `FetchContent_Declare` itself creates
   `FETCHCONTENT_SOURCE_DIR_<name>` as an **empty cache entry**, so
   `if(DEFINED FETCHCONTENT_SOURCE_DIR_BENCHMARK)` is true forever after any
   configure that reached the fetch path — meaning `AUTO` would silently start
   fetching in a build dir that had once been configured `=ON`. Found because
   the status line printed "from local checkout" with an *empty path*. A bare
   `if(VAR)` is false for the empty string and is the question actually meant.
2. **`set -u` in `typecheck_cuda.sh`.** `"$MCKE_BENCHMARK_INCLUDE"` on an unset
   variable aborts the script. Worse, my first "the skip path works" test passed
   only because I had *set* that variable in the test — the default path was
   broken the whole time. Same shape as the warm-`_deps` trap from stage 5a:
   a test that passes for the wrong reason.
3. **Banner on stdout corrupted the JSON artifact.** `--benchmark_format=json`
   writes to stdout, so the `# mcke_...` header made the document unparseable —
   and it failed at the *consumer*, which is the worst place to find it. Moved
   to stderr. Stage 5f's `regen_results.sh` parses this JSON, so stdout must
   stay machine-clean.

**`typecheck_cuda.sh` caught the new file automatically**, exactly as its own
comment intended ("glob `bench/*.cpp` … so a new bench is covered the moment it
exists"). The fix was to give it Google Benchmark's headers via `-isystem`
(not `-I`: GB's headers are not ours to keep warning-clean, the same reasoning
as the `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` fixup in CMake), with absence
degrading to a **per-file announced SKIP** rather than a failure — mirroring
`MCKE_GOOGLE_BENCHMARK=AUTO`, since with GB absent CMake does not build these
targets either. Announced per file, never silent: a silent skip is how coverage
rots, which is the exact failure the `src/*/*.cpp` glob was widened to fix in
Phase 4.

### Design decisions

**The unit conversion is a named function, not an inline `* 1e-3`.** In a
host-only build `rt::Event::elapsed_ms` returns `0.0f` unconditionally
(`stream.hpp:280`), so the GPU paths cannot exercise the conversion on a machine
with no GPU. Extracting `set_iteration_time_from_ms()` makes it callable with a
*known* value, which is the only reason the 1000× guard is testable here at all.
Rejected alternative: leave it inline and verify on Colab — which defeats the
entire purpose of having a host-only GB target.

**The `Profiler` inside `run_burst` is function-local.** `Profiler` has no
`clear()` and `time_op` appends to `records_` on every call, so a hoisted one
would accumulate (repetitions × variants) duplicate rows and `write_csv` would
emit every repetition as a separate measurement. Function-local needs **zero**
changes to `profiler.hpp`. Commented so nobody "optimises" it by hoisting.

**No `SetBytesProcessed`.** It installs its counter with `kIs1024` (verified,
`benchmark.h:892`), so it prints GiB/s where `KernelRecord::gb_per_s()` divides
by `1e9` — a silent 7.4% disagreement on the number rule 5 exists to pin down.
Explicit `Counter`s with `kIs1000` instead.

**Dropped the `bytes` counter** from the allocator rows: GB aggregates every
counter across repetitions, so a constant renders as `bytes=0` on `_stddev` and
`bytes=0.00%` on `_cv` — which reads like a measurement and is not one.

### Verification

- `mcke_alloc_bench_gb` builds and runs on macOS with **no GPU**, 0 warnings.
- Self-check reads **exactly 2000 µs / 500 µs**; both break-tests confirmed above.
- `--benchmark_format=json` produces a valid document carrying
  `mcke_cmdline`, `mcke_timing`, `mcke_with_cuda`, `mcke_caveat`.
- `scripts/typecheck_cuda.sh`: **all clean** in both states — GB present
  (`ok bench/alloc_bench_gb.cpp`, under `MCKE_WITH_CUDA=1`, so it will compile
  on Colab) and GB absent (announced SKIP).
- Host suite unchanged: **145,665 checks, 0 failures** (58,856 + 86,809).

### What's next

Stage 5c — `bench/reduce_bench_gb.cpp` and `bench/bias_act_bench_gb.cpp`, the
pilot GPU targets. Pattern application of the adapter; Sonnet/medium is enough.
Deliberately *not* `gemm_bench`/`graph_bench`: their bespoke tables (wave sweep,
fork/join counts, `enqueue_us`) *are* the Phase 3/4 deliverables and GB's output
format cannot express them. Stage 5g is the Colab run, where `kBurst` must agree
with `Profiler::time_op` to within noise — if it does not, the adapter is wrong.

## 2026-09-22 — Session 14: Phase 5 stage 5c — pilot GPU targets

**Environment:** MacBook Air (Apple Silicon), host-only. No GPU, so these two
targets are typechecked but not run — real execution is stage 5g on Colab.
Nothing in `RESULTS.md` changes.

### What was built

- **`bench/reduce_bench_gb.cpp`** (new) — GB pilot for row-reduce (Phase 3b),
  both timing modes (`kBurst`/`kPerIter`) at the same two shapes
  `reduce_bench.cpp` uses (`8192×4096` saturated, `64×524288` starved).
- **`bench/bias_act_bench_gb.cpp`** (new) — GB pilot for fused bias+GELU-tanh
  (Phase 3a), both modes at the primary `8192×4096` shape and the L2-resident
  `512×512` control.
- **`CMakeLists.txt`** — `mcke_reduce_bench_gb` / `mcke_bias_act_bench_gb`,
  gated on `MCKE_ENABLE_CUDA AND MCKE_HAVE_GOOGLE_BENCHMARK`, linking the
  `mcke` umbrella (not `mcke_core`) since these call into `mcke_kernels`.

Neither `reduce_bench.cpp` nor `bias_act_bench.cpp` was touched — both new
files are pilots alongside them, not replacements.

### Why these two shapes, deliberately

The 512×512 L2-resident shape and the 64×524288 starved shape are the two
**shortest** kernels anywhere in Phase 3. Short kernels are exactly where
`kPerIter`'s per-iteration host sync (see `gb_adapter.hpp`'s banner) costs the
most as a fraction of the measurement — which is what makes stage 5g's
`kBurst`-vs-`kPerIter` cross-check informative rather than academic. Between
the two pilot files, the registered shapes span roughly three orders of
magnitude in kernel duration.

### A design problem worth recording: GB registers before main() can set up a device

Google Benchmark's `BENCHMARK_CAPTURE` macros run at **static-init time** —
before `main()` even parses argv — but device setup (query, allocate, upload,
the one-time correctness check) can only happen inside `main()`. Solved with a
plain `std::unique_ptr<Fixture> g_fx`, set once in `main()` before
`RunSpecifiedBenchmarks()` and only read by the registered benchmark functions.
Not a Google Benchmark `Fixture` class: every other bench in this project uses
free functions and captured state, and matching that style keeps
`gb_adapter.hpp` usable without pulling in GB's class-based API as well.

**Correctness runs exactly once, in `main()`, before any GB benchmark
executes** — not inside a registered benchmark function. GB has no "verify
once, then time" notion; verifying inside a registered function would pay a
device→host copy and a CPU-side compare on *every* iteration, which is not what
`reduce_bench.cpp`/`bias_act_bench.cpp` measure and would make the pilot's
numbers incomparable to the originals for a reason that has nothing to do with
Google Benchmark.

### Verification

Full compilation requires an actual CUDA installation, so this stage's
verification is necessarily partial on the Mac:

- **`scripts/typecheck_cuda.sh`**: `ok bench/reduce_bench_gb.cpp`,
  `ok bench/bias_act_bench_gb.cpp` — both typecheck clean under
  `MCKE_WITH_CUDA=1` via `scripts/fakecuda`, including the `Fixture` aggregate
  construction and the `gb_adapter.hpp` template instantiations. This is real
  coverage (it catches signature mismatches, missing includes, and template
  errors) but it is not a compile against real `cuda_runtime_api.h`/`nvcc`.
- CMake conditional gating confirmed by reconfiguring with
  `-DMCKE_ENABLE_CUDA=OFF`: `mcke_reduce_bench_gb` / `mcke_bias_act_bench_gb`
  correctly **absent** from the target list (they call `mcke_kernels`
  launchers, which don't exist in a host-only build).
- `mcke_alloc_bench_gb` and the host suite unaffected: `build-gb` builds with
  0 warnings, `ctest --test-dir build-host` still 100% pass, 145,665 checks.

**Not yet verified, and cannot be from this machine:** that either file
actually compiles under real `nvcc`, links against real `cublas`/`cudart`, or
produces sane numbers. That is stage 5g's job, and it is the first real test of
whether `kBurst` agrees with `Profiler::time_op` — if it does not, the adapter
is wrong regardless of how clean the typecheck is.

### What's next

Stage 5g — **the Colab run**. First real GPU exercise of everything built in
5a–5c: build with `-DMCKE_ENABLE_CUDA=ON -DMCKE_GOOGLE_BENCHMARK=ON`, run both
`_gb` pilots, and check that `kBurst` agrees with `Profiler::time_op` on the
same kernel to within noise. If it does not agree, stop and fix the adapter
before doing anything else in 5g — everything downstream (the cross-check
itself, the nsys work, the roofline) assumes `kBurst` is trustworthy.

## 2026-09-22 — Session 15: the "untouched" artifacts, and three corrections they forced

**Environment:** MacBook Air, host-only. No benchmarks run; no `RESULTS.md` number changed.

### What happened

The Phase 5 handoff said three untracked files "predate this work and are not to be
touched", so stages 5a–5c never opened them. The owner suspected a misunderstanding.
Reading them showed it was one:

- **`phase3_gemm.csv` + `gemm_run.log`** are the CSV and full stdout of Session 5's
  Colab T4 GEMM run (2026-08-30). Every value matches `RESULTS.md` §3d's nine T4 rows
  exactly. They were the **only surviving raw evidence behind any published table**
  (the other four phase CSVs were never kept) and were one `git clean` from gone.
  Now committed as `reports/colab-t4/phase3_gemm.csv` and
  `reports/colab-t4/phase3_gemm_stdout.log` (sha256-verified byte-identical), with a
  provenance `README.md`. Not at the root: `gemm_bench` writes `phase3_gemm.csv` to
  its CWD, so the next run there would have silently overwritten the evidence.
- **`HPC docs/`** is a 97 MB local copy of Explorer's own user documentation
  (`northeastern-rc/rc-public-documentation`; 58 MB of it `.mp4`). **Not committed** —
  third-party, public upstream, and 97 MB in git history is permanent. Now
  gitignored explicitly and referenced from `docs/ENVIRONMENTS.md`.
- A fourth file surfaced: a root-level `nsys_phase4_fanout4x4.nsys-rep`,
  **byte-identical** to the committed `reports/` copy and hidden from `git status` by
  `*.nsys-rep`. Moved to `~/.Trash` (recoverable), not deleted.

### Corrections — named and kept, per this project's convention

**1. Session 12 was wrong that "Explorer's compute nodes have no outbound network."**
Explorer's own docs (`software/systemwide/modules.md:44`): the default `explorer`
module *sets the HTTP proxy nodes use to reach the internet*, and `module purge`
removes it. So FetchContent works on Explorer; the real hazard is a `module purge` in
a job script. **The `MCKE_GOOGLE_BENCHMARK=AUTO` design is unchanged** — the offline
MacBook alone justifies it, and `module purge` is a documented variant of the same
failure — but its written justification was false. Corrected in `CMakeLists.txt`.
The same claim appears in the messages of commits `9d64ea0`/`6ebad54`; those are
corrected here and in the next commit message rather than by rewriting history.

**2. `CMakeLists.txt` listed "a `module load` on Explorer" as a way to find Google
Benchmark.** Explorer has no such module. On Explorer it is FetchContent (via the
proxy) or `FETCHCONTENT_SOURCE_DIR_BENCHMARK`.

**3. `docs/ENVIRONMENTS.md` said `module load cuda/12.4` and "`ncu` normally works
here".** Both pre-date this phase and both were false: Explorer has `cuda/12.1.1`,
`12.3.0`, `12.8.0` (no 12.4), and `ncu` fails with `ERR_NVGPUCTRPERM` on this account
(measured 2026-08-31). Now pinned to `cmake/3.30.2 cuda/12.3.0` — 12.3.0 being what
every V100 row in `RESULTS.md` used.

**Found, deferred to stage 5h:** `scripts/explorer_gpu.sbatch`'s commented bench
lines use flags that don't exist (`--sizes=`, `--out=`, `--trace=`, `--policies=`,
`--policy=`) — `gemm_bench`/`graph_bench` exit 2 on them — its `ncu` line uses
`--kernel-name regex:gemm`, which also matches cuBLAS (`docs/PROFILING.md` §4), and
its partition differs from the `gpu-interactive` + `--gres=gpu:v100-sxm2:1` that
produced §3d.

**Unresolved inconsistency, for the owner:** `CLAUDE.md` §8 says the RC ticket for
`ERR_NVGPUCTRPERM` "is filed and open"; `RESULTS.md` §5a says fixing it "needs an RC
ticket". One of them is stale.

### What was learned

A handoff instruction was treated as a constraint on *reading*, when at most it was
about *modifying*. The cost was concrete: three claims about Explorer were written
into `CMakeLists.txt`, `PROJECT_LOG.md`, the plan and two commit messages, and the
cluster's own documentation — sitting in the working tree the whole time — refuted
two of them in one `grep`. None of the wrong claims had reached code, only comments
and docs, so nothing needed re-verifying beyond a configure. The rule taken from it:
an unexplained "don't touch" means ask what it protects, not skip looking.

### What's next

Unchanged: stage 5d (`scripts/profile_nsys.sh`, `scripts/profile_ncu.sh`,
`tools/nsys_overlap.py`), then 5e (`tools/plot_roofline.py`, which now reads the
committed `reports/colab-t4/phase3_gemm.csv`), then the combined Colab run, 5g.

## 2026-09-22 — Session 16: Phase 5 stage 5d — profiling scripts

**Environment:** MacBook Air, host-only. No `nsys`, no `ncu`, no GPU anywhere on
this machine — which shaped the whole stage, see below. Nothing in `RESULTS.md`
changed except the RC-ticket wording fixed in Session 15's tail (§5a now reads
"filed 2026-08-31 … still open as of 2026-09-22", confirmed correct by the owner).

### What was built

- **`scripts/machine_tag.sh`** (new) — one `mcke_machine_tag()` function, sourced
  by the other scripts (and meant for stage 5f's `regen_results.sh` too), so
  "which machine produced this" isn't copy-pasted three times and drifts.
  `MCKE_MACHINE_TAG` env override for forcing a name (e.g. `explorer-v100`)
  over whatever `nvidia-smi`'s name string would slug to.
- **`scripts/profile_nsys.sh`** (new) — parameterises the nsys command
  `RESULTS.md` §5c documents as having been run once, by hand. Warns (doesn't
  block) if the target binary lacks `nvtxRangePush*` symbols; immediately runs
  the `nsys stats --report cuda_gpu_trace --format csv` post-processing step so
  `tools/nsys_overlap.py` has ready input without a human remembering a second
  command — that missing second step is exactly why the original analysis was
  ad-hoc shell work instead of a script.
- **`scripts/profile_ncu.sh`** (new) — one `ncu` invocation per GEMM variant,
  per `docs/PROFILING.md` §4's already-established recipe (small shape, one
  invocation per variant because `--kernel-name regex:gemm` also matches
  cuBLAS). Detects `ERR_NVGPUCTRPERM` explicitly and **stops after the first
  variant** rather than repeating the same driver-level error seven times,
  since the restriction is machine-wide, not per-kernel.
- **`tools/nsys_overlap.py`** (new) — the analysis that produced `RESULTS.md`
  §5c's three headline numbers (4,404 overlapping pairs / max 4 concurrent
  streams / 3.46× local concurrency factor), which until now existed only as
  unrecorded shell work. A real sweep-line algorithm, not a placeholder.
- **`scripts/explorer_gpu.sbatch`** — the stale ncu/nsys lines replaced with
  calls to the two new scripts; the equally-stale `module load cuda` fixed to
  pin `cmake/3.30.2 cuda/12.3.0`; the bench-flag lines left as commented-out
  `TODO`s naming the real flags each bench actually has today, rather than
  inventing new ones ahead of stage 5f's design.

### A design constraint that shaped the whole stage: no `nsys`, no `ncu`, on this machine

Every prior GPU-adjacent stage (5b, 5c) at least had a *typecheck* path via
`scripts/fakecuda`. This stage has no equivalent — `nsys`/`ncu` are binary CLI
tools, not headers, so there is nothing to typecheck against. Two consequences,
handled by splitting each piece of work into a **verifiable core** and an
**unverifiable shell**, rather than writing the whole thing on faith:

- **`nsys_overlap.py`** separates `extract_csv_via_nsys()` (shells out to the
  real `nsys` binary — genuinely untestable here) from `parse_gpu_trace_csv()`
  and `compute_overlaps()` (pure Python, fully testable). The column-name
  candidates for the CSV parser are an **informed guess**, not a verified
  schema — checked against NVIDIA's own User Guide (does not publish it) and by
  web search (inconclusive) before writing the guess down as a guess rather
  than presenting it as fact. The parser fails loudly with the actual header
  printed if none of the candidates match, by design, so a wrong guess is a
  one-line fix on Colab rather than a silently wrong number.
- **`scripts/profile_nsys.sh` / `scripts/profile_ncu.sh`** were still smoke-tested
  for real, just not on the paths that need the missing binaries: every
  argument-validation and "tool not found" failure path is exercised directly,
  because this machine genuinely lacking `nsys`/`ncu` makes those *real* test
  cases, not simulated ones (same shape as stage 5b's host-only GB target).

### Verification: the overlap algorithm reproduces RESULTS.md's own numbers

`compute_overlaps()`'s self-tests aren't just synthetic sanity checks — one of
them **reconstructs the exact four (start, end) timestamp pairs printed in
`RESULTS.md` §5c's worked-example table** (streams 14–17, the `fanout4x4` run)
and re-derives, independently:

- duration sum: **1,969,885 ns** — matches the document exactly
- wall-clock span: **569,046 ns** — matches the document exactly
- local concurrency factor: **3.4617×** — matches the document's stated 3.46×
  to its own precision

This is a stronger check than a synthetic unit test: it confirms the
*algorithm*, not just its arithmetic, agrees with a number this project already
published and stands behind.

**A real bug was caught by the first synthetic case, before the reconstruction
even ran.** The sweep-line's event tuples used `kind=0` for END and `kind=1`
for START (correct for the *sort order* — ends must sort before starts on a
tie), but the loop then wrote `if is_end:` testing `kind` directly — which is
backwards, since `1` (START) is truthy. Every self-test failed with a
`KeyError` on the very first case (`disjoint`, the simplest possible input).
Fixed by computing `is_end = (kind == 0)` explicitly rather than treating the
sort key as a boolean. Left in the file as a comment, because it's a shape of
mistake ("the sort key and the semantic flag look like the same thing and
aren't") worth flagging for whoever next touches this function.

**`docs/PROFILING.md`'s own account of the nsys workflow was the best source of
ground truth available**, better than web search: `nsys stats --output <name>`'s
exact semantics couldn't be confirmed (NVIDIA's docs describe per-format,
comma-separated output targets, not a plain basename), so `profile_nsys.sh`
uses shell redirection (`nsys stats ... > file.csv`) instead — confirmed safe
because `RESULTS.md` §5c's own account describes querying the trace exactly
that way ("queried directly via `nsys stats --report cuda_gpu_trace`").

**One planning error caught before it became a wrong doc edit.** The Session 15
corrections plan assumed `scripts/explorer_gpu.sbatch`'s `--partition=gpu`
should become `gpu-interactive` to match the sessions that produced the V100
`RESULTS.md` rows. Checking Explorer's own docs first
(`HPC docs/source/gpus/quickstart-h200.md`) showed `gpu` is a real, separate,
documented **batch** partition — `gpu-interactive` is for `srun` sessions
specifically. Left `--partition=gpu` alone; pinned `--gres=gpu:v100-sxm2:1`
instead, which is the actual source of comparability with existing rows.

### Verification

- `python3 tools/nsys_overlap.py --self-test`: **6/6 pass**, including the
  RESULTS.md §5c reconstruction above.
- CSV parsing tested end-to-end with a synthetic `cuda_gpu_trace`-shaped CSV
  (Start+Duration form and Start+End form), and the missing-column failure path
  prints the actual header and a specific fix instruction, exit 1, no traceback.
- `shellcheck` (freshly installed) clean on all four shell scripts, sanity-
  checked against the *existing* `typecheck_cuda.sh`/`build.sh` to confirm the
  tool itself isn't silently no-op'ing.
- Every real failure path exercised directly: no-args usage (exit 2), missing
  binary (exit 1), missing `nsys` (exit 1), missing `ncu` (exit 1),
  `machine_tag.sh`'s no-`nvidia-smi` fallback (`unknown-gpu`) and its
  `MCKE_MACHINE_TAG` override.
- Host suite and `typecheck_cuda.sh` unaffected (145,665 checks; all clean) —
  expected, since this stage touches no C++/CMake, but checked rather than assumed.
- **Not yet verified, and cannot be from here:** that `extract_csv_via_nsys()`
  actually shells out correctly, or that the column-name guesses match real
  `nsys` output. That is stage 5g's job, against the already-committed
  `reports/nsys_phase4_fanout4x4.nsys-rep` — the target is reproducing 4,404 /
  4 / ~3.46×, the same numbers this stage's self-test already reconstructed
  from the *published* timestamps.

### What's next

Stage 5e — `tools/plot_roofline.py`, reading the now-committed
`reports/colab-t4/phase3_gemm.csv`. Then the combined Colab session (5g), which
is where this stage's genuinely untested half — the real `nsys`/`ncu` shell-outs
and the CSV column-name guess — gets its first real exercise.

## 2026-09-22 — Session 17: Phase 5 stage 5e — the roofline plotter

**Environment:** MacBook Air, host-only. matplotlib 3.11.0 / no numpy dependency.
Rendered against the real committed `reports/colab-t4/phase3_gemm.csv` and a
hand-derived (not committed) synthetic multi-kernel CSV for visual verification.
Nothing in `RESULTS.md` changed.

### What was built

**`tools/plot_roofline.py`** (new) — Phase 5's second exit criterion ("a
roofline plot with all kernels on it"). Reads one or more of `Profiler::write_csv`'s
frozen 13-column CSVs, plots every row as one point on a log-log roofline: x =
arithmetic intensity, y = achieved TFLOP/s, with the memory-bound diagonal and
compute-bound ceiling drawn from explicitly-chosen denominators.

### The one design decision worth restating: denominators are never inferred

A CSV's own `attainable_tflops`/`bound` columns were computed against whatever
machine produced that file. Trusting them here would silently draw "the
roofline for whichever machine happened to make this CSV" — exactly the
`bench_common.hpp` trap (`peak_tflops == 0` giving a confidently wrong `%peak`
with no error) applied to a new context. So `--peak-gb-s=`/`--peak-tflops=`,
`--preset=t4`/`--preset=v100` (RESULTS.md §0's own measured values, transcribed
once with a citation), or `MCKE_PEAK_GB_S`/`MCKE_PEAK_TFLOPS` are the only ways
in — give none, and the script refuses to plot, printing the same shape of
message `benchcfg::make_roofline` prints when it aborts.

Every plotted **point** still comes straight from the CSV — only the roofline
**lines** and the memory/compute-bound marker shape are computed fresh from the
chosen denominators, using the exact formulas in `profiler.hpp`'s `Roofline`
struct, reproduced rather than reimplemented differently by accident.

### What was learned — from actually looking at the rendered output, not just running the self-test

The self-tests (`Roofline` formulas cross-checked against `RESULTS.md`'s own
34.5 ridge point and the committed CSV's 8.13 TFLOP/s ceiling; a CSV-corruption
guard) all passed on the first real render. **The render itself was not
acceptable**, and would not have been caught without actually looking at the
image:

- **Labels for closely-spaced points overlapped into unreadable mush.** Four of
  the nine GEMM variants (`tiled_regblock`, `warptile_nodbuf`, `warptile_dbuf`,
  `warptile_vec4`) land within a few percent of each other in TFLOP/s once the
  ladder nears cuBLAS — the well-known "diminishing returns near the ceiling"
  finding already in `RESULTS.md` §3d, now visible as a real rendering problem.
  A fixed `(5, 2)` text offset for every label put them on top of one another.
- **The ridge-point annotation collided with the roofline legend box.** For a
  GEMM-only plot, the ridge point (~34.5 FLOP/byte) happens to fall inside the
  legend's horizontal span at the top-left.
- **The marker-shape footnote collided with the kernel legend** (both wanted
  the bottom-right corner).

Fixed, in order: (1) `compute_label_offsets()` — a deterministic, log-space
union-find clustering (no `adjustText`/`scipy` dependency) that stacks labels
vertically within a cluster instead of overlapping them, unit-tested without
matplotlib; (2) moved the ridge annotation to the bottom of the axes via a
blended data/axes-fraction transform, which no legend ever occupies regardless
of where the ridge point falls; (3) moved the marker-shape footnote to
bottom-left, the one corner neither existing legend claims.

**The stacking fix itself then created a second, subtler problem, also only
visible by looking:** once labels for 4+ tightly-clustered points stack 11pt
apart, the lower-ranked labels drift far enough from their actual marker
(which may sit within a couple of *pixels* of its neighbours) to look
unattached to any point at all. Fixed with a conditional leader line — added
only for stacked members below the first, since the top of a cluster needs
none. Zoomed into the rendered PNG to confirm the leader lines terminate at
each point's own individual data coordinate rather than all converging on one
marker; they do — the visual convergence is real and honest, because those
five GEMM variants' achieved TFLOP/s genuinely are that close together.

**A second real bug, caught the same way stage 5d's was — by a test that
should have passed and didn't.** The `compute_label_offsets` self-test's first
draft asserted the *first* point in input order would rank first (no leader
line); it failed, because the test's own synthetic data had the second point
at a *slightly higher* TFLOP/s value, so it legitimately ranked first by the
function's own (correct) "highest point first" ordering. Fixed the test's
assertions to match the data, not the other way around — the function was
right; the test's assumption about its own fixture was wrong.

**A real, if minor, bug from static analysis**, done as a matter of course
after stage 5d's shellcheck precedent: `pyflakes` (freshly installed) flagged
an unused `import io` left over from an earlier draft. Removed; both new
Python tools now pyflakes-clean.

### Verification

- Self-test: **5/5 pass** — `Roofline` formulas against `RESULTS.md` §0/§3d's
  own published numbers (34.5 ridge, 8.130 TFLOP/s ceiling), a memory-bound
  worked example, the CSV-corruption guard, and `compute_label_offsets`'s
  clustering/leader-line logic.
- Real render against the committed `reports/colab-t4/phase3_gemm.csv`:
  valid SVG (parsed with `xml.etree`), then re-rendered as PNG and **visually
  inspected** at each iteration — this is what caught both rendering defects
  the self-test structurally cannot, since neither is a numeric error.
- A synthetic (uncommitted) multi-kernel CSV mixing `bias_relu` and
  `row_reduce_sum` (memory-bound) alongside the real `gemm` CSV
  (compute-bound) confirmed the tool's actual purpose — multiple kernels on
  one roofline — with roofline-consistent synthetic values (each below its own
  AI-derived ceiling) after a first attempt with hand-picked round numbers
  correctly triggered the "achieved above attainable" warning path, exposing
  that the synthetic data itself was inconsistent, not a bug in the check.
- `pyflakes` clean on both `tools/plot_roofline.py` and `tools/nsys_overlap.py`.
- Host suite unaffected (145,665 checks) — expected, checked anyway.

### What's next

Phase 5's Mac-side work (stages 5a–5e) is now complete. Stage 5f
(`scripts/regen_results.sh` + `tools/render_results.py`, the riskiest stage —
a script that rewrites `RESULTS.md`) is next, still Mac-doable. After that, the
combined Colab session (5g) exercises everything genuinely untested from here:
the GB adapter against a real GPU, the `nsys`/`ncu` shell-outs, and this
plotter against a full multi-kernel CSV set instead of GEMM alone.

## 2026-09-22 — Session 18: Phase 5 stage 5f — regenerating RESULTS.md from pinned datasets

**Environment:** MacBook Air (Apple Silicon), host-only (AppleClang 21.0.0,
CMake 4.4.3, bash 3.2 for the scripts). No GPU. One real benchmark dataset was
captured on this machine (`reports/macbook-host/2026-09-22_1842_0f08940`).

### Correction first (a loose end from Session 15)

Session 15 recorded an "unresolved inconsistency": CLAUDE.md §8 said the ncu RC
ticket is filed and open, RESULTS.md §5a said it "has not been filed yet". The
owner confirmed it **is** filed and open (sent 2026-08-31, per PROJECT_LOG
Session 4's "RC ticket filed" entry). RESULTS.md §5a was the stale one and is
now fixed (commit `8aa6ea2`). That ticket is about GPU performance-counter
permission only — it has nothing to do with Google Benchmark, which needs no
ticket (no module exists; FetchContent through the `explorer` proxy works).

### What was built (4 slices, commits `c373528`, `14b1325`, `c17b75f`, `0f08940`, + this one)

- **`scripts/regen_results.sh`** — runs every bench this machine has built,
  each with its CWD set to a new dataset dir `reports/<tag>/<run-id>/`; saves
  stdout/stderr per bench; writes `manifest.json` (machine, GPU and clocks at
  start/end, driver, full `nvcc --version`, git sha + dirty flag, CMake cache,
  denominators, per-bench argv/exit/time). Marks the dataset **INVALID** if any
  bench exits nonzero, prints "no CUDA device" (which exits 0), prints a FAILing
  validation line, or a numerics-gate FAIL. **Never touches RESULTS.md.**
- **`tools/bench_outputs.py`** — dataset loading and every parser (CSV + the
  stdout-only columns), each citing the printf it reads and failing loudly on
  anything it doesn't recognise.
- **`tools/render_results.py`** — 13 table specs; the fence engine; `--check`,
  `--diff`, `--audit`, `--preview`, `--stale-prose`, `--self-test`.
- **RESULTS.md fenced:** 13 generated tables in `BEGIN/END GENERATED` fences
  that name their dataset; 14 authored tables marked `AUTHORED: <reason>`.
  27 tables total (planning had estimated ~28). `--audit` is clean.
- `.gitignore` (`reports/*` + `!reports/*/` so datasets are committable), env-aware
  machine tags (`colab-t4`, `explorer-v100`, `macbook-host`), the Session-5 T4
  GEMM files moved into `reports/colab-t4/2026-08-30_session5/` with a
  hand-reconstructed manifest (commit **inferred**, unrecoverable fields null).
- `docs/PROFILING.md` §7 documents the workflow.

### Pre-existing errors found and fixed (DECISIONS.md Q12)

Building the renderer meant inventorying every published cell, which surfaced
errors that predate Phase 5 — each fixed as a named correction, superseded text
kept: §5b's "the SAME ratio, independently" (T4 is **3.90×**, not 4.17×); §3d's
−0.10 pp (table says −0.14); §2a's star footnote (≤ 2×floor = 50 ns, not
25 ns); "measured BW is typically 80–90% of spec" (measured: 73.5% and 70.9%);
240.6 vs 240.5 GB/s; a dangling "§0 rule". Two code bugs: **`summary_table`'s
sticky precision** (why many published ms cells have one decimal — e.g. an
L2-control min of "0.0 ms"; proven by a new host test that fails on exactly
row 2 with the bug reinstated) and **`graph_bench` exiting 0 on a failed
numerics gate**. `fma_peak` gained `--warmup/--iters` with defaults unchanged
(3+10, below rule 3) — its output is the denominator for every compute %peak, so
it is measured both ways in 5g before anything changes.

**Recorded, not fixed:** `stream_triad`'s local Roofline leaves `peak_tflops = 0`
(the `bench_common.hpp` trap), so its stdout summary row prints `%peak 0.0%` /
`compute`. Its headline `achieved … GB/s` line is correct and is the only line
the renderer reads, so no table is affected. Left for the owner to decide.

### Results — measured on this machine

- **G1 (golden, real published data):** §3d's nine T4 rows render
  **byte-identical** from the Session-5 dataset — first run.
- **G2 (end-to-end):** a fresh `regen_results.sh` run reproduced all three §2b
  fragmentation tables **byte-for-byte**, four weeks and many commits after
  2026-08-26 — the third independent reproduction (MacBook 08-26, Colab 08-29,
  MacBook 09-22). `--stale-prose` found 0 orphaned numbers; the render's entire
  diff was 3 rows losing their bold. §2b is the first set of tables in RESULTS.md
  built by the renderer.
- **§2a star rule vs the code:** rendered from alloc's CSV and cross-checked
  against the table alloc_bench prints itself in the same run — 42 rows, 0
  mismatches, 85 starred cells agreeing.
- **Round-trip for the six specs with no real data yet:** §0/§1/§3b/§3c/§4
  exact, §3a's only difference the designed new row. Now a permanent self-test.

### What was learned — mostly from tests that passed for the wrong reason

- **My validity test first passed for the wrong reason.** The planted gate-FAIL
  run exited 1 — because my scratch tag (`zz-scratch-t4b`) didn't match `*-t4`,
  so the denominator guard fired before the gate was reached. Only checking *which
  reason* each manifest recorded exposed it. Same shape as the warm-`_deps` and
  `set -u` traps earlier in Phase 5: an exit code is not evidence of the reason.
- **`nvcc --version | tail -1` is a build string**, not the version; found while
  writing the parser that needed the version.
- **Row alignment by all text cells buries status changes.** The first
  `compare_tables` would have reported a gate flipping PASS→FAIL as "missing
  row + new row" — hiding precisely the change 5g's reproducibility report must
  show. Rows now pair by (first cell, occurrence); a self-test pins it.
- **Deciding what may be generated is itself a design surface:** §3b's
  `__syncthreads` counts look like data in stdout but are hard-coded literals —
  "parsing" them would dress a constant up as a measurement, so they're spec
  constants citing `kernels/reduce.cu`. §2b's "one row per allocator" is now
  *asserted*, so a policy that ever changed fragmentation can't be hidden.
- The one-decimal cells in §3a/§3b/§3c/§3d-V100 are the sticky-precision bug's
  fingerprints. They can't be recovered — no raw data was kept — and are fixed
  only when a run is promoted.

### Verification

Host suite 145,670 checks (58,861 + 86,809), 0 failures; `typecheck_cuda.sh` all
clean; `shellcheck -x` clean on all four scripts; pyflakes clean; self-tests
pass for `render_results.py` (G1, G3, G4, G6, G7, G9, comparison, round-trip),
`nsys_overlap.py`, `plot_roofline.py`; `--audit` 0 unclassified; `--check` 0.

### What's next

**Stage 5g — the Colab T4 run, in a fork chat** (handoff prompt given in chat).
The fork captures a dataset, commits it plus `--preview` output for every T4
table (the reproducibility result), runs the GB cross-check, fma_peak both ways,
nsys + `nsys_overlap.py` against the committed trace, the nsys GPU-metrics trial,
the ncu attempt, and the full roofline. **No promotion and no prose edits in the
fork** — promotion is 5i, in the main chat.

**Addendum (same day, owner's decision):** the "recorded, not fixed" `stream_triad`
Roofline trap above is now **fixed** — and it was in *two* programs, not one:
`tools/smoke_vector_add.cpp` (`mcke_smoke`, the source of §1's vector_add row)
left `peak_tflops = 0` in exactly the same way. Both now set an explicitly
**infinite** compute roof rather than a number: these are bandwidth probes with
no argv and no measured compute peak of their own, so any finite value would be
some other machine's. With an infinite compute roof every AI is memory-bound and
`%peak` = % of spec bandwidth. Verified on the host against the real `Roofline`
struct at the T4's own numbers: before, `0.0%` / `compute` (the trap,
reproduced); after, **73.5%** / `memory` — exactly achieved ÷ spec, matching
RESULTS.md §0. Only the stdout summary row changes; GB/s, times, and every table
are unaffected. Fixing `smoke_vector_add.cpp` also exposed that
`scripts/typecheck_cuda.sh` had **never type-checked `tools/*.cpp`**
(`mcke_smoke`, `mcke_device_query`) on the CUDA path — the same class of silent
exclusion as Phase 4's `src/graph/` gap. The glob now includes it; both clean.

## 2026-09-27 to 2026-09-28 — Session 19: Phase 5 stage 5h — Explorer HPC, real V100 numbers (forked chat)

**Hardware: Northeastern Explorer HPC, Tesla V100-SXM2-32GB (sm_70), driver
545.23.08, CUDA 12.3 (`nvcc` 12.3.52), commit `0bc7a08`.** Run in a forked
chat, not the main session; commands were never executed by the agent —
every HPC command was handed to the owner as text and run by them over their
own SSH session, per their explicit standing instruction not to risk their
account being flagged as bot activity.

### What was built

- `reports/explorer-v100/2026-09-26_2335_0bc7a08/` — a full `VALID` dataset
  from `scripts/regen_results.sh` on real V100 hardware (commit `8ee3218`):
  10 bench outputs, manifest, CSVs, fresh-measured V100 denominators
  (**634.2 GB/s / 15.603 TFLOP/s**, superseding the historical
  636.3/15.601 pair — the guard correctly refused to run without both set
  explicitly, since the machine tag is `explorer-v100`, not `-t4`).
- `reports/explorer-v100/2026-09-26_2335_0bc7a08/gb_cross_check.md`
  (commit `fd6a6c4`) — the Google Benchmark adapter (`bench/gb_adapter.hpp`)
  cross-checked against `Profiler::time_op` on a **second** real architecture
  (design/typecheck only covered T4 before this).
- `tools/nsys_overlap.py`'s `_find_header_line()` (commit `63eb823`) — fixed
  a real parsing bug the V100 nsys capture exposed (see below).
- `RESULTS.md` §5c gained a "Second-GPU corroboration: Explorer Tesla
  V100-SXM2" subsection (this commit) — the `fanout4x4` overlap mechanism
  re-derived on V100 and compared numerically to the existing T4 write-up.

### What was learned

- **The lost V100 GEMM ladder was successfully reproduced.** All 9 GEMM
  variants (naive → cuBLAS) matched the previously-recorded V100 numbers in
  `RESULTS.md` §3d within ~1% noise, on a completely independent fresh
  measurement. This was the primary reason HPC access was attempted at all,
  and it worked.
- **The `peak_tflops` fix (Session 18, commit `0bc7a08`) verified LIVE, not
  just on the host.** `stream_triad.stdout.log` and `smoke.stdout.log` both
  show the summary row reading **71.0% / memory**, not `0.0% / compute` —
  turning that Mac-only arithmetic check into a real one, on real hardware,
  per the exact ask a sibling fork had relayed.
- **`ncu` is still blocked by `ERR_NVGPUCTRPERM`**, now confirmed on a
  *second* node (`d1009`, vs. the original `d1007` from the 2026-08-31 RC
  ticket) — solid evidence the block is account/cluster-wide, not
  node-specific. The open RC ticket remains the right next step, not a
  workaround.
- **`kPerIter` shows no measurable per-iteration-sync overhead on V100**,
  contrary to `profiler.hpp`'s own header comment predicting ~10 µs of
  inflation for short kernels. Two of four configs were *faster* under
  `kPerIter`, one flat, and the shortest kernel only +0.8%. Recorded
  honestly as unresolved — a Colab T4 comparison point (stage 5g, not yet
  run) is needed before drawing any conclusion, not explained away under
  HPC time pressure.
- **The V100 corroborates the T4 overlap mechanism, not just the overlap
  fact** (full arithmetic in `RESULTS.md` §5c): the same fixed 40-block
  `fanout4x4` shape leaves V100 at only ~79% of its own measured DRAM
  ceiling (vs. T4's ~100%), and that gap shows up exactly where the model
  predicts — a much smaller per-node slowdown (1.27× vs. 1.87×) and a
  realized speedup (3.26×) close to the 4× ideal, instead of T4's 1.94×.

### Bugs found (three real, one deferred as a non-fix)

- **`tools/nsys_overlap.py`'s `parse_gpu_trace_csv` assumed the CSV header
  was line 1 — fixed (commit `63eb823`).** `nsys stats --format csv` prints
  its own informational preamble to stdout first (`NOTICE: Existing SQLite
  export found...`, `Processing [...] with [...]...`), and
  `scripts/profile_nsys.sh`'s plain `>` redirect captured all of it. The
  column-name candidate lists themselves were **100% correct on the first
  try** (`Start (ns)`, `Duration (ns)`, `Strm`, `Name` all confirmed present
  verbatim in the real nsys 2024.7.1 header) — the bug was purely the
  line-0 assumption. `_find_header_line()` now scans forward for the first
  line that parses as CSV with a recognisable Start+Stream column, so it is
  robust to any future preamble wording, not just this one. A self-test
  reproducing the exact field failure was added. Confirmed after the fix:
  **4,082 overlapping kernel pairs, max 4 concurrent streams, 3.95× local
  concurrency factor** — the same structural result as T4's 4,404/4/3.46×.
- **`scripts/profile_ncu.sh` greps stderr for `ERR_NVGPUCTRPERM`, but `ncu`
  prints that diagnostic to stdout** — not yet fixed. Consequence: the
  script ran all 7 GEMM variants instead of stopping after the first
  permission failure, and (side effect) each sub-invocation of `gemm_bench`
  wrote its own non-compliant `phase3_gemm.csv` into the repo root,
  overwriting itself each time — cleaned up (`rm -f phase3_gemm.csv`), not
  committed.
- **`profile_nsys.sh`'s NVTX-presence heuristic false-negatives on
  header-only NVTX v3** — `nm | grep nvtxRangePush` doesn't find the symbol
  even when ranges work correctly, because NVTX v3 is resolved dynamically
  at runtime, not statically linked. Confirmed a false alarm: the actual
  `nsys stats --report nvtx_sum` output showed 16 real, correctly-named
  ranges (`:b0n0`, `:b1n0`, etc., matching `executor.cpp`'s per-node
  naming). Not yet fixed or removed.
- **Deferred, not investigated:** every one of the 7 GEMM kernel variants in
  the blocked `ncu` attempt reported the exact same `max_rel_err 8.714e-03`
  spot-check — numerically implausible for independently different
  implementations, likely because `--only=<variant>` means cuBLAS never ran
  in-process to provide a live reference for that invocation. Flagged, not
  investigated live — needs source-level follow-up with `gemm_bench.cpp`
  open, not a guess under HPC time pressure.

### Design decisions (and rejected alternatives)

- **No `tmux`, by the owner's explicit choice** — accepted the consequence
  (a closed laptop drops the SSH session and any pending/allocated SLURM
  job) rather than introduce a persistence tool the owner didn't want.
  Recovered twice by reconnecting and re-submitting `srun`.
- **`--exclusive` tried, then dropped for a non-exclusive allocation** after
  a very long `Priority`-reason queue wait with no allocation — the owner's
  own call, accepting some GPU-sharing timing noise in exchange for
  actually getting a session. All V100 numbers in this session (including
  the `fanout4x4` corroboration) come from a **non-exclusive** allocation.
- **`-xdev` mandated on every `find` from this session on**, after the
  owner flagged that an earlier fork's unbounded `find ~` risked
  traversing into a large shared filesystem mount under `$HOME`. All
  subsequent searches were scoped to specific, known, bounded paths.
- **GitHub push authentication**: HTTPS+password was rejected, and the
  system's GUI askpass helper failed headless (`cannot open display`).
  Resolved with a fine-grained, 7-day, repo-scoped Personal Access Token
  embedded directly in the push URL — no credential helper configured on
  shared infrastructure.

### Verification

`bench/gb_adapter.hpp`'s `kBurst` timing mode agreed with `Profiler::time_op`
within 0.3–4.1% across 4 configs on V100 (full table in `gb_cross_check.md`).
`tools/nsys_overlap.py --self-test` extended with the header-preamble
regression case and passes (8/8 checks). All 9 GEMM variants reproduced
within ~1% of the recorded V100 numbers. `git status` clean; all commits
pushed to `origin/main`.

### What's next

**Stage 5g (Colab T4) remains unrun** — needed for the `kPerIter` overhead
comparison point and as the first real exercise of `nsys_overlap.py`'s
`extract_csv_via_nsys()` path (this session went CSV-first via
`--csv`, never through a live `.nsys-rep`). The roofline plot
(`tools/plot_roofline.py --preset=v100`) is the one fully-optional item that
needs no further GPU time — can run from the Mac directly against the
already-pushed V100 dataset, any time.

**Owner's standing requirement, to be carried forward explicitly: once the
full Phase 5 implementation is done, re-run all benchmarks/measurements for
every phase using an EXCLUSIVE GPU allocation**, for authoritative,
noise-free numbers — this session's V100 numbers (GEMM ladder, GB
cross-check, `fanout4x4` overlap corroboration) were all captured on a
**non-exclusive**, shared node, and should be treated as directionally
solid but not the final authoritative record.

Also carried forward, not yet acted on: fix `profile_ncu.sh`'s
stdout-vs-stderr grep bug; fix or remove `profile_nsys.sh`'s NVTX
false-negative heuristic; investigate the suspicious identical
`max_rel_err` across all 7 ncu GEMM variants.

## 2026-09-28 — Session 20: verifying the 5h fork, and four corrections

**Environment:** MacBook Air, host-only. No GPU work; this session verified
the Explorer fork (Session 19, commits `8ee3218`…`91944a2`) against the repo
rather than taking its handoff on trust, then fixed what it found.

### What checked out, independently

- **Dataset** `reports/explorer-v100/2026-09-26_2335_0bc7a08`: `VALID`, clean
  tree at `0bc7a08`, explicit V100 denominators (the 5f guard worked).
- **The V100 GEMM ladder reproduces:** `render_results.py --preview s3d-gemm
  PENDING <dataset>` shows every timing within **≤1.1%** of the 2026-08-31
  published rows (largest: `warptile_vec4`), and **every structural cell
  identical** (tiles, regs, smem, spill, occupancy). The fork's "~1%" holds.
- **GB `kBurst` vs `Profiler::time_op`:** +0.3% … +4.1%, as stated (4.1% is
  across separate processes on a shared node; `kBurst` is the same code path).
- **`nsys_overlap.py` fix** (scan forward for the CSV header; nsys prints a
  preamble) is correct and tested against the verbatim failing bytes. It also
  settles a 5d unknown: **the guessed column names were right**
  (`Start (ns)`, `Duration (ns)`, `Strm`, `Name`).
- Provenance detail the fork didn't state: the run used denominators
  634.2 / 15.603 while its own `stream_triad` / `fma_peak` measured
  635.2 / 15.602 — from an earlier probe run; ≤0.16% effect; the manifest
  records what was used.

### Corrections

1. **RESULTS.md failed `--audit`.** The fork's new §5c V100 nsys-window table
   had no `AUTHORED` marker — exactly the drift `--audit` exists to catch.
   Marked; audit clean again.
2. **`kPerIter`'s conclusion was the wrong mechanism — and so was my own 5b
   design.** `run_per_iteration` records the stop event and *then*
   synchronizes, so the host round trip is **outside** the device-event
   bracket. The fork read "−3.2% … +0.8% vs `kBurst`" as "the round trip is much
   cheaper than ~10 µs"; the measurement could not see the round trip at all.
   What it does show: launch-latency leakage and idle-state effects are small
   on V100 (≤3%, ~50 ns on the 6 µs kernel) — per-launch *event* timing is robust
   to a per-iteration sync. My `gb_adapter.hpp` banner claimed the mode
   "quantifies what that sync costs" — the same error, at design time.
   **Fixed at the instrument:** `kPerIter` now also brackets each iteration
   with `HostTimer` and reports `host_us` and `roundtrip_us` (host − device), so
   5g's T4 run measures the round trip directly. Named corrections added to the
   adapter banner and to the dataset's `gb_cross_check.md`; the Session 19
   text is kept as written.
3. **"Identical `max_rel_err` across all 7 ncu GEMM variants" is not a bug —
   it's a result the fork misread.** Under `--only=<variant>` cuBLAS isn't
   selected, so each variant falls to `spot_check_gemm` with a *fixed* seed —
   the same 1024 elements every time. And the seven hand-written kernels
   compute each output as one sequential FMA chain over *k* in increasing
   order: tiling, register blocking, double buffering and `float4` loads move
   *operands*, not *arithmetic*. Evidence already in the repo: on the T4 all
   seven match cuBLAS with `max_rel_err 0` at 4096³ (bit-identical, Session 5
   log); on the V100 all seven show the *same* `max_rel_err 4.15` vs cuBLAS,
   whose summation order differs. Caveat: on V100 that is **consistent with**
   bit-identity, not proof — the maximum is set by a near-zero element whose
   relative error is insensitive to a few ulps. (Open item: a validation line
   printing "`max_rel_err 4.15` … OK" is legitimate — it passes on the absolute
   floor — but reads like a 415% failure; worth reporting `max_abs_err` too.)
4. **The exclusivity caveat covers the published V100 rows too.** The
   2026-08-31 V100 run (Session 6) used `srun` on `gpu-interactive` with no
   `--exclusive` — the same kind of allocation. Its "authoritative" standing
   rested on the +0.03% drift check, which measures clock stability, not the
   absence of co-tenants. Added to RESULTS.md §5c. Two independent shared-node
   runs a month apart agreeing within 1.1% is evidence contention was low both
   times — not a substitute for the exclusive re-baseline.

### Two script bugs from Session 19, fixed

- **`profile_ncu.sh`** grepped stderr; ncu prints `ERR_NVGPUCTRPERM` on stdout.
  Now captures both (`2>&1 | tee`) and checks the log **regardless of exit
  code**. Tested against the worst case (block on stdout *and* exit 0): stops
  after 1 variant, exits 1.
- **`profile_nsys.sh`'s NVTX pre-check removed** — NVTX v3 resolves at runtime,
  so `nm` can never find the symbol; it warned on a correct build. Replaced by
  a post-profile check of `nsys stats --report nvtx_sum` (count rows after a
  header naming `Range`). The `Range` column name is from memory, unverified
  here; if wrong, the check prints the raw output loudly rather than passing.

### Standing decision, carried forward (owner's, from the fork)

Once Phase 5's implementation is done, **re-benchmark every phase on an
exclusive GPU allocation** — potentially the whole project's measurements, for
consistency. If exclusive time and/or the ncu permission stay blocked for a
long time, **seriously consider pausing** rather than building on results that
will need redoing. The regen/render machinery built in 5f is what makes that
re-baseline cheap: the full V100 run was ~5 minutes of GPU time, one command.

### What's next

Stage **5g, Colab T4, in a fork** — still unrun and still needed: the T4
dataset (the reproducibility preview for every T4 table), the `kPerIter`
round-trip measurement (now actually measurable), fma_peak at 3+10 vs 5+20,
and the first live exercise of `nsys_overlap.py`'s `extract_csv_via_nsys()`
path. Then 5i (promotion + write-up, main chat) — where a new decision waits:
promote the non-exclusive datasets now and re-promote after the exclusive
re-baseline, or hold promotion until then.
