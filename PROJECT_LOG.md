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
