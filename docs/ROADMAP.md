# ROADMAP — MCKE

Phases are ordered so that **each one changes exactly one variable**. That is what
makes the benchmark table at the end attributable: every speedup has a single
named cause.

Each phase lists: goal → files → environment → exit criteria (what must be
*measured*, not merely written).

Legend for environment: **[Mac]** host-only, no GPU · **[Colab]** fast iteration ·
**[5060]** long interactive sessions, Nsight GUI · **[Explorer]** authoritative
numbers · **[GCP]** reproducible mid-length runs / arch A/B.

---

## Phase 0 — Architecture and scaffold ✅ *(done, session 1)*
**Env: [Mac]**

Interfaces, build system, error handling, host-only test harness.

**Exit:** host build green (`4410 checks, 0 failures`); all headers compile under
`-Wall -Wextra`; docs in place. ✅

---

## Phase 1 — First light on a GPU
**Env: [Colab]** — 30 minutes, ~2 credits. Nothing here needs a good GPU.

1. Build with `-DMCKE_ENABLE_CUDA=ON`. Fix whatever nvcc rejects (the CUDA path
   has never been compiled — expect 1-3 real errors).
2. `mcke_device_query` → paste output into `PROJECT_LOG.md` and
   `RESULTS.md` §0.
3. `mcke_smoke` → verifies correctness, prints achieved bandwidth.
4. Write `bench/stream_triad.cu`: a pure `a[i] = b[i] + s*c[i]` bandwidth
   microbenchmark. **This gives the measured-bandwidth denominator for every
   later % of peak.**
5. Write `bench/fma_peak.cu`: a register-only FMA loop (no memory traffic) to
   measure sustained f32 FLOP/s. **This is the compute denominator.** Do not use
   a spec sheet — see `src/core/profiler.cpp` for why.

**Exit:** measured BW and measured FMA peak recorded per machine; roofline ridge
point computed; `mcke_smoke` verification passes.

**Concepts to be able to explain:** why events not `chrono`; why warmup; why the
spec bandwidth number differs from measured; grid-stride loops; coalescing.

---

## Phase 2 — Memory management ✅ *(done, 2026-08-26 to 2026-08-29)*
**Env: [Mac]** for 2a/2b logic and tests, **[Colab]** for 2c numbers.

- **2a. `BuddyAllocator`** (`src/memory/buddy_allocator.cpp`). Split/merge over
  slabs, per-level free lists, stream-ordered pending-free list.
  *Test on the Mac first* — the host backend makes every code path reachable
  without a GPU, including the OOM paths.
- **2b. `FreeListAllocator`** (`src/memory/freelist_allocator.cpp`). Size classes,
  no coalescing, optional block splitting.
- **2c. `bench/alloc_bench.cpp`** — replay one recorded allocation trace through
  `raw` / `buddy` / `freelist` in a single process. Two traces: (i) uniform
  power-of-two sizes, (ii) DL-shaped sizes (768, 3072, 50257 × dtype).
- **2d. Stream-safety test** — deliberately construct the reuse-across-streams
  race, show that the naive pool produces wrong numbers and the stream-ordered
  one does not. **Do this.** A memory manager whose safety argument is untested
  is a memory manager with a latent race.

**Exit (all measured, into `RESULTS.md` §2):** allocate latency median + p99 for
all three; `raw_malloc_calls` flat after warm-up; utilisation and internal waste
per allocator per trace; a documented case where each design wins; the race test
passing. **All satisfied** — see `RESULTS.md` §2, `PROJECT_LOG.md` Session 3.
Actually built as three traces (added `dl_transformer_bypass` for the
large-allocation bypass path) and all three cross-stream reuse policies
(`kSameStreamOnly`/`kCoarseStreamPoll`/`kPerFreeEvent`) rather than one, per an
owner decision partway through — the deallocate-latency comparison decomposes
cleanly into "pure bookkeeping" vs. "+ one driver call" as a result.

**Concepts:** stream-ordered semantics; why `cudaFree` synchronises; internal vs
external fragmentation; O(1) coalescing via the buddy XOR identity.

---

## Phase 3 — Kernels
**Env: [Colab]** to get each variant *correct*, then **[Explorer]** for the
authoritative table, then **[5060]** for interactive Nsight Compute GUI work.

Order matters — one idea per step, benchmark after each:

- **3a. Fused bias+activation** (easiest real win). Unfused vs fused; then
  `float`/`float2`/`float4` vectorised loads. Measure the traffic halving.
- **3b. Row reduction.** smem tree → warp shuffle (`__shfl_down_sync`) → two-pass
  for long rows. Count `__syncthreads` in each; correlate with the timing.
- **3c. Softmax.** Three-pass stable → online one-pass (Milakov & Gimelshein,
  the FlashAttention rescaling trick). Verify numerics against a NumPy reference
  (`tools/gen_reference.py` — Python here because generating reference data is
  exactly what NumPy is for, and it is not part of the runtime).
- **3d. GEMM**, one variant at a time:
  `naive` → `tiled_smem` → `+register blocking` → `+warp tiling & double
  buffering` → `cuBLAS reference`.
  For each: record regs/thread and smem/block from `-Xptxas -v`, compute
  theoretical occupancy by hand, then compare against Nsight's measured
  `achieved_occupancy`. **The hand-calculation vs. measurement comparison is the
  learning; the TFLOP/s is just the score.**

**Exit:** `RESULTS.md` §3 filled; every speedup attributed to one change; for the
final GEMM, a written explanation of the remaining gap to cuBLAS.

**Concepts:** shared memory bank conflicts; register blocking and why the
accumulator must be a compile-time-sized array; occupancy vs. ILP; `__shfl_*`
semantics and the `_sync` mask; bank-conflict padding; double buffering and why
it removes a `__syncthreads` bubble; numerical stability of softmax.

---

## Phase 4 — Graph engine and async scheduling
**Env: [Mac]** for graph logic (topo sort, liveness, planner — all host code,
all unit-testable), **[Explorer]** for the overlap numbers and `nsys` timelines.

- `src/graph/graph.cpp` — `finalize()`, Kahn sort, levels, live ranges,
  `to_dot()`.
- `src/graph/executor.cpp` — the three schedule policies, event insertion, the
  liveness-based memory planner.
- `src/graph/ops_*.cpp` — the four `Op` subclasses wired to the Phase 3 kernels.
- `bench/graph_bench.cpp` — diamond graph, deep chain, and a small
  transformer-block-shaped graph (GEMM → bias+GELU → GEMM → softmax).
- **Numerics gate:** `kLevelParallel` and `kChainGreedy` must produce
  bit-identical output to `kSequential`. Any difference is a race, not a
  rounding artefact. Make this an automated check
  (`ExecutorOptions::validate_numerics`).

**Exit:** speedup per policy per graph; events recorded per iteration; peak
memory with and without liveness reuse; an `nsys` timeline screenshot showing
actual overlap (or explaining its absence).

**Concepts:** event-based cross-stream dependencies; why chain-greedy minimises
event count; liveness analysis; launch-bound vs compute-bound graphs.

---

## Phase 5 — Benchmark and profiling infrastructure
**Env: [Explorer]** primarily.

- Add **Google Benchmark** (FetchContent) for `bench/*` — now we genuinely need
  its statistics (repetitions, `--benchmark_min_time`, mean/median/stddev).
- `MCKE_USE_NVTX=ON` and NVTX ranges around every graph node.
- `scripts/profile_ncu.sh`, `scripts/profile_nsys.sh` with the metric sets from
  `docs/PROFILING.md`.
- `tools/plot_roofline.py` — reads the profiler CSV, plots the roofline with each
  kernel placed on it. (Python: plotting, not runtime.)

**Exit:** one command regenerates every table in `RESULTS.md`; a roofline plot
with all kernels on it.

---

## Phase 6 — Stretch (pick by interest, not obligation)
**Env: [Explorer]** / **[GCP]** for arch comparisons.

- f16/bf16 + tensor cores (`mma.sync` via inline PTX, or WMMA). sm_80+.
- Inline **PTX** for one kernel — e.g. `ld.global.nc.v4.f32` or a `cp.async`
  pipeline — and diff the SASS against the C++ version to see what changed.
- CUDA Graphs capture of `ExecutionPlan` (cuts per-launch CPU cost ~10×).
- Kernel autotuning: sweep tile shapes, cache the best per (arch, shape).
- Multi-GPU / NCCL-free peer-to-peer copy.
- Arch A/B: same kernel on sm_75 vs sm_80 vs sm_89 vs sm_120, explaining the
  differences from the SM architecture changes.

---

## Environment cheat-sheet

| Task | Use | Why not the others |
|---|---|---|
| Editing, design, host logic tests | **Mac** | Free, instant, no GPU needed |
| Making CUDA code compile at all | **Colab** | Cheapest possible failure loop |
| Kernel correctness iteration | **Colab** | Fast, and correctness doesn't need a fast GPU |
| Authoritative timing tables | **Explorer** | Stable clocks, exclusive nodes, no throttling |
| Nsight Compute interactive GUI | **5060** | Local, no X-forwarding/latency pain |
| Long Nsight Compute batch runs | **Explorer** | `ncu` replays kernels many times; needs time and stability |
| Architecture comparisons | **GCP** | Can pick the GPU model; costs money, so plan the run first |
| Anything where a number goes in the README | **Explorer** | A throttled laptop number is not a result |
