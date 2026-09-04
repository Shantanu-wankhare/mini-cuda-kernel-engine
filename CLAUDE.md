# CLAUDE.md — MCKE (Mini CUDA Kernel Engine)

Agent-facing project brief. Read this first in every session.

---

## 1. What this project is

A from-scratch GPU compute runtime in **C++20 + CUDA**, with **no deep-learning
framework dependencies**. Five components:

1. **Custom device memory management** — buddy allocator + segregated free-list
   pool over pre-reserved slabs, replacing per-tensor `cudaMalloc`/`cudaFree`.
2. **Tensor layer** — `Storage` (owning, refcounted) / `Tensor` (view) split.
3. **Computation graph engine** — SSA-style DAG, topological scheduling,
   liveness-based memory reuse, multi-stream async execution with CUDA events.
4. **Hand-tuned kernels** — tiled GEMM (shared memory → register blocking →
   warp tiling → double buffering), warp-shuffle reductions, online softmax,
   fused bias+GELU.
5. **Profiling/telemetry** — CUDA-event timing, NVTX ranges, roofline analysis,
   Nsight Compute/Systems workflows, CSV output feeding `RESULTS.md`.

The deliverable is **not just working code** — it is a measured, explained
comparison at every step. A number without an explanation of *why* it is that
number is incomplete work in this project.

## 2. Owner and working style — IMPORTANT

The owner is learning GPU systems engineering from first principles. Therefore:

- **Explain before coding.** For every design decision (buddy vs. free-list,
  DAG vs. flat list, warp-shuffle vs. shared-memory reduction, event vs.
  stream-synchronize), state the alternatives and the tradeoff in plain terms
  *before* the code.
- **Explain every file you create**: its name, why that name, what goes in it,
  and why that language/extension (`.cu` vs `.cpp` vs `.hpp` vs `.cuh` vs `.py`).
- **Comment non-trivial CUDA lines inline**, in the code — memory access
  patterns, sync points, occupancy-relevant constants, why a `__restrict__` or
  `#pragma unroll` is there. Not just in prose.
- **Do not skip "obvious" steps.** If something is standard practice, say why in
  one sentence.
- **Check understanding.** After a major design decision or new concept, ask the
  owner to explain it back in their own words. If the explanation is shaky,
  correct it before moving on. Do not let a shaky foundation compound.
- Prefer honest negative results over impressive-sounding ones. "Overlap bought
  nothing because each kernel already saturated the SMs" is a better outcome
  than an unexplained 1.4x.
- **Log every `AskUserQuestion` to `DECISIONS.md` at the repo root, live, in
  every session.** Each time you ask one, append (don't wait until session
  end): the question text, every option offered with its description, which
  one the owner picked, and a one-line reason for the recommendation. Newest
  at the bottom, one entry per question, dated. This is a running decision
  log the owner re-opens in fresh chats once they understand a topic better —
  keep it in the repo root (not `docs/`) so it's easy to find and paste into
  a new session.

## 3. Hardware environments

| Environment | GPU | Best used for | Notes |
|---|---|---|---|
| **MacBook Air (Apple Silicon)** | none | editing, design, host-only unit tests | `-DMCKE_ENABLE_CUDA=OFF` builds and tests all host logic. No nvcc, ever. |
| **Google Colab Pro** (~300 credits) | T4 (sm_75) / L4 (sm_89) / A100 (sm_80) | fast iteration, first-light kernel runs, numeric validation | Sessions die; treat as ephemeral. Budget credits — no long profiling runs here. |
| **GCP ($300 credit)** | configurable (Compute Engine / Vertex AI) | reproducible mid-length runs, arch A/B (sm_80 vs sm_89) | Costs real money per hour. Shut instances down. |
| **Windows laptop, RTX 5060** | Blackwell, **sm_120** | long interactive sessions, Nsight Compute GUI | **Needs CUDA ≥ 12.8** for sm_120. Thermally throttles within seconds — never use for headline numbers. |
| **Northeastern Explorer HPC** | data-center NVIDIA (V100/A100/H100 class) | **all authoritative benchmarks and Nsight profiling** | Stable clocks, exclusive nodes, long jobs. This is where RESULTS.md numbers come from. |

**Rule: quick iteration on Colab, authoritative numbers on Explorer.** Always
record which environment produced a number, alongside the number.

## 4. Build and run

```bash
# --- macOS (host-only: no CUDA). Tests host logic: buddy math, shapes, allocator.
cmake -B build-host -DMCKE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

```bash
# --- No CMake installed? One-command host test (this is verified to work):
clang++ -std=c++20 -Wall -Wextra -I include -I tests -DMCKE_WITH_CUDA=0 \
  tests/test_host_core.cpp src/core/device.cpp src/memory/allocator.cpp \
  src/core/host_timer.cpp \
  src/memory/buddy_allocator.cpp src/memory/freelist_allocator.cpp \
  -o /tmp/mcke_tests && /tmp/mcke_tests
```

```bash
# --- GPU machine (Colab / GCP / 5060 / Explorer)
cmake -B build -DMCKE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/bin/mcke_device_query            # ALWAYS run first on a new machine; log the output
./build/bin/mcke_smoke                   # end-to-end: alloc + H2D + kernel + events + verify
```

```bash
# --- Register/shared-memory usage per kernel (needed for occupancy work, Phase 3)
cmake -B build -DMCKE_ENABLE_CUDA=ON -DMCKE_CUDA_PTXAS_VERBOSE=ON && cmake --build build -j 2>&1 | grep -A2 ptxas
```

```bash
# --- Explicit architecture (when `native` guesses wrong, e.g. cross-compiling)
cmake -B build -DMCKE_CUDA_ARCH="75;80;89;120"
```

Profiling commands live in `docs/PROFILING.md`. Per-environment setup
(Colab bootstrap cell, Explorer `sbatch`) lives in `docs/ENVIRONMENTS.md`.

## 5. Code conventions

- **C++20.** Concepts for kernel element types, `<bit>` for allocator math,
  `[[nodiscard]]` on anything returning `Status`. No `std::expected` (C++23).
- **File extensions carry meaning:**
  - `.hpp` — host-includable header. Must compile with plain `clang++` on macOS.
    Any CUDA header inclusion must be behind `#if MCKE_WITH_CUDA`.
  - `.cuh` — header containing `__device__`/`__global__` code. nvcc only.
  - `.cu` — kernels and `<<<>>>` launch sites only. **Keep these few and small**;
    nvcc recompiles each one per target architecture.
  - `.cpp` — everything else, *including* host-side CUDA runtime API calls
    (`cudaMalloc`, `cudaMemcpyAsync`) — those need no nvcc.
  - `.py` — test harnesses and plotting only. Never part of the runtime.
- **CUDA types stop at the runtime boundary.** Above `mcke::rt`, everything is
  plain C++20 so it stays testable on the Mac. `DeviceInfo` is a POD snapshot of
  `cudaDeviceProp` for exactly this reason.
- **Error handling:** `Status`/`StatusOr` for expected failures (OOM, bad
  shapes); throw for programmer errors and unrecoverable CUDA failures.
  `MCKE_CUDA_RETURN_IF_ERROR` / `MCKE_CUDA_CHECK` — never a bare CUDA call.
- **No synchronisation inside `Op::launch`.** `cudaStreamSynchronize` is allowed
  in exactly two places: `GraphExecutor::synchronize()` and benchmark timing.
- **Every op reports `flops()` and `bytes()` (ideal traffic).** Non-negotiable —
  they are the roofline axes.
- Namespaces: `mcke`, `mcke::rt` (runtime/CUDA boundary), `mcke::kernels`
  (launchers), `mcke::buddy` (allocator math).

## 6. Repository layout

```
CMakeLists.txt          modern CMake; CUDA is an optional language
include/mcke/
  core/       config.hpp  status.hpp  dtype.hpp  device.hpp
  runtime/    cuda_check.hpp  stream.hpp            # the CUDA boundary
  memory/     buddy_math.hpp  allocator.hpp  reuse_policy.hpp
              buddy_allocator.hpp  freelist_allocator.hpp
  tensor/     shape.hpp  tensor.hpp
  graph/      op.hpp  graph.hpp  executor.hpp
  kernels/    kernels.hpp                           # launcher declarations only
  profiling/  profiler.hpp  host_timer.hpp
src/          host implementations (.cpp) — no device code
kernels/      *.cu — kernels + launch sites, nvcc only
tools/        device_query, smoke tests
tests/        host-only unit tests (no GTest dependency)
bench/        benchmark harnesses (Google Benchmark from Phase 5);
              alloc_bench.cpp (Phase 2c) is host-only, no CUDA required
scripts/      build helpers, SLURM job scripts
docs/         ROADMAP.md  PROFILING.md  ENVIRONMENTS.md
```

## 7. Session logging — DO THIS AT THE END OF EVERY SESSION

**Append a dated entry to `PROJECT_LOG.md`** (newest at the bottom) containing:

- Date, and **which hardware environment** the session used (macOS host-only /
  Colab T4 / Explorer A100 / RTX 5060 / GCP).
- What was built (files added or changed, and what they do).
- What was learned — including anything that turned out to be wrong.
- **Benchmarks run, with the actual measured numbers**, not just methodology.
- Design decisions taken and the alternatives rejected.
- What's next, concretely enough to resume cold.

**Whenever a benchmark or profiling pass runs**, record the real measured
results in **both** `PROJECT_LOG.md` (narrative, dated) and `RESULTS.md`
(running tables). Include: GPU model, driver + CUDA version, clocks if known,
problem size, iterations, median and min time, achieved GB/s and TFLOP/s,
arithmetic intensity, % of the relevant peak, and the exact command line. The
standard is: *the owner must be able to explain every number in an interview six
months from now without re-running anything.*

`LEARNING_LOG.md` is the **owner's** file, not the project's. Add to it when a
concept was confusing, had to be re-derived, or remains open — and at the end of
each major phase, append a 3-5 question interviewer-style Q&A for that
component. Do not put project status in `LEARNING_LOG.md`, and do not put
personal learning notes in `PROJECT_LOG.md`.

## 8. Current status

**Phases 0–3 complete.** See `PROJECT_LOG.md` for full session-by-session
detail and `docs/ROADMAP.md` for the phase plan. Now starting **Phase 4**
(graph engine and async scheduling).

- Host-only build verified on macOS: 58,856 checks passing in `test_host_core`.
- CUDA path built and verified on three real GPUs: Colab Tesla T4 (sm_75),
  Northeastern Explorer Tesla V100-SXM2 (sm_70), plus the allocator race tests.
  RTX 5060 (sm_120) not yet touched.
- Implemented and measured: `Status`/`StatusOr`, `DType`, `DeviceInfo`,
  `rt::Stream`/`Event`, buddy allocator + free-list allocator (both with all
  three cross-stream reuse policies), `RawDeviceAllocator`, `Shape`,
  `Profiler::time_op`, roofline math, and every Phase 3 kernel: fused
  bias+activation, row reduction (tree/shuffle/two-pass), row softmax
  (three-pass/online), and the 8-row GEMM ladder (naive through cuBLAS) with
  its own tile-dispatch and occupancy-calculator header
  (`kernels/gemm_tile.hpp`).
- `RESULTS.md` §0–§3d filled with real measured numbers on two architectures.
  §5 (Nsight Compute deep dives) is blocked on `ncu` permissions on Explorer
  (`ERR_NVGPUCTRPERM`) — an RC ticket is filed and open; not blocking further
  project work.
- Declared but **not yet implemented**: `Storage`/`Tensor` methods, all four
  `Op` subclasses, `Graph`, `GraphExecutor` — this is Phase 4.
