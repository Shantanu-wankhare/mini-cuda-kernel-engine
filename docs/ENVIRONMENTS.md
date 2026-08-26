# ENVIRONMENTS.md — per-machine setup

## MacBook Air (Apple Silicon) — host-only

No CUDA, ever. This is a first-class *development* environment for the ~60% of
the codebase that is host logic.

```bash
brew install cmake ninja        # cmake is not installed by default
cmake -B build-host -G Ninja -DMCKE_ENABLE_CUDA=OFF
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

Without CMake (verified working):

```bash
clang++ -std=c++20 -Wall -Wextra -I include -DMCKE_WITH_CUDA=0 \
  tests/test_host_core.cpp src/core/device.cpp src/memory/allocator.cpp \
  -o /tmp/mcke_tests && /tmp/mcke_tests
```

For editor intelligence on files you cannot compile here: `clangd` reads
`build-host/compile_commands.json`. CUDA-only files will show errors in the
editor — that is expected and not a problem.

---

## Google Colab Pro

Runtime → Change runtime type → GPU. Check what you got *before* spending
credits: a T4 (sm_75) and an A100 (sm_80) are very different machines.

```python
# Cell 1 — what am I on?
!nvidia-smi
!nvcc --version
```

```python
# Cell 2 — build
!git clone <repo-url> mcke && cd mcke && \
 cmake -B build -DMCKE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
 cmake --build build -j2
```

```python
# Cell 3 — first light
!cd mcke && ./build/bin/mcke_device_query && ./build/bin/mcke_smoke
```

Notes:
- Colab's CMake is recent enough for `CUDA_ARCHITECTURES native`.
- `-j2` not `-j$(nproc)`: Colab CPUs are few and nvcc is memory-hungry.
- **Sessions are ephemeral.** Copy `RESULTS.md` rows out immediately, or write
  them to Drive. A number left in a dead Colab session did not happen.
- `nsys` is usually available; `ncu` often is **not** (it needs profiling
  permissions the container lacks). Do deep kernel profiling on Explorer or the
  5060.
- Budget: `mcke_smoke` and kernel correctness loops cost almost nothing. Long
  GEMM sweeps do — run those on Explorer.

---

## Windows laptop, RTX 5060 (Blackwell, sm_120)

**Requires CUDA Toolkit ≥ 12.8.** Earlier toolkits do not know `sm_120` and you
will get "no kernel image is available for execution on the device" — from the
*runtime*, not the compiler, which makes it confusing. If `native` misdetects,
pass it explicitly:

```bash
cmake -B build -DMCKE_ENABLE_CUDA=ON -DMCKE_CUDA_ARCH=120
```

WSL2 works and is more pleasant than MSVC for this project (our C++ is
GCC/Clang-flavoured). Native Windows also works; use the "x64 Native Tools"
prompt so CMake finds MSVC.

**Thermal reality:** a laptop 5060 throttles within seconds of sustained load.
Same kernel, minute 3, can be 20% slower. Use this machine for:
- interactive Nsight Compute GUI work (excellent for this),
- long debugging sessions,
- correctness iteration.

Do **not** use it for headline numbers. If you must, log
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -l 1`
alongside the run and say "laptop, throttled" in the results row.

---

## Northeastern Explorer (SLURM) — the authoritative environment

```bash
module avail cuda
module load cuda/12.4          # or whatever is current — record the version
srun -p gpu --gres=gpu:1 --pty /bin/bash    # interactive node for quick checks
```

Batch job: see `scripts/explorer_gpu.sbatch`.

Notes:
- Request the specific GPU type when you need comparability:
  `--gres=gpu:a100:1` vs `--gres=gpu:v100:1`. Mixing V100 and A100 numbers in one
  table is the easiest way to draw a wrong conclusion.
- Exclusive nodes (`--exclusive`) matter for timing: a co-tenant kernel will show
  up as timing variance you spend an hour chasing.
- `ncu` normally works here. If it reports
  `ERR_NVGPUCTRPERM`, profiling counters are restricted — ask RC support, or fall
  back to `nsys` + event timing.
- Long runs: this is where GEMM tile sweeps and multi-size scaling studies belong.

---

## GCP ($300 credit)

Compute Engine with a GPU is the simpler path (Vertex AI adds a managed layer we
do not need). Pick the accelerator deliberately — T4 (sm_75), L4 (sm_89), A100
(sm_80), H100 (sm_90) — because the interesting use of this credit is an
**architecture comparison** that Explorer's fixed fleet can't give you.

- Use a Deep Learning VM image so CUDA is preinstalled.
- **Stop the instance when idle.** A forgotten A100 burns the whole credit in
  about a day.
- Plan the run *before* starting the instance: write the script, test it on
  Colab, then start, run, copy results out, stop.
