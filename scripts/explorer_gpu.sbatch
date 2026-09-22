#!/bin/bash
# =============================================================================
#  scripts/explorer_gpu.sbatch — SLURM batch job for Northeastern Explorer.
#
#  WHY .sbatch: it is a bash script whose leading #SBATCH comments are directives
#  read by the scheduler. The extension is conventional, not required, but makes
#  it obvious this is not meant to be run directly.
#
#  WHY batch instead of an interactive srun for benchmarks: an interactive shell
#  shares the node's state with whatever you were doing before, and you cannot
#  reproduce it later. A batch script IS the record of how a number was produced
#  — commit it next to the result.
#
#  Submit:  sbatch scripts/explorer_gpu.sbatch
#  Watch:   squeue -u $USER ; tail -f logs/mcke_<jobid>.out
# =============================================================================
#SBATCH --job-name=mcke-bench
#SBATCH --partition=gpu
# `gpu` (batch) is a real, separate partition from `gpu-interactive` (srun) --
# confirmed against HPC docs/source/gpus/quickstart-h200.md, which uses this
# exact partition for a batch job. Do not "correct" this to gpu-interactive;
# that is for interactive sessions, not sbatch (checked 2026-09-22 after an
# earlier draft of this correction assumed the two were interchangeable).
#
# v100-sxm2 pinned, not gpu:1: every RESULTS.md V100 row (section 0, 3d, 4)
# was measured on this exact model via an INTERACTIVE gpu-interactive session
# (DECISIONS.md, 2026-08-31) -- pinning it here keeps a BATCH run comparable to
# that existing dataset instead of landing on whatever GPU `gpu:1` picks.
#SBATCH --gres=gpu:v100-sxm2:1
# Other models available on Explorer (comment out the line above, uncomment
# one of these) -- but mixing architectures in one RESULTS.md table invites
# wrong conclusions, so pick ONE per project-wide comparison, not per job:
#   #SBATCH --gres=gpu:a100:1
#   #SBATCH --gres=gpu:h200:1
#SBATCH --exclusive          # co-tenants show up as timing variance you will chase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mcke_%j.out
#SBATCH --error=logs/mcke_%j.err

set -euo pipefail
mkdir -p logs reports

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds) ==="

# Record the toolchain. Six months from now this block is the difference between
# a defensible number and a mystery.
#
# Both versions pinned, not left to whatever `module load cuda` defaults to
# (an earlier version of this line said `cuda/12.4`, which does not exist on
# Explorer -- corrected 2026-09-22 against the cluster's own module list:
# available versions are 12.1.1, 12.3.0, 12.8.0). 12.3.0 matches every V100 row
# already in RESULTS.md. cmake is loaded explicitly because Phase 5 needs it
# for Google Benchmark's FetchContent wiring; Explorer's `explorer` module
# (loaded by default) keeps the HTTP proxy up for that fetch -- do NOT
# `module purge` in this script, and if you must, `module load explorer`
# immediately after (CMakeLists.txt has the full explanation).
module load cmake/3.30.2 cuda/12.3.0
nvcc --version
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,clocks.max.mem \
           --format=csv

# ---- build fresh, in the job, so the binary matches the recorded toolchain ----
./scripts/build.sh cuda

# ---- hardware inventory + first light ----
./build/bin/mcke_device_query
./build/bin/mcke_smoke

# ---- clock/thermal trace alongside the run, so variance is explainable -------
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \
           --format=csv -l 2 > "reports/clocks_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# ---- benchmarks (uncomment as each phase lands) ------------------------------
# TODO(stage 5f/5h): the flags below (--trace=, --out=, --sizes=, --policies=)
# do NOT exist on any bench today -- checked 2026-09-22 against gemm_bench.cpp
# and graph_bench.cpp's actual `parse_args`, both of which are FATAL on any
# unrecognised flag (RESULTS.md rule-2 violations are treated as bugs, not
# warnings). Uncommenting these as written would exit 2 before anything runs.
# Real per-bench flags today: gemm_bench takes `[N] --m= --n= --k= --only=
# --warmup= --iters= --skip-validation`; graph_bench takes `--streams= --warmup=
# --iters= --only= --skip-gate --profile --gemm-n=`; reduce_bench/bias_act_bench
# take only `--peak-gb-s=`/`--peak-tflops=`. A uniform `--out=<path>` across all
# four, and the per-machine reports/<tag>/ layout, is stage 5f's
# regen_results.sh -- don't hand-roll it here ahead of that design.
# ./build/bin/mcke_alloc_bench
# ./build/bin/mcke_gemm_bench 4096
# ./build/bin/mcke_reduce_bench
# ./build/bin/mcke_graph_bench --streams=4

# ---- Nsight passes (slow; keep launch-count small) ---------------------------
# Phase 5 stage 5d: scripts/profile_nsys.sh / scripts/profile_ncu.sh replace
# the two hand-rolled commands that used to be here. Both auto-detect the
# machine tag (scripts/machine_tag.sh) and write into reports/<tag>/, so no
# ${SLURM_JOB_ID} path-building is needed.
#
# The line this replaced used `--policy=chain_greedy`, which does not exist:
# graph_bench has no --policy= flag at all -- it runs all three schedule
# policies per graph in one process, and --only= selects which GRAPH(S), not
# which policy (docs/PROFILING.md section 3 already documents this exact trap).
# ./scripts/profile_nsys.sh ./build/bin/mcke_graph_bench --only=fanout4x4 --iters=5 --warmup=2
#
# The line this replaced used `--kernel-name regex:gemm --launch-skip 5` with
# NO --only=, which would profile whichever of cuBLAS's ~200 kernels or this
# project's own 7 variants happened to come first (docs/PROFILING.md section
# 4). profile_ncu.sh loops --only=<variant> once per variant, correctly.
# ./scripts/profile_ncu.sh

kill $SMI_PID 2>/dev/null || true
echo "=== done at $(date -Iseconds) ==="
echo "Copy reports/*.csv into RESULTS.md, and append a PROJECT_LOG.md entry"
echo "noting: job id, node, GPU model, driver/CUDA versions, observed clocks."
