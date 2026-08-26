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
#SBATCH --gres=gpu:1
# Request a SPECIFIC GPU model when the numbers must be comparable across runs.
# Mixing a V100 result and an A100 result in one table invites wrong conclusions.
#   #SBATCH --gres=gpu:a100:1
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
module load cuda           # pin the version explicitly once you know it, e.g. cuda/12.4
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
# ./build/bin/mcke_alloc_bench  --trace=dl   --out=reports/alloc_${SLURM_JOB_ID}.csv
# ./build/bin/mcke_gemm_bench   --sizes=1024,2048,4096 --out=reports/gemm_${SLURM_JOB_ID}.csv
# ./build/bin/mcke_reduce_bench --out=reports/reduce_${SLURM_JOB_ID}.csv
# ./build/bin/mcke_graph_bench  --policies=all --out=reports/graph_${SLURM_JOB_ID}.csv

# ---- Nsight passes (slow; keep launch-count small) ---------------------------
# nsys profile --trace=cuda,nvtx --output=reports/nsys_${SLURM_JOB_ID} \
#     ./build/bin/mcke_graph_bench --policy=chain_greedy
# ncu --set full --kernel-name regex:gemm --launch-skip 5 --launch-count 3 \
#     --export reports/ncu_gemm_${SLURM_JOB_ID} ./build/bin/mcke_gemm_bench 4096

kill $SMI_PID 2>/dev/null || true
echo "=== done at $(date -Iseconds) ==="
echo "Copy reports/*.csv into RESULTS.md, and append a PROJECT_LOG.md entry"
echo "noting: job id, node, GPU model, driver/CUDA versions, observed clocks."
