#!/usr/bin/env bash
# =============================================================================
#  scripts/profile_ncu.sh — automates docs/PROFILING.md section 4's ncu recipe:
#  one Nsight Compute invocation PER GEMM VARIANT, never one process for the
#  whole ladder. Feeds RESULTS.md section 5a, currently 8 empty rows blocked
#  on ERR_NVGPUCTRPERM (see below).
#
#  WHY .sh: a thin wrapper choosing arguments and looping over variants; no
#  algorithm lives here (compare tools/nsys_overlap.py, which has one).
#
#  WHY ONE ncu INVOCATION PER VARIANT, not a loop inside one process: this is
#  not a style choice, it is load-bearing, and docs/PROFILING.md section 4
#  already explains why -- `--kernel-name regex:gemm` also matches cuBLAS's own
#  kernels (`turing_sgemm_*` / `volta_sgemm_*`, since "sgemm" contains "gemm"),
#  so a single process launching all 7 variants plus the cuBLAS ceiling row
#  would profile whichever kernel happened to come first out of ~200 matches
#  and the report would be silently mislabelled. --only=<variant> per
#  invocation is what keeps each report attributable to the kernel its
#  filename claims.
#
#  WHY THE SMALL SHAPE (1024, not 4096) and WHY --warmup=1 --iters=3: also
#  already justified in docs/PROFILING.md section 4 -- `--set full` replays
#  each kernel a dozen-plus times, the RATIOS this table cares about (sectors
#  per request, bank conflicts, stall reasons) are shape-independent, and the
#  bench's own NON-COMPLIANT-WITH-RULE-3 banner under --warmup=1 --iters=3 is
#  the correct signal that these timings must never reach RESULTS.md, only the
#  counters do.
#
#  ERR_NVGPUCTRPERM, per DECISIONS.md's Q9 (2026-09-22): expected on every
#  environment tried so far (Explorer: confirmed 2026-08-31, RC ticket filed
#  and open, see RESULTS.md section 5a; Colab: undocumented but believed
#  unavailable). This is a DRIVER-LEVEL restriction
#  (NVreg_RestrictProfilingToAdminUsers), identical for every kernel on a given
#  machine -- so the first variant's failure diagnoses the whole run, and
#  there is no reason to repeat the same wall of driver error 7 times. Detected
#  explicitly below; the script prints one clear diagnosis and stops rather
#  than let the loop grind through every variant to the same result.
#
#  UNTESTED FROM THIS MACHINE: no ncu binary, no GPU. The exact ncu command
#  line is not a guess, though -- it is copied verbatim from
#  docs/PROFILING.md's already-established recipe, which this script only
#  parameterises over variant name and machine tag.
#
#  USAGE:
#    scripts/profile_ncu.sh                         # all 7 variants
#    scripts/profile_ncu.sh tiled_regblock warptile_vec4   # just these
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/machine_tag.sh
source scripts/machine_tag.sh

BIN="${MCKE_GEMM_BENCH:-./build/bin/mcke_gemm_bench}"
if [ ! -x "$BIN" ]; then
  echo "FATAL: '$BIN' is not an executable file (build it first, or set" >&2
  echo "       MCKE_GEMM_BENCH=<path>)." >&2
  exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
  echo "FATAL: 'ncu' not found on PATH. On Explorer it ships inside the cuda" >&2
  echo "       module (no separate Nsight Compute module) -- 'module load cuda'." >&2
  exit 1
fi

# The 7 launcher variants from kernels.hpp's GemmVariant ladder -- NOT cuBLAS,
# which docs/PROFILING.md section 4 explicitly excludes from ncu's --only=
# targets (its kernels are not ours to introspect; gemm_bench's own occupancy
# table already prints "(no attributes)" for that row, see gemm_run.log).
DEFAULT_VARIANTS=(naive_uncoalesced naive tiled_smem tiled_regblock
                  warptile_nodbuf warptile_dbuf warptile_vec4)
if [ $# -gt 0 ]; then VARIANTS=("$@"); else VARIANTS=("${DEFAULT_VARIANTS[@]}"); fi

tag="$(mcke_machine_tag)"
outdir="reports/${tag}"
mkdir -p "$outdir"
stamp="$(date +%Y%m%d_%H%M%S)"

for v in "${VARIANTS[@]}"; do
  out="${outdir}/ncu_gemm_${v}_${stamp}"
  echo "=== ncu: ${v} -> ${out}.ncu-rep ==="
  err_log="$(mktemp)"
  # Build with line info (RelWithDebInfo default, see CLAUDE.md section 4) so
  # SASS maps back to source -- ncu needs no special flag for this, but the
  # binary does need -lineinfo, which is why this is stated rather than left
  # implicit: a build without it produces a report with no source correlation
  # and no error message telling you why.
  if ncu --set full --kernel-name-base function --kernel-name regex:gemm \
         --launch-skip 1 --launch-count 3 --export "$out" \
         "$BIN" 1024 --only="$v" --skip-validation --warmup=1 --iters=3 \
         2> "$err_log"; then
    echo "OK -> ${out}.ncu-rep"
  else
    status=$?
    if grep -q ERR_NVGPUCTRPERM "$err_log"; then
      echo
      echo "*** BLOCKED: ERR_NVGPUCTRPERM ***"
      echo "This account does not have GPU performance-counter access on this"
      echo "machine (NVreg_RestrictProfilingToAdminUsers). This is a"
      echo "DRIVER-LEVEL restriction, identical for every kernel and every"
      echo "variant here -- so it will fail exactly the same way for the"
      echo "remaining ${#VARIANTS[@]} variant(s), and there is no value in"
      echo "repeating the same driver error. Status: RESULTS.md section 5a"
      echo "(RC ticket filed 2026-08-31, rchelp@northeastern.edu, still open"
      echo "as of 2026-09-22 -- see PROJECT_LOG.md for the exact ticket text)."
      rm -f "$err_log"
      exit 1
    fi
    echo "FAILED (exit $status), NOT the known ERR_NVGPUCTRPERM block -- see:" >&2
    cat "$err_log" >&2
    echo "Continuing to the next variant (this failure is not known to be" >&2
    echo "systemic, unlike ERR_NVGPUCTRPERM above)." >&2
  fi
  rm -f "$err_log"
done

echo
echo "Done. Metric-to-decision table: docs/PROFILING.md section 4."
