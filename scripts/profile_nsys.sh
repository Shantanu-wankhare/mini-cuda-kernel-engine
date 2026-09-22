#!/usr/bin/env bash
# =============================================================================
#  scripts/profile_nsys.sh — parameterises docs/PROFILING.md section 3's nsys
#  workflow, which until Phase 5 stage 5d had only ever been run by hand once
#  (the fanout4x4 timeline behind RESULTS.md section 5c).
#
#  WHY .sh: this is a thin wrapper around two external CLI invocations (nsys
#  profile, nsys stats) plus some path bookkeeping -- exactly the job shell is
#  for, per CLAUDE.md section 5 ("scripts/ — build helpers, SLURM job
#  scripts"). No algorithm lives here; that is tools/nsys_overlap.py's job.
#
#  WHAT IT DOES, beyond running the one command docs/PROFILING.md already
#  documents:
#   1. Names the output by machine (scripts/machine_tag.sh) so a Colab run and
#      an Explorer run never collide in reports/, matching the reports/<tag>/
#      layout Phase 5 stage 5f builds on.
#   2. Warns (does not block) if the target binary looks like it was NOT built
#      with NVTX -- an unlabelled timeline is technically valid but useless,
#      per this file's own docs/PROFILING.md precedent ("without NVTX ranges
#      you get anonymous kernel bars and cannot tell which graph node is
#      which").
#   3. Immediately runs the `nsys stats --report cuda_gpu_trace --format csv`
#      post-processing step and saves it as a companion CSV, so
#      tools/nsys_overlap.py has ready-made input without a human needing to
#      remember a second command -- exactly the gap that made the fanout4x4
#      overlap analysis ad-hoc shell work instead of a committed, rerunnable
#      script in the first place (RESULTS.md section 5c says as much).
#
#  UNTESTED FROM THIS MACHINE: there is no nsys binary and no GPU here. Every
#  path-handling and argument-forwarding decision below was checked by hand
#  against docs/PROFILING.md's already-verified command and this project's own
#  shell conventions (scripts/typecheck_cuda.sh, scripts/build.sh); the actual
#  nsys invocation itself gets its first real exercise in Phase 5 stage 5g, on
#  Colab.
#
#  USAGE:
#    scripts/profile_nsys.sh <bench-binary> [bench-args...]
#
#  EXAMPLE (reproduces the RESULTS.md section 5c artifact, verbatim):
#    scripts/profile_nsys.sh ./build/bin/mcke_graph_bench \
#        --only=fanout4x4 --iters=5 --warmup=2
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/machine_tag.sh
source scripts/machine_tag.sh

if [ $# -lt 1 ]; then
  echo "usage: $0 <bench-binary> [bench-args...]" >&2
  echo "example: $0 ./build/bin/mcke_graph_bench --only=fanout4x4 --iters=5 --warmup=2" >&2
  exit 2
fi

BIN="$1"
shift
if [ ! -x "$BIN" ]; then
  echo "FATAL: '$BIN' is not an executable file (build it first)." >&2
  exit 1
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "FATAL: 'nsys' not found on PATH. Available on Colab/Explorer via the" >&2
  echo "       cuda module or a dedicated Nsight Systems module; not on the" >&2
  echo "       MacBook, which is why this script was never run from here." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# NVTX heuristic check. `strings`/`nm` looking for the symbol NvtxRange's
# constructor calls (nvtxRangePushA) is not authoritative -- a stripped binary
# or a statically-inlined call could hide it -- but it is a cheap, honest
# early warning rather than silence, and false positives (symbol present but
# MCKE_USE_NVTX was actually off) are impossible since the symbol only exists
# under that macro (profiler.hpp's #if MCKE_WITH_CUDA && defined(MCKE_USE_NVTX)).
# ---------------------------------------------------------------------------
if command -v nm >/dev/null 2>&1; then
  if ! nm "$BIN" 2>/dev/null | grep -q nvtxRangePush; then
    echo "WARNING: '$BIN' does not appear to reference nvtxRangePush*." >&2
    echo "         Rebuild with -DMCKE_USE_NVTX=ON, or the nsys timeline will" >&2
    echo "         be anonymous kernel bars with no graph-node labels" >&2
    echo "         (docs/PROFILING.md section 3). Continuing anyway." >&2
  fi
fi

tag="$(mcke_machine_tag)"
outdir="reports/${tag}"
mkdir -p "$outdir"

name="$(basename "$BIN")"
for a in "$@"; do
  case "$a" in
    --only=*) name="${name}_${a#--only=}" ;;
  esac
done
stamp="$(date +%Y%m%d_%H%M%S)"
rep_base="${outdir}/nsys_${name}_${stamp}"

echo "=== nsys profile -> ${rep_base}.nsys-rep ==="
echo "command: $BIN $*"
nsys profile --trace=cuda,nvtx,osrt --stats=true -o "$rep_base" "$BIN" "$@"

echo
echo "=== nsys stats --report cuda_gpu_trace -> ${rep_base}_gputrace.csv ==="
# Shell redirection, not `nsys stats --output <name>`: this project could not
# confirm --output's exact naming/format semantics from this machine (no nsys
# to test against, and NVIDIA's docs describe it taking per-format,
# comma-separated targets like `.,-` rather than a single plain basename --
# see this file's own module-level honesty note in tools/nsys_overlap.py for
# the same class of gap). `nsys stats --report X --format csv` writing its
# report to STDOUT by default is the one thing confirmed independently: it is
# how RESULTS.md section 5c's own account describes querying the trace
# ("queried directly via nsys stats --report cuda_gpu_trace"). Redirecting
# that stdout needs no unverified flag semantics.
nsys stats --report cuda_gpu_trace --format csv "${rep_base}.nsys-rep" \
    > "${rep_base}_gputrace.csv"

echo
echo "Next: python3 tools/nsys_overlap.py --csv ${rep_base}_gputrace.csv"
