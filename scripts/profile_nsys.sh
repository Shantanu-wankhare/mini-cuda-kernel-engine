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
#   2. Warns (does not block) if the finished trace contains NO NVTX ranges
#      -- checked in the trace itself, after profiling (see below for why not
#      in the binary). An unlabelled timeline is technically valid but useless,
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

# (The pre-profile `nm | grep nvtxRangePush` check that used to sit here was
# REMOVED 2026-09-28. It could not work: NVTX v3 is header-only, and its entry
# points resolve at RUNTIME through a function-pointer table the NVTX injection
# library fills in -- there is no nvtxRangePushA symbol for `nm` to find. On
# Explorer it warned "no NVTX" on a binary whose trace then held 16 correctly
# named ranges (PROJECT_LOG Session 19). A warning that fires on correct builds
# trains people to ignore it, which is worse than no warning. The check now runs
# on the TRACE, after profiling -- the only place the answer is certain.)

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

# ---------------------------------------------------------------------------
# NVTX check, on the trace itself. `nvtx_sum` lists one row per named range; a
# build without -DMCKE_USE_NVTX=ON produces none. We look for a CSV header row
# naming a `Range` column and count the data rows after it -- tolerant of any
# preamble nsys prints first (the same lesson tools/nsys_overlap.py learned on
# Explorer). If nsys's wording ever differs, the fallback is loud, not silent:
# the raw output is shown so a human can judge.
# ---------------------------------------------------------------------------
nvtx_out="$(nsys stats --report nvtx_sum --format csv "${rep_base}.nsys-rep" 2>&1 || true)"
nvtx_rows="$(printf '%s\n' "$nvtx_out" | awk -F, '
  !hdr && /(^|,)"?Range"?(,|$)/ { hdr = 1; next }
  hdr && NF > 1 { n++ }
  END { print n + 0 }')"
if [ "$nvtx_rows" -gt 0 ]; then
  echo "NVTX: ${nvtx_rows} named range(s) in the trace -- timeline is labelled."
else
  echo "WARNING: no NVTX ranges found in the trace. Rebuild with" >&2
  echo "         -DMCKE_USE_NVTX=ON, or the timeline is anonymous kernel bars" >&2
  echo "         (docs/PROFILING.md section 3). nsys nvtx_sum said:" >&2
  printf '%s\n' "$nvtx_out" | head -8 | sed 's/^/           /' >&2
fi

echo
echo "Next: python3 tools/nsys_overlap.py --csv ${rep_base}_gputrace.csv"
