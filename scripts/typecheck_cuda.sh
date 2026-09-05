#!/usr/bin/env bash
# =============================================================================
#  scripts/typecheck_cuda.sh
#
#  Type-check the MCKE_WITH_CUDA=1 code path on a machine with NO CUDA, using
#  the fake headers in scripts/fakecuda/.
#
#  WHY: the MacBook cannot run nvcc, so every CUDA branch would otherwise reach
#  Colab completely unverified -- which is exactly how the Phase 0 -> Phase 1
#  transition burned a session. This catches missing includes, wrong signatures,
#  and broken launcher logic locally, for free.
#
#  It does NOT validate device codegen or kernel correctness. See the prelude's
#  banner for the precise boundary of what this proves.
#
#  Usage:  ./scripts/typecheck_cuda.sh                 # everything
#          ./scripts/typecheck_cuda.sh kernels/gemm.cu # one file
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
FAKE="scripts/fakecuda"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CXX="${CXX:-clang++}"
FLAGS=(-std=c++20 -Wall -Wextra -fsyntax-only -I include -I tests -I bench -I kernels -I "$FAKE" -DMCKE_WITH_CUDA=1)
# Second pass with NVTX on. profiler.hpp's NvtxRange gains a member field under
# MCKE_USE_NVTX, and src/graph/executor.cpp wraps every node launch in one, so
# without this pass the entire Nsight-Systems path -- Phase 4's exit criterion --
# would only ever be compiled for the first time on Explorer.
NVTX_FLAGS=("${FLAGS[@]}" -DMCKE_USE_NVTX)

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  # Host .cpp files first (no language extensions needed), then every .cu.
  # Glob bench/*.cpp rather than naming files, so a new bench is covered the
  # moment it exists instead of when someone remembers to add it here.
  # src/*/*.cpp rather than naming the subdirectories: Phase 4 added src/graph/,
  # and the previous list (src/core + src/memory only) silently excluded it --
  # so the MCKE_WITH_CUDA=1 path of every graph file would have reached Colab
  # completely unverified, which is the exact failure this script exists to
  # prevent. Same reasoning for tests/test_*.cpp.
  targets=(src/*/*.cpp bench/*.cpp tests/test_*.cpp)
  while IFS= read -r f; do targets+=("$f"); done < <(find kernels bench tests tools -name '*.cu' | sort)
fi

fail=0
for f in "${targets[@]}"; do
  case "$f" in
    *.cu)
      # Strip the <<<grid,block,smem,stream>>> launch syntax, which no host
      # compiler can parse, then compile with the language prelude forced in.
      #
      # perl with a NON-GREEDY .*? rather than sed's [^>]*: a launch config
      # legitimately contains '>' characters, e.g.
      #     kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(...)
      # and a [^>]* class stops at the first one, leaving a mangled line that
      # reports as a syntax error in the kernel rather than a limitation here.
      out="$TMP/$(echo "$f" | tr '/' '_').cpp"
      perl -pe 's/<<<.*?>>>//g' "$f" > "$out"
      if "$CXX" "${FLAGS[@]}" -include "$FAKE/cuda_lang_prelude.h" "$out" 2>"$TMP/err"; then
        echo "  ok    $f"
      else
        # Constants referenced only inside a stripped <<<>>> now look unused.
        if grep -qv 'unused-variable\|unused-const-variable\|warning' "$TMP/err"; then
          echo "  FAIL  $f"; sed -n '1,20p' "$TMP/err"; fail=1
        else
          echo "  ok    $f  (unused-* warnings are sed artifacts)"
        fi
      fi
      ;;
    *)
      if "$CXX" "${FLAGS[@]}" "$f" 2>"$TMP/err"; then echo "  ok    $f"
      else echo "  FAIL  $f"; sed -n '1,20p' "$TMP/err"; fail=1; fi
      # NVTX pass, host .cpp only: the ranges live in executor.cpp and the
      # macro changes NvtxRange's layout, so this catches an ODR-shaped
      # mistake and a missing include before Explorer does.
      if ! "$CXX" "${NVTX_FLAGS[@]}" "$f" 2>"$TMP/errn"; then
        echo "  FAIL  $f  (with -DMCKE_USE_NVTX)"; sed -n '1,20p' "$TMP/errn"; fail=1
      fi
      ;;
  esac
done
[ $fail -eq 0 ] && echo "CUDA-path type-check: all clean" || echo "CUDA-path type-check: FAILURES"
exit $fail
