#!/usr/bin/env bash
# =============================================================================
#  scripts/machine_tag.sh — one place to answer "which machine produced this?"
#
#  WHY THIS EXISTS AS ITS OWN FILE rather than copy-pasted into
#  profile_nsys.sh, profile_ncu.sh, and (Phase 5 stage 5f) regen_results.sh:
#  three copies of "how do I name the reports/ subdirectory" is exactly the
#  kind of duplication that drifts — one script gets a bugfix, the others
#  don't, and RESULTS.md rule 1 (record the environment) silently breaks on
#  whichever one nobody touched. Meant to be SOURCED, not executed:
#      source "$(dirname "$0")/machine_tag.sh"
#      tag="$(mcke_machine_tag)"
#
#  WHY .sh and not folded into a .py: profile_nsys.sh / profile_ncu.sh are
#  themselves bash (they exist to shell out to nsys/ncu), and a bash caller
#  sourcing a Python helper for one string is more machinery than the problem
#  needs. tools/nsys_overlap.py takes the resulting tag as a plain --tag flag
#  instead of resolving it itself, for the same reason.
#
#  Override with MCKE_MACHINE_TAG=<name> when the auto-detected slug isn't the
#  name you want in reports/ (e.g. forcing "explorer-v100" on a shared node
#  whose nvidia-smi name string is a generic "Tesla V100-SXM2-32GB").
# =============================================================================

mcke_machine_tag() {
  if [ -n "${MCKE_MACHINE_TAG:-}" ]; then
    echo "$MCKE_MACHINE_TAG"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "$name" ]; then
      # "Tesla T4" -> "t4", "Tesla V100-SXM2-32GB" -> "v100-sxm2-32gb". Strips
      # the "Tesla"/"NVIDIA" marketing prefix (constant across every card this
      # project uses, so it adds nothing) and folds everything else to a safe
      # directory-name slug.
      echo "$name" | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/^(tesla|nvidia) +//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g'
      return
    fi
  fi
  echo "unknown-gpu"
}
