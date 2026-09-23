#!/usr/bin/env bash
# =============================================================================
#  scripts/machine_tag.sh — one place to answer "which machine produced this?"
#
#  WHY THIS EXISTS AS ITS OWN FILE rather than copy-pasted into
#  profile_nsys.sh, profile_ncu.sh and regen_results.sh: three copies of "how do
#  I name the reports/ subdirectory" is exactly the kind of duplication that
#  drifts. Meant to be SOURCED, not executed:
#      source "$(dirname "$0")/machine_tag.sh"
#      tag="$(mcke_machine_tag)"
#
#  WHY .sh and not folded into a .py: its callers are bash scripts that exist
#  to shell out to nsys/ncu/benches; a bash caller sourcing a Python helper for
#  one string is more machinery than the problem needs.
#
#  ---------------------------------------------------------------------------
#  TAG = <environment>-<gpu>   e.g. colab-t4, explorer-v100, macbook-host
#
#  Phase 5 stage 5f changed this from a bare GPU slug ("t4", "v100-sxm2-32gb")
#  to environment + GPU, for two reasons:
#   1. The directories already in the repo were named by hand that way
#      (reports/colab-t4/), and a second, auto-derived name for the same machine
#      ("t4") was a mismatch deferred from Session 16 to be reconciled here.
#   2. The ENVIRONMENT is part of what a number means (CLAUDE.md section 3): the
#      same T4 on Colab (unlocked clocks, shared host) and on GCP is not the same
#      measurement context, so a GPU name alone under-identifies the machine.
#
#  Detection is BEST-EFFORT, and deliberately low-stakes: RESULTS.md tables name
#  the exact dataset directory they render from (DECISIONS.md Q10), so a
#  mis-detected tag only misnames a directory -- it can never corrupt a table.
#  The handoffs for Colab/Explorer runs set MCKE_MACHINE_TAG explicitly anyway,
#  and mcke_machine_tag_reason says how a tag was derived, for the manifest.
# =============================================================================

# Environment, or empty if unknown.
_mcke_env() {
  if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then echo "macbook"; return; fi
  # Colab: COLAB_RELEASE_TAG / COLAB_GPU are set in Colab runtimes, and
  # /content/sample_data is the stock sample directory every Colab VM has.
  if [ -n "${COLAB_RELEASE_TAG:-}${COLAB_GPU:-}" ] || [ -d /content/sample_data ]; then
    echo "colab"; return
  fi
  # Explorer: inside a SLURM job the cluster names itself. Outside a job, fall
  # back to the fully-qualified hostname. Both unverified until run there,
  # which is why the Explorer handoff sets MCKE_MACHINE_TAG explicitly.
  if [ -n "${SLURM_CLUSTER_NAME:-}" ]; then
    echo "$SLURM_CLUSTER_NAME" | tr '[:upper:]' '[:lower:]'; return
  fi
  case "$(hostname -f 2>/dev/null || hostname)" in
    *explorer*|*discovery*) echo "explorer"; return ;;
  esac
  echo ""
}

# Short GPU name, or empty if no GPU is visible.
# "Tesla T4" -> t4, "Tesla V100-SXM2-32GB" -> v100, "NVIDIA A100-SXM4-40GB" ->
# a100, "NVIDIA GeForce RTX 5060 Laptop GPU" -> rtx5060. The full name string
# goes in the dataset manifest; the tag only needs to be short and stable.
_mcke_gpu_short() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ -n "$name" ] || { echo ""; return; }
  echo "$name" | tr '[:upper:]' '[:lower:]' | sed -E \
    -e 's/^(tesla|nvidia) +//' -e 's/^geforce +//' \
    -e 's/^(rtx|gtx) +([0-9]+).*/\1\2/' \
    -e 's/[- ].*//' -e 's/[^a-z0-9]//g'
}

mcke_machine_tag() {
  if [ -n "${MCKE_MACHINE_TAG:-}" ]; then echo "$MCKE_MACHINE_TAG"; return; fi
  local env gpu
  env="$(_mcke_env)"
  gpu="$(_mcke_gpu_short)"
  if   [ -n "$env" ] && [ -n "$gpu" ]; then echo "${env}-${gpu}"
  elif [ -n "$env" ];                  then echo "${env}-host"
  elif [ -n "$gpu" ];                  then echo "$gpu"
  else                                      echo "unknown-host"
  fi
}

# One line explaining how the tag was derived -- recorded in manifest.json so a
# directory name six months from now can be traced back to its evidence.
mcke_machine_tag_reason() {
  if [ -n "${MCKE_MACHINE_TAG:-}" ]; then
    echo "MCKE_MACHINE_TAG override"
    return
  fi
  echo "auto: env='$(_mcke_env)' (uname/COLAB_*/SLURM_CLUSTER_NAME/hostname), gpu='$(_mcke_gpu_short)' (nvidia-smi)"
}
