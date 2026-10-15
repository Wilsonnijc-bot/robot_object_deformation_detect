#!/usr/bin/env bash
# Unattended track -> signals chain for one run, from a login-node tmux session:
#   module load tmux
#   tmux new-session -d -s NAME "bash eris_pipeline.sh WORKSPACE DEPENDENCY RUN SOURCES [run_gpu_batch options]"
# Each stage is its own allocation; a stage starts only after the previous one exited 0.
# Job IDs go to RUN/metadata/pipeline.jobs and progress to RUN/metadata/pipeline.log.
# Stop: tmux kill-session -t NAME (or TERM this script). The active job is cancelled
# and later stages are never submitted.
set -Eeuo pipefail
workspace=${1:?}; dependency=${2:?}; run=${3:?}; sources=${4:?}; shift 4
job="$workspace/robot/workflows/mose3/eris_job.sh"
metadata="$run/metadata"
/usr/bin/mkdir -p "$metadata"
active=''

note() { printf '%s %s\n' "$(date -Is)" "$*" >> "$metadata/pipeline.log"; }
stop() {
  if [ -n "$active" ]; then scancel "$active" || true; fi
  note "stop requested; cancelled ${active:-nothing}; later stages not submitted"
  exit 130
}
trap stop TERM INT HUP

# stage name, sbatch resource options..., -- , stage arguments...
run_stage() {
  local stage=$1; shift
  local options=()
  while [ "$1" != -- ]; do options+=("$1"); shift; done; shift
  active=$(sbatch --parsable --job-name="mose3-$stage" --account=wang "${options[@]}" \
           --output="$run/slurm-$stage-%j.out" "$job" "$stage" "$workspace" "$dependency" "$run" "$sources" "$@") \
    || { note "$stage submission failed"; active=''; return 1; }
  printf '%s %s\n' "$stage" "$active" >> "$metadata/pipeline.jobs"
  note "$stage submitted as $active"
  local marker="$metadata/$stage-$active.exit" state
  while [ ! -f "$marker" ]; do
    state=$(sacct -j "$active" -X -n -P -o State 2>/dev/null | head -n 1 || true)
    case "$state" in
      FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_ME*|PREEMPTED*|BOOT_FAIL*|DEADLINE*) break;;
    esac
    sleep 60 & wait $!          # interruptible, so a stop request is handled at once
  done
  local rc; rc=$(cat "$marker" 2>/dev/null || echo "none (Slurm state ${state:-unknown})")
  note "$stage job $active exited $rc"
  active=''
  [ "$rc" = 0 ]
}

note "pipeline started: run=$run"
run_stage track --partition=gpu-h200 --gres=gpu:nvidia_h200:1 --cpus-per-task=32 --mem=256G --time=01:00:00 -- "$@" \
  || { note 'track failed; signals not submitted'; exit 1; }
run_stage signals --partition=short --cpus-per-task=4 --mem=32G --time=00:30:00 -- \
  || { note 'signals failed'; exit 1; }
note 'pipeline complete'
