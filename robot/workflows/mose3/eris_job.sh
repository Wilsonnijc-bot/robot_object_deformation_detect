#!/usr/bin/env bash
# Stage 1, CPU (PDI geometry env):
#   sbatch --partition=short --account=wang --cpus-per-task=16 --mem=64G --time=01:00:00 \
#     eris_job.sh queries WORKSPACE DEPENDENCY RUN SOURCES
# Stage 2, one shared H200 (mose3 env); worker concurrency is a scheduler option:
#   sbatch --partition=gpu-h200 --account=wang --gres=gpu:nvidia_h200:1 --cpus-per-task=32 --mem=256G \
#     --time=03:00:00 eris_job.sh track WORKSPACE DEPENDENCY RUN SOURCES [run_gpu_batch options]
# Display-only rigidity videos for the replay (after track):
#   sbatch ...same GPU request... eris_job.sh render WORKSPACE DEPENDENCY RUN SOURCES [run_gpu_batch options]
# Stage 3, CPU (PDI geometry env): the link signals, including the dense mask fit (after track):
#   sbatch --partition=short --account=wang --cpus-per-task=4 --mem=32G --time=00:30:00 \
#     eris_job.sh signals WORKSPACE DEPENDENCY RUN SOURCES
# eris_pipeline.sh chains track and signals unattended from a login-node tmux session.
# Slurm spools this file at submission, so a running job never sees later edits.
set -Eeuo pipefail
stage=${1:?queries, track, render or signals}; workspace=${2:?}; dependency=${3:?}; run=${4:?}; sources=${5:?}; shift 5
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm allocation required' >&2; exit 2; }
cd "$workspace"
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PATH="$dependency/bin:$PATH" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
/usr/bin/mkdir -p "$run/metadata"
log="$run/metadata/$stage-$SLURM_JOB_ID.log"
trap 'rc=$?; printf "%s\n" "$rc" > "$run/metadata/$stage-$SLURM_JOB_ID.exit"' EXIT
case "$stage" in
  queries)
    export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
    command=("$dependency/env/geometry/bin/python" -u -m robot.workflows.mose3.initialize_queries
             --inputs "$run/inputs" --sources "$sources" --output "$run/queries"
             --processes "${SLURM_CPUS_PER_TASK:-8}" "$@")
    ;;
  track)
    command=("$dependency/env/mose3/bin/python" -u -m robot.workflows.mose3.run_gpu_batch
             --inputs "$run/inputs" --queries "$run/queries" --sources "$sources" --output "$run/cases"
             --checkpoint "$dependency/models/mose3" --mose3-root "$dependency/source/MoSE3"
             --metadata "$run/metadata" "$@")
    ;;
  render)
    command=("$dependency/env/mose3/bin/python" -u -m robot.workflows.mose3.run_gpu_batch
             --worker render --tracking "$run/cases"
             --inputs "$run/inputs" --queries "$run/queries" --sources "$sources" --output "$run/renders"
             --checkpoint "$dependency/models/mose3" --mose3-root "$dependency/source/MoSE3"
             --metadata "$run/metadata" "$@")
    ;;
  signals)
    export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
    command=("$dependency/env/geometry/bin/python" -u -m robot.scoring.mose3.signals --run "$run" "$@")
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2;;
esac
"${command[@]}" > "$log" 2>&1 &
pid=$!
# scancel/stop: forward to the scheduler, which cancels every worker and queued video.
trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; exit 130' TERM INT HUP
wait "$pid"
