#!/usr/bin/env bash
set -Eeuo pipefail
root=${1:?owned experiment directory}
test -n "${SLURM_JOB_ID:-}"
cd "$root"
export HOSTNAME=$(hostname)
module load CUDA/12.9.0 tmux/3.5a-GCCcore-13.3.0
# The Reg2Inv environment and native libraries were deleted with the CAD workspace on 2026-10-15;
# rebuild them (see reg2inv_core/README.md) and point REG2INV_ENV / REG2INV_NATIVE at them to rerun.
env_root=${REG2INV_ENV:?rebuilt Reg2Inv environment}; native_root=${REG2INV_NATIVE:?rebuilt Reg2Inv native libraries}
dependency=/PHShome/zy992/Wilson/dependency
export CUDA_HOME=/apps/software/CUDA/12.9.0
export PATH="$env_root/bin:$dependency/env/compiler/bin:$CUDA_HOME/bin:$PATH"
export CC="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-cc"
export CXX="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$native_root/lib:${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH="$root/results/20261014-45/scratch/cuda-cache"
export TORCH_EXTENSIONS_DIR="$root/results/20261014-45/scratch/torch-extensions"
export TRITON_CACHE_DIR="$root/results/20261014-45/scratch/triton-cache"
mkdir -p "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root"
export FRAME0_PYTHON="$env_root/bin/python"
socket="reg2inv-frame0-$SLURM_JOB_ID"
session="frame0-experiment-$SLURM_JOB_ID"
log="$root/results/20261014-45/metadata/experiment-$SLURM_JOB_ID.log"
exitfile="$root/results/20261014-45/metadata/experiment-$SLURM_JOB_ID.exit"
test ! -e "$exitfile"
cleanup(){ tmux -L "$socket" kill-session -t "$session" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
tmux -L "$socket" new-session -d -s "$session" "bash '$root/gpu_inner.sh' '$root' '$log' '$exitfile'"
printf 'TMUX_SESSION=%s SOCKET=%s LOG=%s\n' "$session" "$socket" "$log"
while test ! -f "$exitfile"; do
    if ! tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
        test -f "$exitfile" || exit 4
    fi
    sleep 5
done
exit "$(cat "$exitfile")"
