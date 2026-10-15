#!/usr/bin/env bash
set -Eeuo pipefail
root=${1:?}; log=${2:?}; exitfile=${3:?}
cd "$root"
exec > "$log" 2>&1
trap 'rc=$?; printf "EXIT_STATUS=%s\n" "$rc"; printf "%s\n" "$rc" > "$exitfile"' EXIT
printf 'JOB=%s HOST=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES"
"$FRAME0_PYTHON" -u preflight.py
"$FRAME0_PYTHON" -u run.py
