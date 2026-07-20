#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/step1x3d-matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/step1x3d-triton}"
mkdir -p "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}"
cd "${PROJECT_ROOT}"

exec conda run --no-capture-output -n step1x \
  python train.py --config configs/uni3d_repa.yaml --train --gpus 0 "$@"
