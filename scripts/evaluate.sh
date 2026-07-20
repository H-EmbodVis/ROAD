#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: GPU_IDS=0,1 bash scripts/evaluate.sh MANIFEST IMAGE_ROOT GLB_ROOT OUTPUT_DIR" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS:-0,1}"
if [[ ${#GPUS[@]} -lt 2 ]]; then
  echo "GPU_IDS must contain two GPU IDs, for example GPU_IDS=0,1" >&2
  exit 2
fi

MANIFEST=$1
IMAGE_ROOT=$2
GLB_ROOT=$3
OUTPUT_DIR=$4
mkdir -p "${OUTPUT_DIR}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${OUTPUT_DIR}/torch_extensions}"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m evaluation.evaluate_uni3d \
  --manifest "${MANIFEST}" \
  --image-root "${IMAGE_ROOT}" \
  --glb-root "${GLB_ROOT}" \
  --openclip-checkpoint pretrained/evaluation/uni3d_openclip.bin \
  --output "${OUTPUT_DIR}/uni3d_i.json" &
UNI3D_PID=$!

CUDA_VISIBLE_DEVICES="${GPUS[1]}" python -m evaluation.evaluate_ulip \
  --manifest "${MANIFEST}" \
  --image-root "${IMAGE_ROOT}" \
  --glb-root "${GLB_ROOT}" \
  --openclip-checkpoint pretrained/evaluation/ulip_openclip.bin \
  --ulip-checkpoint pretrained/evaluation/ulip2_pointbert.pt \
  --output "${OUTPUT_DIR}/ulip_i.json" &
ULIP_PID=$!

status=0
wait "${UNI3D_PID}" || status=$?
wait "${ULIP_PID}" || status=$?
exit "${status}"
