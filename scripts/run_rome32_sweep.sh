#!/usr/bin/env bash
# Run phase1_adapt over all CIFAR-10-C corruptions for csood_source=rome32, open-set only.
# Usage: bash scripts/run_rome32_sweep.sh tent|bn_adapt|nova-tta
set -euo pipefail

METHOD="${1:?usage: $0 <tent|bn_adapt|nova-tta>}"
SEED="${SEED:-0}"
CORRUPTIONS=(
  gaussian_noise shot_noise impulse_noise defocus_blur glass_blur
  motion_blur zoom_blur snow frost fog brightness contrast
  elastic_transform pixelate jpeg_compression
)

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONPATH="${PYTHONPATH:-.}"

for c in "${CORRUPTIONS[@]}"; do
  echo "=== ${METHOD} | ${c} | rome32 | open | seed=${SEED} ==="
  uv run python scripts/phase1_adapt.py \
    --method "${METHOD}" \
    --corruption "${c}" \
    --csood_source rome32 \
    --open_set true \
    --seed "${SEED}"
done
