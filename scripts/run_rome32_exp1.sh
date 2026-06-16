#!/usr/bin/env bash
# Run exp1_auroc.py for every Rome32 open-set stream across the three methods.
set -euo pipefail

CORRUPTIONS=(
  gaussian_noise shot_noise impulse_noise defocus_blur glass_blur
  motion_blur zoom_blur snow frost fog brightness contrast
  elastic_transform pixelate jpeg_compression
)

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONPATH="${PYTHONPATH:-.}"

for c in "${CORRUPTIONS[@]}"; do
  echo "=== exp1 | ${c} ==="
  uv run python scripts/exp1_auroc.py --streams \
    "tent/${c}_rome32_open_seed0" \
    "bn_adapt/${c}_rome32_open_seed0" \
    "nova-tta/${c}_rome32_open_seed0"
done
