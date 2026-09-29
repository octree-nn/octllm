#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# `vae` selects the paper's bottleneck model without cross-scale skips.
exec torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
    "${REPO_ROOT}/scripts/train_completion.py" \
    --amp --save-best --scheduler cosine --grad-clip 1.0 \
    --max-epochs 50 --recon-loss bce --model-type vae \
    --checkpoint-every 20000000 "$@"
