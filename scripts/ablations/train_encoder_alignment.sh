#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
    "${REPO_ROOT}/scripts/ablations/train_encoder_alignment.py" \
    --amp --save-best --scheduler cosine --grad-clip 1.0 \
    --max-epochs 100 --batch-size 32 --checkpoint-every 20000000 "$@"
