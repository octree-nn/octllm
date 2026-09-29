#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS:-0}"
PIDS=()
for rank in "${!GPU_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[rank]}" \
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/decode_octree.py" \
        --rank "$rank" --world-size "${#GPU_ARRAY[@]}" "$@" &
    PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
exit "$FAILED"
