#!/usr/bin/env bash
# Auxiliary ShapeLLM baseline; provide the matching checkpoint configuration.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
: "${CONFIG:?Set CONFIG to the ShapeLLM baseline inference YAML.}"
: "${DATASET_PATH:?Set DATASET_PATH to the baseline evaluation JSON.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the baseline prediction directory.}"
NUM_GPUS="${NUM_GPUS:-8}"
MODE="${MODE:-image}"
PROMPT='Generate a 3D asset from the following image:<image>'
if [[ "${MODE}" == "text" ]]; then
  PROMPT='Describe the 3D asset in detail:'
fi
PIDS=()
for (( SHARD_ID=0; SHARD_ID<NUM_GPUS; SHARD_ID++ )); do
  CUDA_VISIBLE_DEVICES="${SHARD_ID}" python scripts/batch_predict_shapellm.py \
    --config "${CONFIG}" --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" --prompt "${PROMPT}" \
    --temperature 0.7 --top_p 0.7 --top_k 8192 \
    --input_mode "${MODE}" --num_shards "${NUM_GPUS}" --shard_id "${SHARD_ID}" "$@" &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "${PID}" || FAILED=1; done
exit "${FAILED}"
