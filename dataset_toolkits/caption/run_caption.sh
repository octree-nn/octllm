#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${TOOLS_DIR}/../.." && pwd)"
QWEN35_ENV_NAME="${QWEN35_ENV_NAME:-qwen35-caption}"
GPU_IDS="${GPU_IDS:-0}"
: "${ROOT_DIR:?Set ROOT_DIR to the directory of per-asset condition renders.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the caption output directory.}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_DIR}/caption_inputs.json}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/caption/qwen35.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN35_ENV_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${DATASET_PATH}" ]]; then
    python "${TOOLS_DIR}/prepare_caption_inputs.py" --root-dir "${ROOT_DIR}" --output-path "${DATASET_PATH}"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
PIDS=()
for SHARD_INDEX in "${!GPU_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[SHARD_INDEX]}" python "${TOOLS_DIR}/generate_captions.py" \
        --config "${CONFIG_PATH}" --dataset-path "${DATASET_PATH}" --output-dir "${OUTPUT_DIR}" \
        --num-shards "${#GPU_ARRAY[@]}" --shard-id "${SHARD_INDEX}" --skip-existing "$@" &
    PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
if (( FAILED )); then exit 1; fi
python "${TOOLS_DIR}/export_captions.py" --input-dir "${OUTPUT_DIR}" --output-csv "${OUTPUT_DIR}/captions.csv"
