#!/usr/bin/env bash

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
MODE="${MODE:-image}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/inference/octllm.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/shapenet/${MODE}}"
BOS_TOP_K="${BOS_TOP_K:-2}"

KEEP_MASK_ARGS=()
if [ "${KEEP_MASK_IN_CACHE:-0}" = "1" ]; then
  KEEP_MASK_ARGS=(--keep_mask_in_cache)
fi

if [ "${MODE}" = "text" ]; then
  DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/shapenet/sft/llm_shapenet_test_airplane_description.json}"
  PROMPT="${PROMPT:-Generate a 3D asset based on the following description:}"
elif [ "${MODE}" = "image" ]; then
  DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/shapenet/sft/mllm_shapenet_test_airplane_image.json}"
  PROMPT="${PROMPT:-Generate a 3D asset from the following image:<image>}"
else
  DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/shapenet/sft/llm_shapenet_test_airplane_understanding.json}"
  PROMPT="${PROMPT:-Describe the 3D asset in detail:}"
fi

if [ -n "${GPU_IDS:-}" ]; then
  GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
  read -r -a GPU_ARR <<< "${GPU_IDS_NORMALIZED}"
  NUM_GPUS="${#GPU_ARR[@]}"
else
  GPU_ARR=()
  for ((SHARD_ID=0; SHARD_ID<NUM_GPUS; SHARD_ID++)); do
    GPU_ARR+=("${SHARD_ID}")
  done
fi

PIDS=()
echo "Launching ${NUM_GPUS} shards on GPUs: ${GPU_ARR[*]} (MODE=${MODE})"

for ((SHARD_ID=0; SHARD_ID<NUM_GPUS; SHARD_ID++)); do
  CUDA_VISIBLE_DEVICES="${GPU_ARR[SHARD_ID]}" python "${REPO_ROOT}/scripts/batch_generate_shapenet.py" \
    --config "${CONFIG_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt "${PROMPT}" \
    --system_prompt "You are a helpful assistant specialized in 3D asset generation and understanding, image understanding and chatting. When you are asked to generate a 3D asset from an image, you should reply in formats like: \"I've produced a 3D model based on the image: \", \"Based on the image, here's the 3D mesh asset I've created: \", etc. And if you are asked to generate a 3D asset from a text description, you should reply in formats like: \"I've produced a 3D mesh asset based on your description: \", \"Of course! I've generated a 3D mesh based on your text prompt: \", etc. If you are asked to describe a 3D asset, just describe it in detail. The text can be varied, but the colon \":\" must be included." \
    --temperature 0.5 \
    --top_p 0.9 \
    --top_k 40 \
    --bos_top_k "${BOS_TOP_K}" \
    --max_layer 6 \
    --full_depth 3 \
    --input_mode "${MODE}" \
    --num_shards "${NUM_GPUS}" \
    --shard_id "${SHARD_ID}" \
    "${KEEP_MASK_ARGS[@]}" "$@" &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
exit "$FAILED"
