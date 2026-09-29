#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${REPO_ROOT}/scripts/generate_octree.py" \
  --config "${REPO_ROOT}/configs/inference/octllm.yaml" \
  --input_json '{"role":"user","content":"Generate a 3D mesh based on the following text description: a dolphin which has a long body and a short tail."}' \
  --system_prompt "You are a helpful assistant specialized in 3D asset generation and understanding, image understanding and chatting. When you are asked to generate a 3D asset from an image, you should reply in formats like: \"I've produced a 3D model based on the image: \", \"Based on the image, here's the 3D mesh asset I've created: \", etc. And if you are asked to generate a 3D asset from a text description, you should reply in formats like: \"I've produced a 3D mesh asset based on your description: \", \"Of course! I've generated a 3D mesh based on your text prompt: \", etc. If you are asked to describe a 3D asset, just describe it in detail. The text can be varied, but the colon \":\" must be included." \
  --temperature 0.5 \
  --top_p 0.9 \
  --top_k 40 \
  --bos_top_k 1 \
  --max_layer 6 \
  --full_depth 3 "$@"
  # --image_path ${REPO_ROOT}/imgs/image9.png \
  # --save_preprocessed_dir ./octllm_preprocessed \
  # --no_preprocess_image

# Image-conditioned inference example:
#   add --image_path /path/to/image.png and use an image-to-3D prompt.
