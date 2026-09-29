#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen35-caption}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_PATH="${SCRIPT_DIR}/requirements.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install --upgrade -r "${REQUIREMENTS_PATH}"

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
  python -m pip install --upgrade flash-attn --no-build-isolation
fi

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
