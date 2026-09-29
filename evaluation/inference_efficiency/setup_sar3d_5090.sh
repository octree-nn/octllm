#!/usr/bin/env bash
set -euo pipefail

# Reuse an explicitly selected CUDA 12.8+ environment for the SAR3D baseline.
: "${SAR3D_SOURCE_ENV:?Set SAR3D_SOURCE_ENV to an existing CUDA 12.8+ environment.}"
SOURCE_ENV="${SAR3D_SOURCE_ENV}"
TARGET_ENV="${SAR3D_TARGET_ENV:-${PWD}/.venvs/sar3d-5090}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PINNED_SAR3D_SOURCE_DIR="${SCRIPT_DIR}/third_party/SAR3D"

if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
  echo "Source environment not found: ${SOURCE_ENV}" >&2
  exit 1
fi

# Prepare the small, pinned source checkout before validating its imports.  The
# benchmark proxy setting is used here; no model checkpoint is downloaded.
"${SOURCE_ENV}/bin/python" "${SCRIPT_DIR}/benchmark.py" prepare --methods sar3d

if [[ -x "${TARGET_ENV}/bin/python" ]]; then
  echo "Using existing SAR3D environment: ${TARGET_ENV}"
else
  if [[ -e "${TARGET_ENV}" ]]; then
    echo "Target exists but has no Python: ${TARGET_ENV}. Choose an empty SAR3D_TARGET_ENV." >&2
    exit 1
  fi
  echo "Creating lightweight venv at ${TARGET_ENV} (reuses ${SOURCE_ENV}; no 12GB copy)"
  "${SOURCE_ENV}/bin/python" -m venv --system-site-packages "${TARGET_ENV}"
fi

echo "Installing SAR3D Python extras into ${TARGET_ENV}"
"${TARGET_ENV}/bin/python" -m pip install \
  --disable-pip-version-check \
  --requirement "${SCRIPT_DIR}/requirements_sar3d_5090.txt"

"${TARGET_ENV}/bin/python" - <<'PY'
import torch
import transformers
import xformers
import beartype
import blobfile
import ipdb
import timm
import tap

if torch.version.cuda is None:
    raise RuntimeError("The reused PyTorch build has no CUDA runtime")
cuda_major_minor = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
if cuda_major_minor < (12, 8):
    raise RuntimeError(f"RTX 5090 requires a CUDA 12.8+ PyTorch build; found {torch.version.cuda}")
print(f"SAR3D environment ready: torch={torch.__version__}, CUDA={torch.version.cuda}, "
      f"transformers={transformers.__version__}, xformers={xformers.__version__}")
PY

(
  cd "${PINNED_SAR3D_SOURCE_DIR}"
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="${PINNED_SAR3D_SOURCE_DIR}" "${TARGET_ENV}/bin/python" - <<'PY'
from models import build_vae_var_3D_VAR
from models.basic_var import memory_efficient_attention

assert callable(build_vae_var_3D_VAR)
if memory_efficient_attention is None:
    raise RuntimeError("SAR3D FP32 inference requires xformers.ops.memory_efficient_attention")
print("SAR3D official model imports verified")
PY
)

echo "No model files were downloaded. The SAR3D benchmark environment is ready."
