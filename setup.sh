#!/usr/bin/env bash
# Install the pinned OctLLM inference environment. No model weights are downloaded.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="OctLLM"
RESUME=0
ENV_SET=0

usage() {
    cat <<'HELP'
Usage: bash setup.sh [--env NAME] [--resume]
       bash setup.sh [NAME] [--resume]

Create the OctLLM environment (Linux x86_64, Python 3.10, CUDA 12.1 wheels).
  --env NAME   Choose a conda environment name (default: OctLLM).
  --resume     Continue installation in an existing environment.
  -h, --help   Show this message.

Set CONDA_EXE if conda is not on PATH. Standard proxy/index environment
variables are respected. OCTLLM_SETUP_PROXY optionally overrides HTTPS_PROXY;
no local proxy is assumed. Model weights and Blender are installed separately.
HELP
}

while (( $# )); do
    case "$1" in
        --env)
            if (( $# < 2 )); then echo "--env needs a name." >&2; exit 2; fi
            ENV_NAME="$2"; ENV_SET=1; shift 2 ;;
        --resume) RESUME=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if (( ENV_SET )); then echo "Specify one environment name." >&2; exit 2; fi
            ENV_NAME="$1"; ENV_SET=1; shift ;;
    esac
done
if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
    echo "Use an environment name containing letters, digits, _, . or -." >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "The bundled CUDA wheels require Linux x86_64; see README.md#installation." >&2
    exit 2
fi

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "Conda was not found. Initialize conda or set CONDA_EXE." >&2
    exit 1
fi
if [[ -n "${OCTLLM_SETUP_PROXY:-}" ]]; then
    export HTTPS_PROXY="${OCTLLM_SETUP_PROXY}"
    export https_proxy="${OCTLLM_SETUP_PROXY}"
fi

ENV_EXISTS=0
ENV_LIST="$("${CONDA_BIN}" env list)"
while IFS= read -r env_line; do
    if [[ "${env_line%% *}" == "$ENV_NAME" ]]; then ENV_EXISTS=1; fi
done <<< "${ENV_LIST}"
if (( ENV_EXISTS && ! RESUME )); then
    echo "Environment '${ENV_NAME}' exists. Use --resume to update it explicitly." >&2
    exit 2
fi
if (( ! ENV_EXISTS )); then
    "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.10 pip -y
fi
run_python() { "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python "$@"; }
run_python -c 'import sys; assert sys.version_info[:2] == (3, 10), "This installer requires Python 3.10."'
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
export PIP_RETRIES="${PIP_RETRIES:-3}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT
cat > "${CONSTRAINTS}" <<'PINS'
torch==2.4.0
torchvision==0.19.0
xformers==0.0.27.post2
PINS
# Reuse the version pins for both installs; exclude direct wheel/Git URLs.
awk '/^[[:alnum:]_.-]+==/ {print}' "${REPO_ROOT}/requirements_inference.txt" >> "${CONSTRAINTS}"
run_python -m pip install --upgrade pip setuptools wheel
run_python -m pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu121

# These distributions share module paths with their GPU counterparts.
if run_python -m pip show open3d-cpu >/dev/null 2>&1; then
    run_python -m pip uninstall -y open3d-cpu open3d
fi
if run_python -m pip show onnxruntime >/dev/null 2>&1; then
    run_python -m pip uninstall -y onnxruntime onnxruntime-gpu
fi
run_python -m pip install -c "${CONSTRAINTS}" -r "${REPO_ROOT}/requirements_inference.txt"
run_python -m pip install -c "${CONSTRAINTS}" -e "${REPO_ROOT}"
run_python -m pip check
run_python -c '
import torch
import open3d as o3d
import onnxruntime as ort
assert torch.version.cuda == "12.1", torch.version.cuda
assert o3d._build_config["BUILD_CUDA_MODULE"], "Open3D has no CUDA module"
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print("CUDA-enabled distributions: OK; device availability:", torch.cuda.is_available())
'
run_python "${REPO_ROOT}/inference.py" --help
printf '\nEnvironment ready. Run: conda activate %s\n' "${ENV_NAME}"
