#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Configure each model's interpreter in benchmark_config.json before running.
exec "${BENCHMARK_PYTHON:-python}" "${SCRIPT_DIR}/benchmark.py" all "$@"
