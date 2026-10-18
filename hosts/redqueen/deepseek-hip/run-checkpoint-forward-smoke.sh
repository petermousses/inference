#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DS4_HIP_LIBRARY:-}" ]]; then
  "${root}/build.sh"
  export DS4_HIP_LIBRARY="${BUILD_DIR:-${root}/build}/libredqueen_ds4_hip.so"
elif [[ ! -f "${DS4_HIP_LIBRARY}" ]]; then
  echo "native DeepSeek HIP library not found: ${DS4_HIP_LIBRARY}" >&2
  exit 1
fi

export PYTHONPATH="${root}/python${PYTHONPATH:+:${PYTHONPATH}}"
python="${PYTHON:-/srv/ai/comfyui/venv/bin/python}"
if [[ ! -x "${python}" ]]; then
  echo "DeepSeek smoke Python is unavailable: ${python}; set PYTHON to a ROCm Python with transformers" >&2
  exit 1
fi
exec "${python}" "${root}/python/run_dense_checkpoint_smoke.py" --forward-smoke "$@"
