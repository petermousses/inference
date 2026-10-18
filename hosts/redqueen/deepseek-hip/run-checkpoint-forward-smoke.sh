#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${DS4_HIP_LIBRARY:-}" && ! -f "${DS4_HIP_LIBRARY}" ]]; then
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
