#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
inference_dir="${DS4_INFERENCE_DIR:-/srv/ai/models/deepseek-v4.1-flash-1017/inference}"
if [[ ! -f "${inference_dir}/model.py" ]]; then
  echo "official DeepSeek inference/model.py not found: ${inference_dir}/model.py" >&2
  exit 1
fi

export DS4_INFERENCE_DIR="${inference_dir}"
export PYTHONPATH="${root}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" "${root}/python/run_reference_smoke.py"
