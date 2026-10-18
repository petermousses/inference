#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="${root}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON:-python3}" "${root}/python/run_engram_row_reference_smoke.py" "$@"
