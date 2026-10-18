#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/redqueen-ds4-hip.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

BUILD_DIR="${build_dir}" "${root}/build.sh"
DS4_HIP_LIBRARY="${build_dir}/libredqueen_ds4_quantization.so" \
  "${PYTHON:-python3}" "${root}/tests/test_quantization.py"
