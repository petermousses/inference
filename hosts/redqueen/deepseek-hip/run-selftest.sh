#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/redqueen-ds4-hip.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

DS4_HIP_TEST_SUPPORT=1 BUILD_DIR="${build_dir}" "${root}/build.sh"
DS4_HIP_LIBRARY="${build_dir}/libredqueen_ds4_hip.so" \
  "${PYTHON:-python3}" -m unittest discover -s "${root}/tests" -p 'test_*.py' -v
