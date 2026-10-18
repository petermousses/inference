#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm}"
build_dir="${BUILD_DIR:-${root}/build}"

mkdir -p "${build_dir}"
"${rocm_path}/bin/hipcc" \
  --offload-arch=gfx1151 \
  -std=c++17 \
  -O3 \
  -fPIC \
  -shared \
  "${root}/src/quantization.hip" \
  -o "${build_dir}/libredqueen_ds4_quantization.so"
