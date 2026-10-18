#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm}"
build_dir="${BUILD_DIR:-${root}/build}"
sources=(
  "${root}/src/quantization.hip"
  "${root}/src/fp8_gemm.hip"
  "${root}/src/rmsnorm.hip"
  "${root}/src/sparse_attn.hip"
  "${root}/src/rope.hip"
)

if [[ "${DS4_HIP_TEST_SUPPORT:-0}" == "1" ]]; then
  sources+=("${root}/tests/rmsnorm_test_support.hip")
fi

mkdir -p "${build_dir}"
"${rocm_path}/bin/hipcc" \
  --offload-arch=gfx1151 \
  -std=c++17 \
  -O3 \
  -ffp-contract=off \
  -fPIC \
  -shared \
  "${sources[@]}" \
  -o "${build_dir}/libredqueen_ds4_hip.so"
