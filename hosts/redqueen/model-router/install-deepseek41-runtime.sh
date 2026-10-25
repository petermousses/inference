#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly revision=d81235049384534c167caea52b85a694f6103d14
readonly runtime=llama.cpp-d812350-deepseek41-moe-v17
readonly target="/srv/ai/runtimes/${runtime}"
readonly repository=https://github.com/ggml-org/llama.cpp.git
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly runtime_metadata="${script_dir}/runtime-manifests/${runtime}"
readonly patch_file="${runtime_metadata}/moe-stream.patch"
readonly architecture_patch_file="${runtime_metadata}/deepseek41.patch"
readonly direct_patch_file="${runtime_metadata}/deepseek41-direct.patch"
readonly vision_patch_file="${runtime_metadata}/deepseek41v.patch"
readonly adaptive_patch_file="${runtime_metadata}/adaptive-cache.patch"
readonly capacity_patch_file="${runtime_metadata}/capacity-sizing.patch"
(cd -- "${runtime_metadata}" && sha256sum --check PATCH_SHA256)
readonly patch_sha="$(sha256sum "${runtime_metadata}/PATCH_SHA256" | cut -d ' ' -f 1)"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
if [[ -e "${target}" ]]; then
  target_revision="$(awk -F= '$1 == "revision" { print substr($0, index($0, "=") + 1) }' "${target}/SOURCE_REVISION" 2>/dev/null || true)"
  if [[ ! -x "${target}/bin/llama-server" || \
    "${target_revision}" != "${revision}" || \
    "$(cat "${target}/PATCH_SHA256" 2>/dev/null || true)" != "$(cat "${runtime_metadata}/PATCH_SHA256")" ]]; then
    printf 'existing runtime differs from pinned %s; preserve it and review before replacing\n' \
      "${revision}" >&2
    exit 1
  fi
  if [[ ! -x "${target}/bin/llama-fit-params" ]]; then
    printf 'pinned runtime lacks the scheduler memory-fit planner\n' >&2
    exit 1
  fi
  fit_help="$("${target}/bin/llama-fit-params" --help 2>&1)"
  if [[ "${fit_help}" != *"--moe-stream"* || "${fit_help}" != *"--moe-stream-cache-max"* || "${fit_help}" != *"--fit-print"* || "${fit_help}" != *"on-direct"* ]]; then
    printf 'pinned memory-fit planner lacks adaptive MoE streaming or direct Engram support\n' >&2
    exit 1
  fi
  server_help="$("${target}/bin/llama-server" --help 2>&1)"
  if [[ "${server_help}" != *"--moe-stream-cache-max"* ]]; then
    printf 'pinned llama-server lacks adaptive MoE streaming support\n' >&2
    exit 1
  fi
  if [[ "${server_help}" != *"on-direct"* ]]; then
    printf 'pinned llama-server lacks direct Engram row support\n' >&2
    exit 1
  fi
  if [[ "${server_help}" != *"--mmproj"* || "${server_help}" != *"--image-max-tokens"* ]]; then
    printf 'pinned llama-server lacks multimodal image support\n' >&2
    exit 1
  fi
  runtime_version="$("${target}/bin/llama-server" --version 2>&1)"
  if [[ "${runtime_version}" != *"commit ${revision:0:7}"* ]]; then
    printf 'installed llama-server does not report the pinned commit\n' >&2
    exit 1
  fi
  missing_libraries="$(ldd "${target}/bin/llama-server" | sed -n '/not found/p')"
  if [[ -n "${missing_libraries}" ]]; then
    printf '%s\n' "${missing_libraries}" >&2
    exit 1
  fi
  available_devices="$("${target}/bin/llama-server" --list-devices)"
  if [[ "${available_devices}" != *'Radeon 8060S Graphics'* ]]; then
    printf 'pinned llama-server does not enumerate the expected Radeon device\n' >&2
    exit 1
  fi
  (cd -- "${target}" && sha256sum --check --quiet RUNTIME_SHA256SUMS)
  printf 'DeepSeek-V4.1 MoE runtime already installed and verified\n'
  exit 0
fi

for command in apt-get dpkg-deb git ninja c++ hipcc hipconfig ldd install mv python3 sha256sum find sort xargs; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done
kernel_release="$(uname -r)"
if [[ "$(printf '5.14\n%s\n' "${kernel_release}" | sort -V | head -n 1)" != '5.14' ]]; then
  printf 'Linux 5.14 or newer is required for checked MADV_POPULATE_WRITE staging commitment\n' >&2
  exit 1
fi
if ! printf '#include <sys/mman.h>\n#ifndef MADV_POPULATE_WRITE\n#error MADV_POPULATE_WRITE is unavailable in the system headers\n#endif\nint main() { return 0; }\n' | \
  "${CXX:-c++}" -x c++ -fsyntax-only -; then
  printf 'system C++ headers must define MADV_POPULATE_WRITE for checked staging commitment\n' >&2
  exit 1
fi

readonly temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/moe-runtime.XXXXXXXX")"
stage=''
cleanup() {
  if [[ -n "${stage}" && -d "${stage}" ]]; then
    rm -r -- "${stage}"
  fi
  rm -r -- "${temporary_root}"
}
trap cleanup EXIT

cmake_bin="$(command -v cmake || true)"
cmake_version=''
if [[ -n "${cmake_bin}" ]]; then
  cmake_version="$(${cmake_bin} --version 2>/dev/null | sed -n '1s/.*version //p' || true)"
fi
if [[ -z "${cmake_version}" ]] || \
  [[ "$(printf '3.31.6\n%s\n' "${cmake_version}" | sort -V | head -n 1)" != '3.31.6' ]]; then
  readonly cmake_root="${temporary_root}/cmake-root"
  install -d -m 0755 -- "${temporary_root}/cmake-debs" "${cmake_root}"
  (
    cd -- "${temporary_root}/cmake-debs"
    apt-get download \
      cmake=3.31.6-2 cmake-data=3.31.6-2 \
      librhash1=1.4.5-1 libjsoncpp26=1.9.6-3
  )
  for package in "${temporary_root}"/cmake-debs/*.deb; do
    dpkg-deb -x "${package}" "${cmake_root}"
  done
  cmake_bin="${cmake_root}/usr/bin/cmake"
  export LD_LIBRARY_PATH="${cmake_root}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ "$("${cmake_bin}" --version | sed -n '1s/.*version //p')" != '3.31.6' ]] && \
  [[ "$(printf '3.31.6\n%s\n' "$("${cmake_bin}" --version | sed -n '1s/.*version //p')" | sort -V | head -n 1)" != '3.31.6' ]]; then
  printf 'CMake 3.31.6 or newer is required\n' >&2
  exit 1
fi
ctest_bin="$(dirname -- "${cmake_bin}")/ctest"
if [[ ! -x "${ctest_bin}" ]]; then
  printf 'CTest is required to run the live-resize regression\n' >&2
  exit 1
fi

source_root="${temporary_root}/source"
build_root="${temporary_root}/build"
git init --quiet "${source_root}"
git -C "${source_root}" remote add origin "${repository}"
git -C "${source_root}" fetch --quiet --depth 1 origin \
  "${revision}"
git -C "${source_root}" checkout --quiet --detach "${revision}"
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${revision}" ]]; then
  printf 'fetched llama.cpp revision does not match the pinned commit\n' >&2
  exit 1
fi

git -C "${source_root}" apply --check "${patch_file}"
git -C "${source_root}" apply "${patch_file}"
git -C "${source_root}" apply --check "${architecture_patch_file}"
git -C "${source_root}" apply "${architecture_patch_file}"
git -C "${source_root}" apply --check "${direct_patch_file}"
git -C "${source_root}" apply "${direct_patch_file}"
git -C "${source_root}" apply --check "${vision_patch_file}"
git -C "${source_root}" apply "${vision_patch_file}"
git -C "${source_root}" apply --check "${adaptive_patch_file}"
git -C "${source_root}" apply "${adaptive_patch_file}"
git -C "${source_root}" apply --check "${capacity_patch_file}"
git -C "${source_root}" apply "${capacity_patch_file}"

export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
readonly build_jobs="${BUILD_JOBS:-8}"
if [[ ! "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BUILD_JOBS must be a positive integer\n' >&2
  exit 1
fi
readonly install_rpath='$ORIGIN/../lib;/opt/rocm/lib;/opt/rocm/core-7.14/lib'
"${cmake_bin}" -S "${source_root}" -B "${build_root}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_RPATH="${install_rpath}" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_MTMD=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF
"${cmake_bin}" --build "${build_root}" --parallel "${build_jobs}"
"${ctest_bin}" --test-dir "${build_root}" --no-tests=error --output-on-failure -R '^test-moe-stream$'

install -d -m 0755 -- /srv/ai/runtimes
stage="$(mktemp -d /srv/ai/runtimes/.${runtime}.XXXXXXXX)"
"${cmake_bin}" --install "${build_root}" --prefix "${stage}"
# Install the public headers needed by the parity fixture.
install -d "${stage}/include"
install -m 0644 "${source_root}/include/llama.h" "${source_root}"/ggml/include/*.h "${stage}/include/"
"${CXX:-c++}" -std=c++17 -O2 \
  -I "${stage}/include" -L "${stage}/lib" \
  -Wl,-rpath,'$ORIGIN/../lib' -Wl,-rpath-link,"${stage}/lib" \
  "${script_dir}/qualification/moe-stream-probe.cpp" \
  -o "${stage}/bin/moe-stream-probe" "${stage}/lib/libllama.so" "${stage}/lib/libggml.so"
install -m 0644 "${runtime_metadata}/SOURCE_REVISION" "${runtime_metadata}/BUILD_OPTIONS" "${runtime_metadata}/PATCH_SHA256" "${stage}/"
"${CXX:-c++}" -std=c++17 -O2 \
  -I "${source_root}/src" -I "${source_root}/include" -I "${source_root}/ggml/include" \
  -L "${stage}/lib" -Wl,-rpath,"${stage}/lib" -Wl,-rpath-link,"${stage}/lib" \
  "${script_dir}/qualification/test-lazy-reader.cpp" \
  -o "${temporary_root}/test-lazy-reader" \
  "${stage}/lib/libllama.so" "${stage}/lib/libggml.so" "${stage}/lib/libggml-base.so"
"${temporary_root}/test-lazy-reader"
python3 - "${stage}" "${revision}" "${patch_sha}" <<'PYMETA'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])/"EXPERT_STREAMING.json"
p.write_text(json.dumps({"version":1,"base_revision":sys.argv[2],"patch_sha256":sys.argv[3],"max_contexts":1,"max_ubatch":1,"direct_io_required":True,"live_resize":True},sort_keys=True)+"\n")
PYMETA
runtime_version="$("${stage}/bin/llama-server" --version 2>&1)"
if [[ "${runtime_version}" != *"commit ${revision:0:7}"* ]]; then
  printf 'built llama-server does not report the pinned commit\n' >&2
  exit 1
fi
missing_libraries="$(ldd "${stage}/bin/llama-server" | sed -n '/not found/p')"
if [[ -n "${missing_libraries}" ]]; then
  printf '%s\n' "${missing_libraries}" >&2
  exit 1
fi
missing_probe_libraries="$(ldd "${stage}/bin/moe-stream-probe" | sed -n '/not found/p')"
if [[ -n "${missing_probe_libraries}" ]]; then
  printf '%s\n' "${missing_probe_libraries}" >&2
  exit 1
fi
if [[ ! -x "${stage}/bin/llama-fit-params" ]]; then
  printf 'built runtime lacks the scheduler memory-fit planner\n' >&2
  exit 1
fi
fit_help="$("${stage}/bin/llama-fit-params" --help 2>&1)"
if [[ "${fit_help}" != *"--moe-stream"* || "${fit_help}" != *"--moe-stream-cache-max"* || "${fit_help}" != *"--fit-print"* || "${fit_help}" != *"on-direct"* ]]; then
  printf 'built memory-fit planner lacks adaptive MoE streaming or direct Engram support\n' >&2
  exit 1
fi
server_help="$("${stage}/bin/llama-server" --help 2>&1)"
if [[ "${server_help}" != *"--moe-stream-cache-max"* ]]; then
  printf 'built llama-server lacks adaptive MoE streaming support\n' >&2
  exit 1
fi
if [[ "${server_help}" != *"on-direct"* ]]; then
  printf 'built llama-server lacks direct Engram row support\n' >&2
  exit 1
fi
if [[ "${server_help}" != *"--mmproj"* || "${server_help}" != *"--image-max-tokens"* ]]; then
  printf 'built llama-server lacks multimodal image support\n' >&2
  exit 1
fi
missing_fit_libraries="$(ldd "${stage}/bin/llama-fit-params" | sed -n '/not found/p')"
if [[ -n "${missing_fit_libraries}" ]]; then
  printf '%s\n' "${missing_fit_libraries}" >&2
  exit 1
fi
available_devices="$("${stage}/bin/llama-server" --list-devices)"
if [[ "${available_devices}" != *'Radeon 8060S Graphics'* ]]; then
  printf 'built llama-server does not enumerate the expected Radeon device\n' >&2
  exit 1
fi
(cd -- "${stage}" && find bin lib -type f -print0 | sort -z | xargs -0 sha256sum > RUNTIME_SHA256SUMS)
mv --no-target-directory --no-clobber -- "${stage}" "${target}"
if [[ -e "${stage}" ]]; then
  printf 'runtime target appeared during installation; preserve both paths and review\n' >&2
  exit 1
fi
stage=''
printf 'installed DeepSeek-V4.1 MoE runtime at %s\n' "${target}"
