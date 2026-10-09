#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly revision=d81235049384534c167caea52b85a694f6103d14
readonly version=0.6.0
readonly target="/srv/ai/runtimes/llama.cpp-v${version}-d812350"
readonly repository=https://github.com/ggml-org/llama.cpp.git

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
if [[ -e "${target}" ]]; then
  if [[ ! -x "${target}/bin/llama-server" || \
    "$(cat "${target}/SOURCE_REVISION" 2>/dev/null || true)" != "${revision}" ]]; then
    printf 'existing runtime differs from pinned %s; preserve it and review before replacing\n' \
      "${revision}" >&2
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
  printf 'pinned llama.cpp v%s runtime already installed and verified\n' "${version}"
  exit 0
fi

for command in apt-get dpkg-deb git ninja c++ hipcc hipconfig ldd install mv; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

readonly temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/glm53-runtime.XXXXXXXX")"
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

source_root="${temporary_root}/source"
build_root="${temporary_root}/build"
git init --quiet "${source_root}"
git -C "${source_root}" remote add origin "${repository}"
git -C "${source_root}" fetch --quiet --depth 1 origin \
  refs/tags/v0.6.0:refs/tags/v0.6.0
git -C "${source_root}" checkout --quiet --detach refs/tags/v0.6.0
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${revision}" ]]; then
  printf 'fetched llama.cpp revision does not match the pinned commit\n' >&2
  exit 1
fi

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
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_UI=OFF
"${cmake_bin}" --build "${build_root}" --parallel "${build_jobs}"

install -d -m 0755 -- /srv/ai/runtimes
stage="$(mktemp -d /srv/ai/runtimes/.llama.cpp-v${version}-d812350.XXXXXXXX)"
"${cmake_bin}" --install "${build_root}" --prefix "${stage}"
printf '%s\n' "${revision}" >"${stage}/SOURCE_REVISION"
printf '%s\n' 'GGML_HIP=ON AMDGPU_TARGETS=gfx1151' >"${stage}/BUILD_OPTIONS"
chmod 0644 "${stage}/SOURCE_REVISION" "${stage}/BUILD_OPTIONS"
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
available_devices="$("${stage}/bin/llama-server" --list-devices)"
if [[ "${available_devices}" != *'Radeon 8060S Graphics'* ]]; then
  printf 'built llama-server does not enumerate the expected Radeon device\n' >&2
  exit 1
fi
mv --no-target-directory --no-clobber -- "${stage}" "${target}"
if [[ -e "${stage}" ]]; then
  printf 'runtime target appeared during installation; preserve both paths and review\n' >&2
  exit 1
fi
stage=''
printf 'installed llama.cpp v%s at %s\n' "${version}" "${target}"
