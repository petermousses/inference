#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/versions.env"
readonly repository="${COMFYUI_GGUF_REPOSITORY:?missing pinned ComfyUI-GGUF repository}"
readonly revision="${COMFYUI_GGUF_REVISION:?missing pinned ComfyUI-GGUF revision}"
readonly gguf_requirement="${COMFYUI_GGUF_REQUIREMENT:?missing pinned gguf package}"
readonly custom_node_root=/srv/ai/comfyui/state/custom_nodes
readonly node_target="${custom_node_root}/ComfyUI-GGUF"
readonly comfy_venv=/srv/ai/comfyui/venv

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
for command in git install; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done
if [[ ! -x "${comfy_venv}/bin/python" ]]; then
  printf 'ComfyUI venv is missing: %s\n' "${comfy_venv}" >&2
  exit 1
fi
if [[ ! -d "${custom_node_root}" || ! -w "${custom_node_root}" ]]; then
  printf 'ComfyUI custom node directory must exist and be writable: %s\n' \
    "${custom_node_root}" >&2
  exit 1
fi
if [[ ! -s /srv/ai/models/qwen-image-2.1-uncensored/SHA256SUMS ]]; then
  printf 'uncensored Qwen Image checksum manifest is missing\n' >&2
  exit 1
fi
(cd /srv/ai/models/qwen-image-2.1-uncensored && sha256sum --check SHA256SUMS)

if [[ -e "${node_target}" ]]; then
  if [[ ! -d "${node_target}/.git" ]] \
    || [[ "$(git -C "${node_target}" rev-parse HEAD)" != "${revision}" ]] \
    || [[ -n "$(git -C "${node_target}" status --porcelain)" ]]; then
    printf 'existing ComfyUI-GGUF checkout differs from the pinned clean revision; preserve and inspect it\n' >&2
    exit 1
  fi
else
  readonly staging="$(mktemp -d "${custom_node_root}/.ComfyUI-GGUF.XXXXXXXX")"
  cleanup() {
    rm -r -- "${staging}"
  }
  trap cleanup EXIT
  git clone --filter=blob:none --no-checkout \
    "https://github.com/${repository}.git" "${staging}/checkout"
  git -C "${staging}/checkout" fetch --depth 1 origin "${revision}"
  git -C "${staging}/checkout" checkout --detach "${revision}"
  [[ "$(git -C "${staging}/checkout" rev-parse HEAD)" == "${revision}" ]]
  mv -- "${staging}/checkout" "${node_target}"
  rmdir -- "${staging}"
  trap - EXIT
fi

"${comfy_venv}/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --only-binary=:all: \
  --no-deps \
  "${gguf_requirement}"
"${comfy_venv}/bin/python" -c 'import gguf; print("GGUF loader dependency import OK")'

printf 'ComfyUI-GGUF %s installed with %s; no ComfyUI service was restarted\n' \
  "${revision}" "${gguf_requirement}"
printf 'after installing the updated model paths and unit, restart comfyui.service and verify the custom node loads\n'
