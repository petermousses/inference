#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${source_root}/.." && pwd)"
readonly target_root=/srv/ai/model-router
readonly binary_root=/srv/ai/bin
readonly binary_target="${binary_root}/llama-swap"
readonly release_url="https://github.com/mostlygeek/llama-swap/releases/download/v260/llama-swap_260_linux_amd64.tar.gz"
readonly release_sha256=d856a908507560cbdc253300bcf49092c7ead3c85098687428b0c9d4832ff46d
readonly release_version=v260

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
"${host_root}/link-runtime.sh"
bash "${host_root}/model-router/create-venv.sh"
"${source_root}/install-glm53-runtime.sh"
"${source_root}/install-embeddinggemma2-runtime.sh"
"${source_root}/install-moe-runtime.sh"
"${source_root}/install-gemma4-moe-runtime.sh"
"${source_root}/install-deepseek41-runtime.sh"
for command in curl sha256sum tar cmp install systemctl python3; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

if [[ ! -s /srv/ai/secrets/llama-swap.env ]]; then
  printf 'router environment file is missing: %s\n' /srv/ai/secrets/llama-swap.env >&2
  exit 1
fi

for model_id in \
  qwen3.8-27b \
  gemma-4-e4b-it \
  gemma-4-12b-it \
  gemma-4-26b-a4b-it \
  qwen3.6-35b-a3b \
  deepseek-v4.1-flash-1017-unsloth \
  huihui-glm-5.3-flash-abliterated-gguf \
  embeddinggemma-2 \
  clef-flash \
  clef; do
  manifest="/srv/ai/models/${model_id}/SHA256SUMS"
  if [[ ! -s "${manifest}" ]]; then
    printf 'model checksum manifest is missing: %s\n' "${manifest}" >&2
    exit 1
  fi
  printf 'verifying model files for %s\n' "${model_id}"
  (cd -- "/srv/ai/models/${model_id}" && sha256sum --check SHA256SUMS)
done

readonly temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/llama-swap-install.XXXXXXXX")"
cleanup() {
  rm -r -- "${temporary_root}"
}
trap cleanup EXIT
curl --fail --location --retry 3 --connect-timeout 20 \
  --output "${temporary_root}/llama-swap.tar.gz" "${release_url}"
printf '%s  %s\n' "${release_sha256}" "${temporary_root}/llama-swap.tar.gz" \
  | sha256sum --check
tar -xzf "${temporary_root}/llama-swap.tar.gz" \
  -C "${temporary_root}" llama-swap LICENSE.md
chmod 0755 "${temporary_root}/llama-swap"

if [[ -e "${binary_target}" ]]; then
  if ! cmp -s -- "${temporary_root}/llama-swap" "${binary_target}"; then
    printf 'an existing llama-swap binary differs from pinned %s; preserve it and review before replacing\n' \
      "${release_version}" >&2
    exit 1
  fi
else
  install -d -m 0755 -- "${binary_root}"
  install -m 0755 -- "${temporary_root}/llama-swap" "${binary_target}"
fi

if [[ -e "${target_root}/THIRD-PARTY-LICENSE.md" ]] && \
  ! cmp -s -- "${temporary_root}/LICENSE.md" "${target_root}/THIRD-PARTY-LICENSE.md"; then
  printf 'existing router license differs from pinned %s\n' "${release_version}" >&2
  exit 1
fi

QWEN_API_KEY=validation-only "${binary_target}" \
  -config "${source_root}/config.yaml" -validate

install -d -m 0750 -- "${target_root}" "${target_root}/bin" "${target_root}/run" \
  "${HOME}/.config/systemd/user"
if [[ ! -e "${target_root}/run/comfyui-paused-by-glm" ]]; then
  install -m 0600 /dev/null "${target_root}/run/comfyui-paused-by-glm"
fi

printf 'llama-swap %s installed and config validated; no service was enabled, started, stopped or restarted\n' \
  "${release_version}"
printf 'restricted ComfyUI-control helper installed; no service was enabled, started, stopped or restarted\n'
