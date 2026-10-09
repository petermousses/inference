#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${source_root}/.." && pwd)"
readonly repo_root="$(cd -- "${host_root}/../.." && pwd)"
readonly target_root=/srv/ai/model-router
readonly binary_root=/srv/ai/bin
readonly binary_target="${binary_root}/llama-swap"
readonly unit_target="${HOME}/.config/systemd/user/llama-swap.service"
readonly control_unit_target="${HOME}/.config/systemd/user/comfyui-control.service"
readonly previous_config_sha256=f4b50d096f7a6ef01a643e1ca2f643ff8d05f9adf7fcd57d9b695568cd14a55e
readonly previous_unit_sha256=4d74f70b0adffc7ed9705857f0733f96ddcb2385bf90904fcf27142fd991904f
readonly previous_control_unit_sha256=e6c186ace1b81439b9da397a11065da2ec323c5603a5baebc9e08edb054b9866
readonly release_url="https://github.com/mostlygeek/llama-swap/releases/download/v260/llama-swap_260_linux_amd64.tar.gz"
readonly release_sha256=d856a908507560cbdc253300bcf49092c7ead3c85098687428b0c9d4832ff46d
readonly release_version=v260

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
"${source_root}/install-glm53-runtime.sh"
"${source_root}/install-embeddinggemma2-runtime.sh"
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

if [[ -e "${target_root}/config.yaml" ]] && \
  ! cmp -s -- "${source_root}/config.yaml" "${target_root}/config.yaml"; then
  read -r installed_config_sha256 _ < <(sha256sum -- "${target_root}/config.yaml")
  if [[ "${installed_config_sha256}" != "${previous_config_sha256}" ]]; then
    printf 'existing router config differs from this and the recognized prior release; preserve it and review\n' >&2
    exit 1
  fi
fi
if [[ -e "${unit_target}" ]] && \
  ! cmp -s -- "${host_root}/systemd/llama-swap.service" "${unit_target}"; then
  read -r installed_unit_sha256 _ < <(sha256sum -- "${unit_target}")
  if [[ "${installed_unit_sha256}" != "${previous_unit_sha256}" ]]; then
    printf 'existing llama-swap unit differs from this and the recognized prior release; preserve it and review\n' >&2
    exit 1
  fi
fi
if [[ -e "${control_unit_target}" ]] && \
  ! cmp -s -- "${host_root}/systemd/comfyui-control.service" "${control_unit_target}"; then
  read -r installed_control_unit_sha256 _ < <(sha256sum -- "${control_unit_target}")
  if [[ "${installed_control_unit_sha256}" != "${previous_control_unit_sha256}" ]]; then
    printf 'existing ComfyUI-control unit differs from this and the recognized prior release; preserve it and review\n' >&2
    exit 1
  fi
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
install -m 0640 -- "${source_root}/config.yaml" "${target_root}/config.yaml"
install -m 0750 -- "${source_root}/glm-5.3-flash-abliterated-launcher.sh" \
  "${target_root}/bin/glm-5.3-flash-abliterated-launcher.sh"
install -m 0750 -- "${source_root}/comfyui-control.py" \
  "${target_root}/bin/comfyui-control.py"
install -m 0640 -- "${temporary_root}/LICENSE.md" \
  "${target_root}/THIRD-PARTY-LICENSE.md"
install -m 0644 -- "${host_root}/systemd/llama-swap.service" "${unit_target}"
install -m 0644 -- "${host_root}/systemd/comfyui-control.service" "${control_unit_target}"
systemctl --user daemon-reload

printf 'llama-swap %s installed and config validated; no service was enabled, started, stopped or restarted\n' \
  "${release_version}"
printf 'restricted ComfyUI-control helper installed; no service was enabled, started, stopped or restarted\n'
