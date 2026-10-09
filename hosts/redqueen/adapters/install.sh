#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${source_root}/.." && pwd)"
readonly repo_root="$(cd -- "${host_root}/../.." && pwd)"
readonly target_root="/srv/ai/adapters"
readonly package_target="${target_root}/source/redqueen_adapters"
readonly workflow_target="${target_root}/workflows"
readonly venv="${target_root}/venv"
readonly user_unit_dir="${HOME}/.config/systemd/user"
readonly secret_dir=/srv/ai/secrets
readonly qwen_key_file="${secret_dir}/qwen38-api-keys"
readonly adapter_env_file="${secret_dir}/redqueen-adapters.env"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to install the adapters as root\n' >&2
  exit 1
fi
for command in python3 systemctl; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

install -d -m 0750 -- "${package_target}" "${workflow_target}"
install -m 0640 -- "${repo_root}/LICENSE" "${target_root}/LICENSE"
install -m 0640 -- "${source_root}/NOTICE" "${target_root}/NOTICE"
install -m 0640 -- "${source_root}/requirements.lock" "${target_root}/requirements.lock"
for source in "${source_root}"/redqueen_adapters/*.py; do
  install -m 0640 -- "${source}" "${package_target}/$(basename -- "${source}")"
done
for name in \
  qwen-image-2.1-t2i-smoke-api.json \
  qwen-image-2.1-transparency-smoke-api.json \
  qwen-image-2.1-edit-smoke-api.json \
  qwen-image-2.1-multiref-smoke-api.json; do
  install -m 0640 -- "${host_root}/comfyui/${name}" "${workflow_target}/${name}"
done

if [[ ! -x "${venv}/bin/python" ]]; then
  python3 -m venv "${venv}"
fi
"${venv}/bin/pip" install \
  --disable-pip-version-check \
  --no-input \
  --only-binary=:all: \
  --upgrade \
  --requirement "${source_root}/requirements.lock"
PYTHONPATH="${target_root}/source" "${venv}/bin/python" -m unittest discover \
  -s "${source_root}/tests" -v

if [[ -e "${adapter_env_file}" ]]; then
  [[ -f "${adapter_env_file}" && ! -L "${adapter_env_file}" && \
    -r "${adapter_env_file}" && -w "${adapter_env_file}" ]] || {
    printf 'adapter environment file must be a readable regular file owned by ai\n' >&2
    exit 1
  }
  [[ -r "${qwen_key_file}" && -s "${qwen_key_file}" ]] || {
    printf 'missing existing Qwen router credential: %s\n' "${qwen_key_file}" >&2
    exit 1
  }
  mapfile -t qwen_key_lines <"${qwen_key_file}"
  [[ "${#qwen_key_lines[@]}" -eq 1 && \
    "${qwen_key_lines[0]}" =~ ^[a-f0-9]{64}$ ]] || {
    printf 'Qwen router credential must contain one 64-character hex key\n' >&2
    exit 1
  }
  mapfile -t configured_router_keys < <(
    sed -n 's/^QWEN_API_KEY=//p' "${adapter_env_file}"
  )
  if [[ "${#configured_router_keys[@]}" -eq 0 ]]; then
    readonly adapter_env_partial="$(mktemp --tmpdir="${secret_dir}" .redqueen-adapters-env.XXXXXXXX)"
    trap 'rm -f -- "${adapter_env_partial}"' EXIT
    cat -- "${adapter_env_file}" >"${adapter_env_partial}"
    printf 'QWEN_API_KEY=%s\n' "${qwen_key_lines[0]}" >>"${adapter_env_partial}"
    chmod 0600 "${adapter_env_partial}"
    mv -- "${adapter_env_partial}" "${adapter_env_file}"
    trap - EXIT
  elif [[ "${#configured_router_keys[@]}" -ne 1 || \
    "${configured_router_keys[0]}" != "${qwen_key_lines[0]}" ]]; then
    printf 'adapter environment has a mismatched or duplicate QWEN_API_KEY; preserving it\n' >&2
    exit 1
  fi
fi

install -d -m 0755 -- "${user_unit_dir}"
for unit in qwen38.service qwen-image-adapter.service jevk5-adapter.service; do
  install -m 0644 -- "${host_root}/systemd/${unit}" "${user_unit_dir}/${unit}"
done
systemctl --user daemon-reload

printf 'adapter files and user units installed; credentials are intentionally not created\n'
