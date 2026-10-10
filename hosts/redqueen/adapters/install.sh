#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${source_root}/.." && pwd)"
readonly target_root="/srv/ai/adapters"
readonly venv="${target_root}/venv"
readonly secret_dir=/srv/ai/secrets
readonly qwen_key_file="${secret_dir}/qwen38-api-keys"
readonly adapter_env_file="${secret_dir}/redqueen-adapters.env"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to install the adapters as root\n' >&2
  exit 1
fi
for command in openssl python3 systemctl; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

"${host_root}/link-runtime.sh"

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
    readonly qwen_env_partial="$(mktemp --tmpdir="${secret_dir}" .redqueen-adapters-env.XXXXXXXX)"
    trap 'rm -f -- "${qwen_env_partial}"' EXIT
    cat -- "${adapter_env_file}" >"${qwen_env_partial}"
    printf 'QWEN_API_KEY=%s\n' "${qwen_key_lines[0]}" >>"${qwen_env_partial}"
    chmod 0600 "${qwen_env_partial}"
    mv -- "${qwen_env_partial}" "${adapter_env_file}"
    trap - EXIT
  elif [[ "${#configured_router_keys[@]}" -ne 1 || \
    "${configured_router_keys[0]}" != "${qwen_key_lines[0]}" ]]; then
    printf 'adapter environment has a mismatched or duplicate QWEN_API_KEY; preserving it\n' >&2
    exit 1
  fi

  mapfile -t configured_clm_keys < <(
    sed -n 's/^CLM_API_KEY=//p' "${adapter_env_file}"
  )
  if [[ "${#configured_clm_keys[@]}" -eq 0 ]]; then
    clm_key="$(openssl rand -hex 32)"
    readonly clm_env_partial="$(mktemp --tmpdir="${secret_dir}" .redqueen-adapters-env.XXXXXXXX)"
    trap 'rm -f -- "${clm_env_partial}"' EXIT
    cat -- "${adapter_env_file}" >"${clm_env_partial}"
    printf 'CLM_API_KEY=%s\n' "${clm_key}" >>"${clm_env_partial}"
    chmod 0600 "${clm_env_partial}"
    mv -- "${clm_env_partial}" "${adapter_env_file}"
    trap - EXIT
  elif [[ "${#configured_clm_keys[@]}" -ne 1 || \
    ! "${configured_clm_keys[0]}" =~ ^[a-f0-9]{64}$ ]]; then
    printf 'adapter environment has an invalid or duplicate CLM_API_KEY; preserving it\n' >&2
    exit 1
  fi
fi

printf 'adapter files and user units installed; CLM_API_KEY created if the adapter environment lacked one\n'
