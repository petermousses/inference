#!/usr/bin/env bash
set -euo pipefail
umask 0077

readonly secret_dir=/srv/ai/secrets
readonly key_file="${secret_dir}/qwen38-api-keys"
readonly env_file="${secret_dir}/llama-swap.env"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
if [[ ! -r "${key_file}" || ! -s "${key_file}" ]]; then
  printf 'missing or unreadable existing Qwen credential: %s\n' "${key_file}" >&2
  exit 1
fi
if [[ -e "${env_file}" ]]; then
  printf 'refusing to overwrite existing router environment file: %s\n' \
    "${env_file}" >&2
  exit 1
fi

mapfile -t key_lines <"${key_file}"
if [[ "${#key_lines[@]}" -ne 1 || ! "${key_lines[0]}" =~ ^[a-f0-9]{64}$ ]]; then
  printf 'existing Qwen credential must contain exactly one 64-character hex key\n' >&2
  exit 1
fi

temporary_file="$(mktemp --tmpdir="${secret_dir}" .llama-swap-env.XXXXXXXX)"
cleanup() {
  rm -f -- "${temporary_file}"
}
trap cleanup EXIT
chmod 0600 "${temporary_file}"
printf 'QWEN_API_KEY=%s\n' "${key_lines[0]}" >"${temporary_file}"
if ! ln -- "${temporary_file}" "${env_file}"; then
  printf 'could not create router environment file without replacing an existing path\n' >&2
  exit 1
fi
printf 'created %s using the existing Qwen key; the value was not printed\n' \
  "${env_file}"
