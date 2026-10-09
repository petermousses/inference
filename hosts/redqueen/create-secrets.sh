#!/usr/bin/env bash
set -euo pipefail
umask 0077

readonly secret_dir=/srv/ai/secrets
readonly qwen_file="${secret_dir}/qwen38-api-keys"
readonly adapter_file="${secret_dir}/redqueen-adapters.env"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
if [[ ! -d "${secret_dir}" || ! -w "${secret_dir}" ]]; then
  printf '%s must exist and be writable by ai\n' "${secret_dir}" >&2
  exit 1
fi
if [[ -e "${qwen_file}" || -e "${adapter_file}" ]]; then
  printf 'refusing to overwrite existing redqueen credentials\n' >&2
  exit 1
fi
if ! command -v openssl >/dev/null; then
  printf 'openssl is required\n' >&2
  exit 1
fi

readonly temporary_dir="$(mktemp -d --tmpdir="${secret_dir}" .bootstrap.XXXXXXXX)"
cleanup() {
  rm -f -- "${temporary_dir}/qwen38-api-keys" "${temporary_dir}/redqueen-adapters.env"
  rmdir -- "${temporary_dir}" 2>/dev/null || true
}
trap cleanup EXIT

readonly qwen_key="$(openssl rand -hex 32)"
readonly image_key="$(openssl rand -hex 32)"
readonly jev_key="$(openssl rand -hex 32)"

printf '%s\n' "${qwen_key}" >"${temporary_dir}/qwen38-api-keys"
printf 'REDQUEEN_IMAGE_API_KEY=%s\nREDQUEEN_JEV_API_KEY=%s\nQWEN_API_KEY=%s\n' \
  "${image_key}" "${jev_key}" "${qwen_key}" >"${temporary_dir}/redqueen-adapters.env"
chmod 0600 "${temporary_dir}/qwen38-api-keys" "${temporary_dir}/redqueen-adapters.env"
mv -- "${temporary_dir}/qwen38-api-keys" "${qwen_file}"
mv -- "${temporary_dir}/redqueen-adapters.env" "${adapter_file}"

printf 'created %s and %s; values were not printed\n' "${qwen_file}" "${adapter_file}"
