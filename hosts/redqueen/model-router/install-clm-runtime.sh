#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly host_root="$(cd -- "${source_root}/.." && pwd -P)"
readonly target_root=/srv/ai/clm
readonly model_root=/srv/ai/models/clm-v0.1-8b
readonly venv="${target_root}/venv"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
for command in python3.13 systemctl; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

[[ -s "${model_root}/SHA256SUMS" ]] || {
  printf 'CLM model checksum manifest is missing; run download-llama-models.sh first\n' >&2
  exit 1
}
(cd -- "${model_root}" && sha256sum --check SHA256SUMS)
[[ -r /srv/ai/secrets/redqueen-adapters.env ]] || {
  printf 'adapter secrets are missing: /srv/ai/secrets/redqueen-adapters.env\n' >&2
  exit 1
}
mapfile -t clm_keys < <(
  sed -n 's/^CLM_API_KEY=//p' /srv/ai/secrets/redqueen-adapters.env
)
[[ "${#clm_keys[@]}" -eq 1 && "${clm_keys[0]}" =~ ^[a-f0-9]{64}$ ]] || {
  printf 'CLM_API_KEY must be initialized by create-secrets.sh or adapters/install.sh\n' >&2
  exit 1
}

"${host_root}/link-runtime.sh"
install -d -m 0750 -- "${target_root}"
if [[ ! -x "${venv}/bin/python" ]]; then
  python3.13 -m venv "${venv}"
fi
"${venv}/bin/pip" install \
  --disable-pip-version-check \
  --no-input \
  --only-binary=:all: \
  --upgrade \
  --requirement "${host_root}/clm/requirements.lock"
"${venv}/bin/pip" install \
  --disable-pip-version-check \
  --no-input \
  --no-deps \
  --upgrade \
  'contrastive-lm==0.1.0'

printf 'CLM CPU head runtime installed; the encoder uses the existing pinned llama.cpp runtime.\n'
printf 'No services were enabled, started, stopped or restarted.\n'
