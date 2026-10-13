#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd -P)"
readonly venv=/srv/ai/model-router/venv
[[ "$(id -u)" -ne 0 ]] || { printf 'run this as ai\n' >&2; exit 1; }
if [[ ! -x "$venv/bin/python" ]]; then
  python3.13 -m venv "$venv"
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13), "model-router venv requires Python 3.13"'
"$venv/bin/pip" install --disable-pip-version-check --no-input --only-binary=:all: \
  --requirement "$source_root/requirements.lock"
