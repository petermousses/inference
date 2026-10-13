#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly host_root="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")/.." && pwd -P)"
readonly venv=/srv/ai/model-router/venv
[[ "$(id -u)" -ne 0 ]] || { printf 'run this as ai\n' >&2; exit 1; }
bash "$host_root/model-router/create-venv.sh"
export PYTHONPATH="$host_root/adapters"
export PYTHONDONTWRITEBYTECODE=1
set -a
. /srv/ai/secrets/llama-swap.env
set +a
exec "$venv/bin/python" -m redqueen_adapters.deploy_scheduler "$@"
