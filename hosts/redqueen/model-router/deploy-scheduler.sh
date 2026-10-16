#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly host_root="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")/.." && pwd -P)"
readonly venv=/srv/ai/model-router/venv
[[ "$(id -u)" -ne 0 ]] || { printf 'run this as ai\n' >&2; exit 1; }
readonly recovery=/srv/ai/model-router/libexec/host_reliability.py
check_only=no
for argument in "$@"; do
  [[ "$argument" != --check ]] || check_only=yes
done
# Recover before importing candidate application modules or checking candidate
# dependency pins. The installed helper is independent of this checkout.
if [[ "$check_only" != yes && -f "$recovery" && ! -L "$recovery" ]]; then
  /usr/bin/python3 "$recovery" recover
fi
bash "$host_root/model-router/create-venv.sh"
export PYTHONPATH="$host_root/adapters"
export PYTHONDONTWRITEBYTECODE=1
set -a
. /srv/ai/secrets/llama-swap.env
set +a
exec "$venv/bin/python" -m redqueen_adapters.deploy_scheduler "$@"
