#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd -P)"
readonly venv=/srv/ai/model-router/venv
[[ "$(id -u)" -ne 0 ]] || { printf 'run this as ai\n' >&2; exit 1; }
mkdir -p /srv/ai/model-router/run
exec 9>/srv/ai/model-router/run/venv.lock
flock --exclusive 9
if [[ -e "$venv" ]]; then
  [[ -x "$venv/bin/python" ]] || { printf 'existing model-router venv is incomplete\n' >&2; exit 1; }
  "$venv/bin/python" - "$source_root/requirements.lock" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 13), "model-router venv requires Python 3.13"
for pin in Path(sys.argv[1]).read_text().splitlines():
    if not pin.strip() or pin.startswith("#"):
        continue
    name, expected = pin.split("==")
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        actual = "missing"
    if actual != expected:
        raise SystemExit(f"existing venv has {name}=={actual}; expected {expected}. build a separate replacement venv; live dependencies will not be changed")
PY
  "$venv/bin/python" -m pip check
  exit 0
fi
python3.13 -m venv "$venv"
"$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13), "model-router venv requires Python 3.13"'
"$venv/bin/python" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
  --requirement "$source_root/requirements.lock"
"$venv/bin/python" -m pip check
