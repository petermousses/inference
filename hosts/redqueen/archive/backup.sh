#!/usr/bin/env bash
set -euo pipefail

readonly ai_root="${REDQUEEN_AI_ROOT:-/srv/ai}"
readonly unit_root="${AI_USER_UNIT_DIR:-${HOME}/.config/systemd/user}"
readonly borg_bin="${BORG_BIN:-/srv/ai/bin/borg}"
readonly archive_name="redqueen-ai-$(date -u +%Y-%m-%dT%H%M%SZ)-$$"

[[ -n "${BORG_REPO:-}" ]] || {
  printf 'BORG_REPO is not configured; source the installed archive environment first\n' >&2
  exit 1
}
[[ -x "${borg_bin}" ]] || {
  printf 'Borg binary is not executable: %s\n' "${borg_bin}" >&2
  exit 1
}
[[ -d "${ai_root}/models" ]] || {
  printf 'authoritative model directory is missing: %s/models\n' "${ai_root}" >&2
  exit 1
}
[[ -n "$(find "${ai_root}/models" -type f -print -quit)" ]] || {
  printf 'refusing to archive an empty model directory: %s/models\n' "${ai_root}" >&2
  exit 1
}
[[ -d "${unit_root}" ]] || {
  printf 'installed user systemd units are missing: %s\n' "${unit_root}" >&2
  exit 1
}

exec "${borg_bin}" create \
  --stats \
  --compression auto,lz4 \
  --checkpoint-interval 900 \
  --exclude "${ai_root}/secrets" \
  --exclude "${ai_root}/cache" \
  --exclude "${ai_root}/archive-state" \
  --exclude "${ai_root}/model-router/run" \
  --exclude "${ai_root}/comfyui/state/temp" \
  "${BORG_REPO}::${archive_name}" \
  "${ai_root}" \
  "${unit_root}"
