#!/usr/bin/env bash
set -euo pipefail

readonly test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly backup_script="${test_root}/redqueen/archive/backup.sh"
readonly temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/redqueen-archive-test.XXXXXXXX")"
cleanup() {
  rm -rf -- "${temporary_root}"
}
trap cleanup EXIT

mkdir -p "${temporary_root}/ai/models" "${temporary_root}/units" "${temporary_root}/bin"
printf 'model fixture\n' >"${temporary_root}/ai/models/model.gguf"
cat >"${temporary_root}/bin/borg-stub" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"${BORG_TEST_ARGS}"
exit "${BORG_TEST_EXIT:-0}"
STUB
chmod 0755 "${temporary_root}/bin/borg-stub"

export BORG_BIN="${temporary_root}/bin/borg-stub"
export BORG_REPO=ssh://redqueen-archive@nas.example/repo
export BORG_TEST_ARGS="${temporary_root}/borg-args"
export REDQUEEN_AI_ROOT="${temporary_root}/ai"
export AI_USER_UNIT_DIR="${temporary_root}/units"

"${backup_script}"
assert_argument() {
  if ! grep -Fqx -- "$1" "${BORG_TEST_ARGS}"; then
    printf 'expected Borg argument was absent: %s\n' "$1" >&2
    exit 1
  fi
}
assert_argument create
assert_argument --exclude
assert_argument "${REDQUEEN_AI_ROOT}/secrets"
assert_argument "${REDQUEEN_AI_ROOT}/cache"
assert_argument "${REDQUEEN_AI_ROOT}/archive-state"
assert_argument "${REDQUEEN_AI_ROOT}/comfyui/state/temp"
assert_argument "${REDQUEEN_AI_ROOT}/model-router/run"
assert_argument "${REDQUEEN_AI_ROOT}"
assert_argument "${AI_USER_UNIT_DIR}"
grep -Fq -- "${BORG_REPO}::redqueen-ai-" "${BORG_TEST_ARGS}"
if grep -Eq -- '(^|/)(delete|prune|compact)( |$)' "${BORG_TEST_ARGS}"; then
  printf 'backup script unexpectedly requested a destructive Borg operation\n' >&2
  exit 1
fi

export BORG_TEST_EXIT=23
if "${backup_script}"; then
  printf 'backup script hid Borg failure status\n' >&2
  exit 1
fi

printf 'Borg backup invocation and failure propagation passed\n'
