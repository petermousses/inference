#!/usr/bin/env bash
set -Eeuo pipefail

readonly server=/srv/ai/runtimes/llama.cpp-v0.6.0-d812350/bin/llama-server
readonly curl=/usr/bin/curl
readonly control_socket=/srv/ai/model-router/run/comfyui-control.sock

if [[ ! -x "${server}" ]]; then
  printf 'pinned GLM runtime is missing: %s\n' "${server}" >&2
  exit 1
fi
if [[ ! -x "${curl}" ]]; then
  printf 'curl is missing: %s\n' "${curl}" >&2
  exit 1
fi
control_request() {
  "${curl}" --unix-socket "${control_socket}" --silent --show-error --fail \
    --connect-timeout 5 --max-time 75 --request POST "http://localhost/$1"
}

restore_comfyui=1
server_pid=''
signal_status=0
cleanup() {
  status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" || true
  fi
  if ((restore_comfyui)); then
    if ! control_request resume >/dev/null; then
      printf 'failed to restore ComfyUI after GLM stopped\n' >&2
      status=1
    fi
  fi
  exit "${status}"
}
forward_signal() {
  local signal="$1"
  case "${signal}" in
    INT) signal_status=130 ;;
    TERM) signal_status=143 ;;
    HUP) signal_status=129 ;;
  esac
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -s "${signal}" "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

pause_result="$(control_request pause)"
case "${pause_result}" in
  restore)
    ;;
  leave)
    restore_comfyui=0
    ;;
  *)
    printf 'ComfyUI-control helper returned an unexpected pause response\n' >&2
    exit 1
    ;;
esac

"${server}" "$@" &
server_pid=$!
while true; do
  set +e
  wait "${server_pid}"
  status=$?
  set -e
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
done
server_pid=''
if ((signal_status)); then
  exit "${signal_status}"
fi
exit "${status}"
