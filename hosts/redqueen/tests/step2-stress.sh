#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly duration_seconds="${1:-3600}"
readonly run_base="${RUN_BASE:-/srv/ai/cache/step2-stress}"
readonly workflow_root="${WORKFLOW_ROOT:-/srv/ai/cache/step2-gates}"
readonly t2i_workflow="${workflow_root}/qwen-image-2.1-t2i-smoke-api.json"
readonly edit_workflow="${workflow_root}/qwen-image-2.1-edit-smoke-api.json"
readonly t2i_2k_workflow="${workflow_root}/qwen-image-2.1-t2i-2k-smoke-api.json"
readonly t2i_2k_every="${T2I_2K_EVERY:-5}"
readonly min_host_available_bytes="${MIN_HOST_AVAILABLE_BYTES:-8589934592}"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to run the stress test as root\n' >&2
  exit 1
fi
if [[ ! "${duration_seconds}" =~ ^[0-9]+$ || "${duration_seconds}" -lt 60 ]]; then
  printf 'duration must be an integer of at least 60 seconds\n' >&2
  exit 1
fi
if [[ ! "${min_host_available_bytes}" =~ ^[0-9]+$ ]]; then
  printf 'MIN_HOST_AVAILABLE_BYTES must be a non-negative integer\n' >&2
  exit 1
fi
if [[ ! "${t2i_2k_every}" =~ ^[0-9]+$ || "${t2i_2k_every}" -eq 0 ]]; then
  printf 'T2I_2K_EVERY must be a positive integer\n' >&2
  exit 1
fi
for command in curl jq systemctl awk; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done
for workflow in "${t2i_workflow}" "${edit_workflow}" "${t2i_2k_workflow}"; do
  if [[ ! -r "${workflow}" ]] || ! jq -e . "${workflow}" >/dev/null; then
    printf 'workflow is missing, unreadable or invalid: %s\n' "${workflow}" >&2
    exit 1
  fi
done

readonly run_id="$(date -u +%Y%m%dT%H%M%SZ)"
readonly run_dir="${run_base}/${run_id}"
readonly start_epoch="$(date +%s)"
readonly end_epoch="$((start_epoch + duration_seconds))"
readonly start_iso="$(date --iso-8601=seconds)"
readonly boot_id="$(< /proc/sys/kernel/random/boot_id)"
mkdir -p -- "${run_dir}"
ln -sfn -- "${run_dir}" "${run_base}/latest"

printf '%s\n' \
  "RUN_ID=${run_id}" \
  "BOOT_ID=${boot_id}" \
  "START_EPOCH=${start_epoch}" \
  "START_ISO=${start_iso}" \
  "PLANNED_END_EPOCH=${end_epoch}" \
  "DURATION_SECONDS=${duration_seconds}" \
  >"${run_dir}/metadata.env"

for service in qwen38.service jevk5.service comfyui.service; do
  if [[ "$(systemctl --user is-active "${service}")" != active ]]; then
    printf 'required service is not active: %s\n' "${service}" >&2
    exit 1
  fi
done
curl -fsS http://127.0.0.1:8081/health >/dev/null
curl -fsS http://127.0.0.1:8082/health >/dev/null
curl -fsS http://127.0.0.1:8189/system_stats >/dev/null

readonly starting_qwen_restarts="$(systemctl --user show qwen38.service -p NRestarts --value)"
readonly starting_jevk_restarts="$(systemctl --user show jevk5.service -p NRestarts --value)"
readonly starting_comfy_restarts="$(systemctl --user show comfyui.service -p NRestarts --value)"
printf '%s\n' \
  "STARTING_QWEN_RESTARTS=${starting_qwen_restarts}" \
  "STARTING_JEVK_RESTARTS=${starting_jevk_restarts}" \
  "STARTING_COMFY_RESTARTS=${starting_comfy_restarts}" \
  "MIN_HOST_AVAILABLE_BYTES=${min_host_available_bytes}" \
  "T2I_2K_EVERY=${t2i_2k_every}" \
  >>"${run_dir}/metadata.env"

readonly qwen38_cgroup="/sys/fs/cgroup$(systemctl --user show qwen38.service -p ControlGroup --value)"
readonly jevk5_cgroup="/sys/fs/cgroup$(systemctl --user show jevk5.service -p ControlGroup --value)"
readonly comfyui_cgroup="/sys/fs/cgroup$(systemctl --user show comfyui.service -p ControlGroup --value)"
readonly slice_cgroup="/sys/fs/cgroup$(systemctl --user show ai-inference.slice -p ControlGroup --value)"
gtt_file="$(find -L /sys/class/drm/card*/device -maxdepth 1 -name mem_info_gtt_used -print -quit 2>/dev/null || true)"
readonly gtt_file

cp -- "${slice_cgroup}/memory.events" "${run_dir}/slice-memory.events.start"
for service in qwen38 jevk5 comfyui; do
  cgroup_var="${service}_cgroup"
  cp -- "${!cgroup_var}/memory.events" "${run_dir}/${service}-memory.events.start"
done

declare -a worker_pids=()
cleanup() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${worker_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

chat_worker() {
  local worker="$1"
  local port="$2"
  local model="$3"
  local prompt="$4"
  local response="${run_dir}/${worker}.response.json"
  local log="${run_dir}/${worker}.tsv"
  local summary="${run_dir}/${worker}.summary"
  local iteration=0
  local successes=0
  local failures=0
  local request_start request_end result
  printf 'iteration\tstart_epoch\tend_epoch\tresult\n' >"${log}"
  while (( $(date +%s) < end_epoch )); do
    iteration=$((iteration + 1))
    request_start="$(date +%s)"
    if jq -nc \
        --arg model "${model}" \
        --arg prompt "${prompt}" \
        '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:128}' \
        | curl -fsS \
            --connect-timeout 5 \
            --max-time 180 \
            -H 'Content-Type: application/json' \
            --data-binary @- \
            --output "${response}" \
            "http://127.0.0.1:${port}/v1/chat/completions" \
      && jq -e '.choices[0].finish_reason | strings | length > 0' "${response}" >/dev/null \
      && jq -e '.usage.completion_tokens > 0' "${response}" >/dev/null; then
      successes=$((successes + 1))
      result=success
    else
      failures=$((failures + 1))
      result=failure
    fi
    request_end="$(date +%s)"
    printf '%s\t%s\t%s\t%s\n' \
      "${iteration}" "${request_start}" "${request_end}" "${result}" >>"${log}"
  done
  printf 'worker=%s\nsuccesses=%s\nfailures=%s\n' \
    "${worker}" "${successes}" "${failures}" >"${summary}"
}

image_worker() {
  local log="${run_dir}/image.tsv"
  local summary="${run_dir}/image.summary"
  local response="${run_dir}/image.response.json"
  local history_file="${run_dir}/image.history.json"
  local iteration=0
  local successes=0
  local failures=0
  local t2i_2k_attempts=0
  local t2i_2k_successes=0
  local seed workflow prompt_id request_start request_end result
  printf 'iteration\tstart_epoch\tend_epoch\tworkflow\tseed\tresult\n' >"${log}"
  while (( $(date +%s) < end_epoch )); do
    iteration=$((iteration + 1))
    seed=$((start_epoch * 1000 + iteration))
    if (( iteration == 1 || iteration % t2i_2k_every == 0 )); then
      workflow="${t2i_2k_workflow}"
    elif (( iteration % 2 == 1 )); then
      workflow="${t2i_workflow}"
    else
      workflow="${edit_workflow}"
    fi
    request_start="$(date +%s)"
    result=failure
    if [[ "${workflow}" == "${t2i_2k_workflow}" ]]; then
      t2i_2k_attempts=$((t2i_2k_attempts + 1))
    fi
    if jq -c \
        --argjson seed "${seed}" \
        --arg prefix "stress/${run_id}/qwen-image-2.1" \
        '."7".inputs.seed = $seed | ."9".inputs.filename_prefix = $prefix | {prompt:.}' \
        "${workflow}" \
        | curl -fsS \
            --connect-timeout 5 \
            --max-time 30 \
            -H 'Content-Type: application/json' \
            --data-binary @- \
            --output "${response}" \
            http://127.0.0.1:8189/prompt; then
      prompt_id="$(jq -er '.prompt_id' "${response}" 2>/dev/null || true)"
      if [[ -n "${prompt_id}" ]]; then
        for _ in $(seq 1 300); do
          if curl -fsS \
              --connect-timeout 5 \
              --max-time 30 \
              --output "${history_file}" \
              "http://127.0.0.1:8189/history/${prompt_id}" \
            && jq -e --arg id "${prompt_id}" '.[ $id ]' "${history_file}" >/dev/null; then
            if jq -e --arg id "${prompt_id}" \
                '.[ $id ].status.status_str == "success"' "${history_file}" >/dev/null; then
              result=success
            fi
            break
          fi
          sleep 2
        done
      fi
    fi
    if [[ "${result}" == success ]]; then
      successes=$((successes + 1))
      if [[ "${workflow}" == "${t2i_2k_workflow}" ]]; then
        t2i_2k_successes=$((t2i_2k_successes + 1))
      fi
    else
      failures=$((failures + 1))
    fi
    request_end="$(date +%s)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${iteration}" "${request_start}" "${request_end}" \
      "$(basename -- "${workflow}")" "${seed}" "${result}" >>"${log}"
  done
  printf 'worker=image\nsuccesses=%s\nfailures=%s\nt2i_2k_attempts=%s\nt2i_2k_successes=%s\n' \
    "${successes}" "${failures}" \
    "${t2i_2k_attempts}" "${t2i_2k_successes}" >"${summary}"
}

chat_worker \
  qwen \
  8081 \
  qwen3.8-27b \
  'Explain in approximately 100 words why deterministic validation matters for GPU inference systems.' &
worker_pids+=("$!")
for index in 1 2 3 4; do
  chat_worker \
    "jevk-${index}" \
    8082 \
    jevk5-4b-v0.3 \
    'Classify this support request and briefly explain the decision: My parcel was delivered to the wrong address.' &
  worker_pids+=("$!")
done
image_worker &
worker_pids+=("$!")

readonly metrics="${run_dir}/metrics.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  timestamp epoch qwen_active jevk_active comfy_active \
  qwen_restarts jevk_restarts comfy_restarts \
  qwen_current qwen_peak jevk_current jevk_peak \
  comfy_current comfy_peak slice_current slice_peak gtt_used mem_available loadavg \
  >"${metrics}"
service_failure_latched=0
sample_metrics() {
  now_epoch="$(date +%s)"
  qwen_active="$(systemctl --user is-active qwen38.service || true)"
  jevk_active="$(systemctl --user is-active jevk5.service || true)"
  comfy_active="$(systemctl --user is-active comfyui.service || true)"
  qwen_restarts="$(systemctl --user show qwen38.service -p NRestarts --value)"
  jevk_restarts="$(systemctl --user show jevk5.service -p NRestarts --value)"
  comfy_restarts="$(systemctl --user show comfyui.service -p NRestarts --value)"
  if [[ "${qwen_active}" != active \
    || "${jevk_active}" != active \
    || "${comfy_active}" != active \
    || "${qwen_restarts}" != "${starting_qwen_restarts}" \
    || "${jevk_restarts}" != "${starting_jevk_restarts}" \
    || "${comfy_restarts}" != "${starting_comfy_restarts}" ]]; then
    service_failure_latched=1
  fi
  if [[ -n "${gtt_file}" && -r "${gtt_file}" ]]; then
    gtt_used="$(< "${gtt_file}")"
  else
    gtt_used=NA
  fi
  mem_available="$(awk '$1 == "MemAvailable:" {print $2 * 1024}' /proc/meminfo)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" \
    "${now_epoch}" \
    "${qwen_active}" \
    "${jevk_active}" \
    "${comfy_active}" \
    "${qwen_restarts}" \
    "${jevk_restarts}" \
    "${comfy_restarts}" \
    "$(< "${qwen38_cgroup}/memory.current")" \
    "$(< "${qwen38_cgroup}/memory.peak")" \
    "$(< "${jevk5_cgroup}/memory.current")" \
    "$(< "${jevk5_cgroup}/memory.peak")" \
    "$(< "${comfyui_cgroup}/memory.current")" \
    "$(< "${comfyui_cgroup}/memory.peak")" \
    "$(< "${slice_cgroup}/memory.current")" \
    "$(< "${slice_cgroup}/memory.peak")" \
    "${gtt_used}" \
    "${mem_available}" \
    "$(< /proc/loadavg)" \
    >>"${metrics}"
}

while (( $(date +%s) < end_epoch )); do
  sample_metrics
  sleep 5
done

while true; do
  worker_running=0
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      worker_running=1
      break
    fi
  done
  if [[ "${worker_running}" -eq 0 ]]; then
    break
  fi
  sample_metrics
  sleep 5
done

for pid in "${worker_pids[@]}"; do
  wait "${pid}"
done
trap - INT TERM EXIT

readonly end_iso="$(date --iso-8601=seconds)"
readonly actual_end_epoch="$(date +%s)"
printf '%s\n' \
  "END_EPOCH=${actual_end_epoch}" \
  "END_ISO=${end_iso}" \
  >>"${run_dir}/metadata.env"

cp -- "${slice_cgroup}/memory.events" "${run_dir}/slice-memory.events.end"
for service in qwen38 jevk5 comfyui; do
  cgroup_var="${service}_cgroup"
  cp -- "${!cgroup_var}/memory.events" "${run_dir}/${service}-memory.events.end"
done
curl -fsS http://127.0.0.1:8081/metrics >"${run_dir}/qwen.metrics"
curl -fsS http://127.0.0.1:8082/metrics >"${run_dir}/jevk.metrics"
systemctl --user show qwen38.service jevk5.service comfyui.service \
  -p Id -p ActiveState -p Result -p NRestarts -p MemoryCurrent -p MemoryPeak \
  >"${run_dir}/services.end"
systemctl --user --failed --no-legend >"${run_dir}/failed-units.end"
if [[ "$(systemctl --user show qwen38.service -p NRestarts --value)" != "${starting_qwen_restarts}" \
  || "$(systemctl --user show jevk5.service -p NRestarts --value)" != "${starting_jevk_restarts}" \
  || "$(systemctl --user show comfyui.service -p NRestarts --value)" != "${starting_comfy_restarts}" ]]; then
  service_failure_latched=1
fi

oom_event_increases=0
for scope in qwen38 jevk5 comfyui slice; do
  for event in oom oom_kill oom_group_kill; do
    start_value="$(awk -v event="${event}" '$1 == event {print $2}' "${run_dir}/${scope}-memory.events.start")"
    end_value="$(awk -v event="${event}" '$1 == event {print $2}' "${run_dir}/${scope}-memory.events.end")"
    if (( end_value > start_value )); then
      oom_event_increases=$((oom_event_increases + end_value - start_value))
    fi
  done
done

successes="$(awk -F= '$1 == "successes" {sum += $2} END {print sum + 0}' "${run_dir}"/*.summary)"
failures="$(awk -F= '$1 == "failures" {sum += $2} END {print sum + 0}' "${run_dir}"/*.summary)"
t2i_2k_attempts="$(awk -F= '$1 == "t2i_2k_attempts" {print $2 + 0}' "${run_dir}/image.summary")"
t2i_2k_successes="$(awk -F= '$1 == "t2i_2k_successes" {print $2 + 0}' "${run_dir}/image.summary")"
read -r max_qwen max_jevk max_comfy max_slice max_gtt min_host_available < <(
  awk -F '\t' '
    NR > 1 {
      if ($9 > q) q=$9
      if ($11 > j) j=$11
      if ($13 > c) c=$13
      if ($15 > s) s=$15
      if ($17 != "NA" && $17 > g) g=$17
      if (m == 0 || $18 < m) m=$18
    }
    END {print q+0, j+0, c+0, s+0, g+0, m+0}
  ' "${metrics}"
)
printf '%s\n' \
  "SUCCESSES=${successes}" \
  "FAILURES=${failures}" \
  "T2I_2K_ATTEMPTS=${t2i_2k_attempts}" \
  "T2I_2K_SUCCESSES=${t2i_2k_successes}" \
  "MAX_QWEN_MEMORY=${max_qwen}" \
  "MAX_JEVK_MEMORY=${max_jevk}" \
  "MAX_COMFY_MEMORY=${max_comfy}" \
  "MAX_SLICE_MEMORY=${max_slice}" \
  "MAX_GTT_USED=${max_gtt}" \
  "MIN_HOST_AVAILABLE=${min_host_available}" \
  "OOM_EVENT_INCREASES=${oom_event_increases}" \
  "SERVICE_FAILURE_LATCHED=${service_failure_latched}" \
  >"${run_dir}/summary.env"

if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi
if [[ "${t2i_2k_successes}" -lt 1 ]]; then
  exit 1
fi
if [[ -s "${run_dir}/failed-units.end" ]]; then
  exit 1
fi
if [[ "${oom_event_increases}" -ne 0 ]]; then
  exit 1
fi
if [[ "${service_failure_latched}" -ne 0 ]]; then
  exit 1
fi
if (( min_host_available < min_host_available_bytes )); then
  exit 1
fi
for service in qwen38.service jevk5.service comfyui.service; do
  if [[ "$(systemctl --user is-active "${service}")" != active ]]; then
    exit 1
  fi
done
