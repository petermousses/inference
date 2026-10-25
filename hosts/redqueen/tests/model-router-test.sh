#!/usr/bin/env bash
set -euo pipefail

readonly test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${test_root}/.." && pwd)"
readonly repo_root="$(cd -- "${host_root}/../.." && pwd)"
readonly router_config="${host_root}/model-router/config.yaml"
readonly scheduler_policy="${host_root}/model-router/scheduler.json"

readonly -a text_models=(
  qwen3.8-27b
  gemma-4-e4b-it
  gemma-4-12b-it
  gemma-4-26b-a4b-it
  qwen3.6-35b-a3b
  deepseek-v4.1-flash-1017-unsloth
  glm-5.3-flash-abliterated
)
readonly -a clef_models=(
  clef-flash-bf16
  clef-flash-q8
  clef-flash-q4
  clef-q4
)

for model in "${text_models[@]}"; do
  grep -Fq "  ${model}:" "${router_config}"
  grep -Fq -- "--alias ${model}" "${router_config}"
  grep -Fq "      - ${model}" "${router_config}"
done

for model in "${clef_models[@]}"; do
  grep -Fq "  ${model}:" "${router_config}"
  grep -Fq -- "--alias ${model}" "${router_config}"
  grep -Fq "            - ${model}" "${router_config}"
done

grep -Fq '  - "${env.QWEN_API_KEY}"' "${router_config}"
grep -Fq 'globalTTL: 0' "${router_config}"
grep -Fq -- '--ctx-size 220000' "${router_config}"
grep -Fq -- "--chat-template-kwargs '{\"clear_thinking\":true}'" "${router_config}"
grep -Fq '          swap: false' "${router_config}"
grep -Fq '          exclusive: false' "${router_config}"

download_test_root="$(mktemp -d)"
trap 'rm -rf -- "${download_test_root}"' EXIT
download_test_bin="${download_test_root}/bin"
mkdir -p -- "${download_test_bin}"
cat >"${download_test_bin}/sha256sum" <<'SH'
#!/bin/sh
[ "${MODEL_ROUTER_TEST_SHA256SUM_OK:-0}" = 1 ]
SH
chmod +x "${download_test_bin}/sha256sum"
expected_unsloth_remaining_bytes="$(python3 - "${download_test_root}" \
  "${host_root}/models/deepseek-v4.1-flash-1017-unsloth.files.tsv" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "deepseek-v4.1-flash-1017-unsloth"
manifest = pathlib.Path(sys.argv[2])
rows = [line.split("\t") for line in manifest.read_text().splitlines() if line]
assert len(rows) == 8

completed_path = root / rows[0][0]
completed_path.parent.mkdir(parents=True, exist_ok=True)
with completed_path.open("wb") as completed:
    completed.truncate(int(rows[0][1]))

partial_bytes = 1 << 30
partial_path = root / (rows[1][0] + ".part")
partial_path.parent.mkdir(parents=True, exist_ok=True)
with partial_path.open("wb") as partial:
    partial.truncate(partial_bytes)

# Range staging bytes are intentionally not credited: they may need to be
# fetched again and are duplicated briefly when appended to the contiguous part.
range_path = pathlib.Path(str(partial_path) + ".range." + str(partial_bytes))
with range_path.open("wb") as staged_range:
    staged_range.truncate(512 << 20)

print(sum(int(row[1]) for row in rows) - int(rows[0][1]) - partial_bytes)
PY
)"
unsloth_plan="$(MODEL_ROOT="${download_test_root}" \
  MODEL_ROUTER_TEST_SHA256SUM_OK=1 PATH="${download_test_bin}:${PATH}" \
  bash "${host_root}/models/download-evaluation-models.sh" --plan)"
if [[ "${unsloth_plan}" != *"remaining targeted Unsloth DeepSeek bytes: ${expected_unsloth_remaining_bytes}"* ]]; then
  printf 'Unsloth download plan did not account for completed and resumable bytes\n%s\n' \
    "${unsloth_plan}" >&2
  exit 1
fi
if invalid_unsloth_plan="$(MODEL_ROOT="${download_test_root}" \
  MODEL_ROUTER_TEST_SHA256SUM_OK=0 PATH="${download_test_bin}:${PATH}" \
  bash "${host_root}/models/download-evaluation-models.sh" --plan 2>&1)"; then
  printf 'Unsloth download plan accepted a completed file with the wrong SHA-256\n' >&2
  exit 1
fi
if [[ "${invalid_unsloth_plan}" != *'existing file has the wrong SHA-256'* ]]; then
  printf 'unexpected failure while rejecting an invalid completed shard\n%s\n' \
    "${invalid_unsloth_plan}" >&2
  exit 1
fi

qwen36_command="$(awk '
  /^  qwen3\.6-35b-a3b:$/ { in_model = 1; next }
  /^  [^ ]/ { if (in_model) exit }
  in_model { print }
' "${router_config}")"
cache_max_gib="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_budget_mib"] // 1024)' "${scheduler_policy}")"
if [[ "${qwen36_command}" != *'--ubatch-size 1'* ]] || \
  [[ "${qwen36_command}" != *'--moe-stream'* ]] || \
  [[ "${qwen36_command}" != *"--moe-stream-cache-max ${cache_max_gib}g"* ]]; then
  printf 'Qwen3.6 must use the adaptive streamed expert-cache configuration\n' >&2
  exit 1
fi

for flag in '--moe-stream' '--moe-stream-direct' '--moe-stream-cache 4g' "--moe-stream-cache-max ${cache_max_gib}g" '--moe-stream-io-threads 1'; do
  count="$(grep -Fxc -- "      ${flag}" "${router_config}")"
  if [[ "${count}" -ne 4 ]]; then
    printf 'expected all streamed MoE workers to use %s\n' "${flag}" >&2
    exit 1
  fi
done
for model in gemma-4-26b-a4b-it qwen3.6-35b-a3b deepseek-v4.1-flash-1017-unsloth glm-5.3-flash-abliterated; do
  concurrency_limit="$(awk -v model="${model}" '
    $0 == "  " model ":" { in_model = 1; next }
    /^  [^ ]/ { if (in_model) exit }
    in_model && /concurrencyLimit:/ { print $2; exit }
  ' "${router_config}")"
  if [[ "${concurrency_limit}" != 2 ]]; then
    printf 'adaptive worker %s must allow one resize progress request alongside its resize stream\n' \
      "${model}" >&2
    exit 1
  fi
done
grep -Fq '/srv/ai/runtimes/llama.cpp-d812350-moe-v9/bin/llama-fit-params' "${scheduler_policy}"
grep -Fq '/srv/ai/runtimes/llama.cpp-d812350-deepseek41-moe-v13/bin/llama-server' "${router_config}"
grep -Fq '/srv/ai/runtimes/llama.cpp-d812350-moe-v9/bin/llama-server' "${router_config}"
grep -Fq 'readonly runtime=llama.cpp-d812350-moe-v9' "${host_root}/model-router/install-moe-runtime.sh"
grep -Fq 'readonly runtime=llama.cpp-d812350-deepseek41-moe-v13' "${host_root}/model-router/install-deepseek41-runtime.sh"
(cd "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9" && sha256sum --check --quiet PATCH_SHA256)
(cd "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13" && sha256sum --check --quiet PATCH_SHA256)
grep -Fq 'actual_capacity_bytes <= capacity_bytes' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'actual_capacity_bytes <= capacity_bytes' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'static_cast<__uint128_t>(target_slots) * capacity_bytes' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'report_progress("capacity_adjustment")' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'static_cast<__uint128_t>(target_slots) * capacity_bytes' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'report_progress("capacity_adjustment")' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
if grep -Fq '+            target_slots--;' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"; then
  printf 'MoE runtime still corrects allocator over-allocation one slot at a time\n' >&2
  exit 1
fi
grep -Fq 'tensor->data = nullptr;' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'arch != LLM_ARCH_GEMMA4' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/moe-stream.patch"
grep -Fq 'cache could not reserve all routed experts' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/moe-stream.patch"
grep -Fq 'cache could not reserve all routed experts' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/moe-stream.patch"
grep -Fq 'llama_moe_stream_resize_cache(context.get(), resize_bytes, nullptr, nullptr, &resized)' \
  "${host_root}/model-router/qualification/moe-stream-probe.cpp"
grep -Fq 'waiting_for_quiescence' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'using llama_moe_stream_progress_callback = void (*' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'llama_moe_stream_progress_callback progress_callback = nullptr' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq 'restoring_residents' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'using llama_moe_stream_progress_callback = void (*' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'llama_moe_stream_progress_callback progress_callback = nullptr' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-deepseek41-moe-v13/adaptive-cache.patch"
grep -Fq 'queued_loads' \
  "${host_root}/model-router/runtime-manifests/llama.cpp-d812350-moe-v9/adaptive-cache.patch"
grep -Fq '"${source_root}/install-moe-runtime.sh"' "${host_root}/model-router/install.sh"
python3 - "${scheduler_policy}" <<'PY'
import json
import pathlib
import sys

policy = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert policy["upstream_url"] == "http://127.0.0.1:8084"
assert policy["gpu_budget_mib"] == 120 * 1024
assert policy["host_budget_mib"] == 120 * 1024
assert policy["minimum_host_available_mib"] == 8 * 1024
assert policy["prefill_headroom_mib"] == 5 * 1024
assert policy["moe_stream_overhead_mib"] == 1536
assert policy["public_request_limit"] == 4
assert policy["model_root"] == "/srv/ai/models"
assert policy["artifact_metadata_root"].endswith("/hosts/redqueen/models/metadata")
assert policy["services"]["jevk5"]["supported_request_shapes"] == ["systemone.decision"]
assert policy["services"]["jevk5"]["concurrency_limit"] == 4
assert policy["services"]["clm"]["supported_request_shapes"] == ["systemone.decision"]
assert policy["services"]["comfyui"]["supported_request_shapes"] == ["openai.image_generation", "openai.image_edit"]
assert policy["services"]["deepseek"]["expert_cache_mib"] == 1024
assert policy["services"]["deepseek"]["expert_cache_max_mib"] == policy["gpu_budget_mib"]
assert policy["services"]["deepseek"]["expert_cache_url"] == "http://127.0.0.1:8085/expert-cache"
assert policy["services"]["deepseek"].get("exclusive_execution") is not True
PY
if grep -Fq 'deepseek-v4.1-flash-q2' "${router_config}"; then
  printf 'DeepSeek is not eligible for registration until its weights verify\n' >&2
  exit 1
fi

for file in \
  "${host_root}/models/download-llama-models.sh" \
  "${host_root}/models/download-evaluation-models.sh" \
  "${host_root}/models/materialize_deepseek_revision.py" \
  "${host_root}/adapters/install.sh" \
  "${host_root}/create-secrets.sh" \
  "${host_root}/model-router/create-env.sh" \
  "${host_root}/model-router/create-venv.sh" \
  "${host_root}/model-router/install-glm53-runtime.sh" \
  "${host_root}/model-router/install-moe-runtime.sh" \
  "${host_root}/model-router/glm-5.3-flash-abliterated-launcher.sh" \
  "${host_root}/model-router/install.sh" \
  "${host_root}/model-router/deploy-scheduler.sh" \
  "${host_root}/model-router/comfyui-control.py" \
  "${host_root}/comfyui/install-qwen-image-gguf.sh"; do
  if [[ "${file}" == *.py ]]; then
    python3 - "${file}" <<'PY'
import ast
import pathlib
import sys

ast.parse(pathlib.Path(sys.argv[1]).read_text(), filename=sys.argv[1])
PY
  else
    bash -n "${file}"
  fi
done

python3 -m unittest discover -s "${test_root}" -p 'test_deepseek_revision_materialization.py'

grep -Fq -- '--listen 127.0.0.1:8084' "${host_root}/systemd/llama-swap.service"
grep -Fq 'ProtectHome=read-only' "${host_root}/systemd/redqueen-scheduler.service"

printf 'static model-router and image-pipeline checks passed\n'
