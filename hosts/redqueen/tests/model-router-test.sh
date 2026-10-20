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
for flag in '--moe-stream' '--moe-stream-direct' '--moe-stream-cache 4g' '--moe-stream-io-threads 1'; do
  count="$(grep -Fxc -- "      ${flag}" "${router_config}")"
  if [[ "${count}" -ne 2 ]]; then
    printf 'expected both streamed MoE workers to use %s\n' "${flag}" >&2
    exit 1
  fi
done
grep -Fq '/srv/ai/runtimes/llama.cpp-d812350-moe-v1/bin/llama-fit-params' "${scheduler_policy}"
grep -Fq '"${source_root}/install-moe-runtime.sh"' "${host_root}/model-router/install.sh"
python3 - "${scheduler_policy}" <<'PY'
import json
import pathlib
import sys

policy = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert policy["upstream_url"] == "http://127.0.0.1:8084"
assert policy["gpu_budget_mib"] == 88 * 1024
assert policy["host_budget_mib"] == 104 * 1024
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
PY
if grep -Fq 'deepseek-v4.1-flash-q2' "${router_config}"; then
  printf 'DeepSeek is not eligible for registration until its weights verify\n' >&2
  exit 1
fi

for file in \
  "${host_root}/models/download-llama-models.sh" \
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
