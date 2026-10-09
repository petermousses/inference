#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly workflow_revision="99e3d43745926b78466d99f937c0cd2bb622423a"
readonly workflow_root="${WORKFLOW_ROOT:-/srv/ai/comfyui/workflows}"
readonly user_workflow_root="${USER_WORKFLOW_ROOT:-/srv/ai/comfyui/state/user/default/workflows}"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly upstream_base="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${workflow_revision}/templates"
readonly model_root="${QWEN_IMAGE_MODEL_ROOT:-/srv/ai/models/qwen-image-2.1}"
readonly -a required_model_paths=(
  "${model_root}/diffusion_models/qwen_image_2.1_bf16.safetensors"
  "${model_root}/text_encoders/qwen3vl_8b_bf16.safetensors"
  "${model_root}/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
  "${model_root}/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
  "${model_root}/vae/qwen_image_2.1_vae_bf16.safetensors"
)
model_note=''
printf -v model_note '%s\n' \
  '## redqueen model status' \
  '' \
  'all required model files are already installed on redqueen and visible to ComfyUI. do not use browser download links; those save files to the workstation, not the server.' \
  '' \
  '**authoritative server path**' \
  '' \
  "\`${model_root}/\`" \
  '' \
  'in Model Library, select **Load All Folders** or expand these folders:' \
  '' \
  '- `diffusion_models` — 1 model' \
  '- `text_encoders` — 3 models' \
  '- `vae` — 1 model' \
  '' \
  'source repository: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)'
readonly model_note

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to prepare workflows as root\n' >&2
  exit 1
fi

missing_model=0
for required_model_path in "${required_model_paths[@]}"; do
  if [[ ! -s "${required_model_path}" || ! -r "${required_model_path}" ]]; then
    printf 'required model is missing, empty or unreadable: %s\n' "${required_model_path}" >&2
    missing_model=1
  fi
done
if [[ "${missing_model}" -ne 0 ]]; then
  exit 1
fi
readonly missing_model

mkdir -p -- "${workflow_root}" "${user_workflow_root}"

prepare() {
  local upstream_name="$1"
  local local_name="$2"
  local upstream_sha256="$3"
  local bf16_sha256="$4"
  local upstream_path="${workflow_root}/${local_name}.upstream.json"
  local bf16_path="${workflow_root}/${local_name}-bf16.json"

  curl --fail --location --retry 5 --retry-all-errors \
    --output "${upstream_path}.part" \
    "${upstream_base}/${upstream_name}"
  printf '%s  %s\n' "${upstream_sha256}" "${upstream_path}.part" | sha256sum --check --status
  mv -- "${upstream_path}.part" "${upstream_path}"

  sed \
    -e 's/qwen_image_2\.1_int8_convrot\.safetensors/qwen_image_2.1_bf16.safetensors/g' \
    -e 's/qwen3vl_8b_int8_convrot\.safetensors/qwen3vl_8b_bf16.safetensors/g' \
    "${upstream_path}" \
    | jq --arg model_note "${model_note}" \
      '(.nodes[] | select(.type == "MarkdownNote" and .title == "Note: Model links") | .widgets_values[0]) = $model_note' \
      >"${bf16_path}.part"
  printf '%s  %s\n' "${bf16_sha256}" "${bf16_path}.part" | sha256sum --check --status
  mv -- "${bf16_path}.part" "${bf16_path}"
  chmod 0440 "${upstream_path}" "${bf16_path}"
}

prepare \
  image_qwen_image_2_1_t2i.json \
  qwen-image-2.1-t2i \
  7d947d9c59d54830acbb990a04395b5c4aaa646c27ff43e63d755ce7f3563008 \
  8ee71a715f5879f3e06c50dc34ea891355287600fa2663f483819dff9cffe8a9
prepare \
  image_qwen_image_2_1_image_edit.json \
  qwen-image-2.1-edit \
  a1ff79f02bb0e5553699e3f2813d1c2ed53c37f7ff7d2dd4823eabb4a2178975 \
  810abcd5dbec13e99a2ad6a8312e701dbda5cd34ea93f9d71a824ef2c4c23cfa

install -m 0440 "${script_dir}/qwen-image-2.1-workflows.sha256" "${workflow_root}/SHA256SUMS"
(
  cd -- "${workflow_root}"
  sha256sum --check SHA256SUMS
)

ln -sfn \
  "${workflow_root}/qwen-image-2.1-t2i-bf16.json" \
  "${user_workflow_root}/qwen-image-2.1-t2i.json"
ln -sfn \
  "${workflow_root}/qwen-image-2.1-edit-bf16.json" \
  "${user_workflow_root}/qwen-image-2.1-edit.json"
