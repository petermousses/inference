#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly source_workflow_root="$script_dir/workflows"
readonly workflow_root="${WORKFLOW_ROOT:-/srv/ai/comfyui/workflows}"
readonly user_workflow_root="${USER_WORKFLOW_ROOT:-/srv/ai/comfyui/state/user/default/workflows}"
readonly model_root="${QWEN_IMAGE_MODEL_ROOT:-/srv/ai/models/qwen-image-2.1}"
readonly -a required_model_paths=(
  "$model_root/diffusion_models/qwen_image_2.1_bf16.safetensors"
  "$model_root/text_encoders/qwen3vl_8b_bf16.safetensors"
  "$model_root/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
  "$model_root/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
  "$model_root/vae/qwen_image_2.1_vae_bf16.safetensors"
)

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to prepare workflows as root\n' >&2
  exit 1
fi

missing_model=0
for required_model_path in "${required_model_paths[@]}"; do
  if [[ ! -s "$required_model_path" || ! -r "$required_model_path" ]]; then
    printf 'required model is missing, empty or unreadable: %s\n' "$required_model_path" >&2
    missing_model=1
  fi
done
if [[ "$missing_model" -ne 0 ]]; then
  exit 1
fi

[[ -d "$workflow_root" ]] || {
  printf 'workflow runtime path is missing; run link-runtime.sh first: %s\n' "$workflow_root" >&2
  exit 1
}
[[ "$(readlink -f -- "$workflow_root")" == "$(readlink -f -- "$source_workflow_root")" ]] || {
  printf 'workflow runtime path must resolve to the tracked source directory: %s\n' "$source_workflow_root" >&2
  exit 1
}
(
  cd -- "$source_workflow_root"
  sha256sum --check SHA256SUMS
)

mkdir -p -- "$user_workflow_root"
ln -sfnT -- \
  "$workflow_root/qwen-image-2.1-t2i-bf16.json" \
  "$user_workflow_root/qwen-image-2.1-t2i.json"
ln -sfnT -- \
  "$workflow_root/qwen-image-2.1-edit-bf16.json" \
  "$user_workflow_root/qwen-image-2.1-edit.json"
