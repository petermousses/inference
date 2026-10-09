#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly model_root="${MODEL_ROOT:-/srv/ai/models/qwen-image-2.1}"
readonly weight_revision="9a44dbdb47cefd046be9c0a13476192f34c8db8e"
readonly license_revision="790c92633540aa0cb11d9abf19eb46d861714758"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to download model files as root\n' >&2
  exit 1
fi

mkdir -p \
  "${model_root}/diffusion_models" \
  "${model_root}/text_encoders" \
  "${model_root}/vae"

exec 9>"${model_root}/.download.lock"
if ! flock -n 9; then
  printf 'another Qwen Image 2.1 download is already running\n' >&2
  exit 1
fi

download() {
  local url="$1"
  local relative_path="$2"
  local expected_sha256="$3"
  local destination="${model_root}/${relative_path}"
  local partial="${destination}.part"
  local http_status

  if [[ -f "${destination}" ]] && printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status; then
    printf 'verified existing %s\n' "${relative_path}"
    return
  fi

  if [[ -f "${partial}" ]] && printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status; then
    mv -- "${partial}" "${destination}"
    chmod 0440 "${destination}"
    printf 'promoted verified partial %s\n' "${relative_path}"
    return
  fi

  if ! http_status="$(curl \
    --fail \
    --location \
    --retry 10 \
    --retry-delay 5 \
    --connect-timeout 20 \
    --continue-at - \
    --output "${partial}" \
    --write-out '%{http_code}' \
    "${url}")" && [[ "${http_status}" != 416 ]]; then
    return 1
  fi

  if ! printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status; then
    printf 'discarding corrupt partial and retrying %s from byte zero\n' "${destination}" >&2
    rm -f -- "${partial}"
    curl \
      --fail \
      --location \
      --retry 10 \
      --retry-all-errors \
      --retry-delay 5 \
      --connect-timeout 20 \
      --output "${partial}" \
      "${url}"
    printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status
  fi
  mv -- "${partial}" "${destination}"
  chmod 0440 "${destination}"
  printf 'downloaded and verified %s\n' "${relative_path}"
}

readonly weights_base="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/${weight_revision}"
download "${weights_base}/diffusion_models/qwen_image_2.1_bf16.safetensors" \
  diffusion_models/qwen_image_2.1_bf16.safetensors \
  89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc
download "${weights_base}/text_encoders/qwen3vl_8b_bf16.safetensors" \
  text_encoders/qwen3vl_8b_bf16.safetensors \
  68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9
download "${weights_base}/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors" \
  text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors \
  9182abae56fe05459840a86d22abd21f972061c92fce032630af680c8c5178d3
download "${weights_base}/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors" \
  text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors \
  32707d01b427e488af252b95c551989aad59f9fec611a694f5db6bde7f0f1f6c
download "${weights_base}/vae/qwen_image_2.1_vae_bf16.safetensors" \
  vae/qwen_image_2.1_vae_bf16.safetensors \
  bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9

readonly license_base="https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/${license_revision}"
download "${license_base}/LICENSE" \
  LICENSE \
  8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d
download "${license_base}/README.md" \
  QWEN_MODEL_CARD.md \
  ee79e9dccc074f71fed40bf1cd74f35928b2f4e03a992ec193c437907308289f
download "${weights_base}/README.md" \
  COMFYUI_MODEL_CARD.md \
  2bc1a021663ae28110cd0b18a9486aa87161efb6a4550c3ecca3e1b84619052e

install -m 0440 "${script_dir}/qwen-image-2.1.sha256" "${model_root}/SHA256SUMS"
source_revision_partial="$(mktemp "${model_root}/.SOURCE_REVISION.XXXXXX")"
readonly source_revision_partial
printf '%s\n' \
  "weights_repository=Comfy-Org/Qwen-Image-2.1" \
  "weights_revision=${weight_revision}" \
  "license_repository=Qwen/Qwen-Image-2.1" \
  "license_revision=${license_revision}" \
  >"${source_revision_partial}"
chmod 0440 "${source_revision_partial}"
mv -- "${source_revision_partial}" "${model_root}/SOURCE_REVISION"

(
  cd -- "${model_root}"
  sha256sum --check SHA256SUMS
)
