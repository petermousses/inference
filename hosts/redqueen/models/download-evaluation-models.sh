#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly model_root="${MODEL_ROOT:-/srv/ai/models}"
readonly required_bytes=429332209744
readonly reserve_bytes=100000000000
# the xet resolver rejects open-ended ranges for very large artifacts.
readonly ranged_download_threshold_bytes=2147483648
readonly ranged_download_chunk_bytes=1073741824

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to download model files as root\n' >&2
  exit 1
fi

plan() {
  printf '%-32s %12s  %s\n' MODEL BYTES FILE
  printf '%-32s %12s  %s\n' deepseek-v4.1-flash-q2 365713686528 \
    DeepSeek-V4.1-Flash-Q2.gguf
  printf '%-32s %12s  %s\n' gemma-4-e4b-it 5150682208 \
    'Q4_0 GGUF + Q8_0 vision projector'
  printf '%-32s %12s  %s\n' gemma-4-12b-it 7378660832 \
    'Q4_0 GGUF + Q8_0 vision projector'
  printf '%-32s %12s  %s\n' gemma-4-26b-a4b-it 15424554144 \
    'Q4_0 GGUF + Q8_0 vision projector'
  printf '%-32s %12s  %s\n' qwen-image-2.1-uncensored 14630866160 \
    'Q4_K_M GGUF + INT8 ConvRot encoder + BF16 VAE'
  printf '%-32s %12s  %s\n' qwen3.6-35b-a3b 21033759872 \
    'Q4_K_M GGUF + Q8_0 vision projector'
  printf 'total model/encoder/projector bytes: %s\n' "${required_bytes}"
  printf 'destination root: %s\n' "${model_root}"
}

verify_all() {
  local model_id root
  for model_id in \
    gemma-4-e4b-it \
    gemma-4-12b-it \
    gemma-4-26b-a4b-it \
    qwen-image-2.1-uncensored \
    qwen3.6-35b-a3b \
    deepseek-v4.1-flash-q2; do
    root="${model_root}/${model_id}"
    [[ -s "${root}/SHA256SUMS" ]] || {
      printf 'missing checksum manifest: %s\n' "${root}/SHA256SUMS" >&2
      return 1
    }
    printf 'verifying %s\n' "${model_id}"
    (cd -- "${root}" && sha256sum --check SHA256SUMS)
  done
}

case "${1:---plan}" in
  --plan)
    plan
    exit 0
    ;;
  --verify-only)
    verify_all
    exit 0
    ;;
  --download)
    ;;
  *)
    printf 'usage: %s [--plan|--download|--verify-only]\n' "$0" >&2
    exit 2
    ;;
esac

mkdir -p -- "${model_root}"
exec 9>"${model_root}/.evaluation-models-download.lock"
flock -n 9 || {
  printf 'another evaluation-model download is already running\n' >&2
  exit 1
}

available_bytes="$(df -B1 --output=avail "${model_root}" | tail -n 1 | tr -d ' ')"
if (( available_bytes < required_bytes + reserve_bytes )); then
  printf 'need at least %s bytes plus %s bytes of free-space reserve; found %s\n' \
    "${required_bytes}" "${reserve_bytes}" "${available_bytes}" >&2
  exit 1
fi

cleanup_range_partials() {
  local partial="$1"
  local range_path
  for range_path in "${partial}".range.*; do
    [[ -e "${range_path}" || -L "${range_path}" ]] || continue
    rm -- "${range_path}"
  done
}

download_artifact() {
  local model_id="$1"
  local repository="$2"
  local revision="$3"
  local source_path="$4"
  local destination_path="$5"
  local expected_size="$6"
  local expected_sha256="$7"
  local root="${model_root}/${model_id}"
  local destination="${root}/${destination_path}"
  local partial="${destination}.part"
  local actual_size range_start range_end range_path range_bytes range_size
  local range_attempt next_range_start remaining_range_bytes previous_range_size
  local range_url

  mkdir -p -- "$(dirname -- "${destination}")"
  if [[ -f "${destination}" ]]; then
    printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status || {
      printf 'existing file has the wrong SHA-256; preserving it: %s\n' \
        "${destination}" >&2
      return 1
    }
    printf 'verified existing %s\n' "${destination}"
  else
    [[ ! -e "${destination}" ]] || {
      printf 'refusing to replace non-file destination: %s\n' "${destination}" >&2
      return 1
    }

    [[ ! -e "${partial}" || -f "${partial}" ]] || {
      printf 'refusing to replace non-file partial: %s\n' "${partial}" >&2
      return 1
    }

    if [[ -f "${partial}" ]]; then
      actual_size="$(stat -c '%s' -- "${partial}")"
      if (( actual_size == expected_size )); then
        printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status || {
          printf 'complete partial has the wrong SHA-256; preserving it: %s\n' \
            "${partial}" >&2
          return 1
        }
        cleanup_range_partials "${partial}"
        mv -- "${partial}" "${destination}"
        chmod 0440 "${destination}"
        printf 'promoted verified partial %s\n' "${destination}"
      elif (( actual_size > expected_size )); then
        printf 'partial exceeds expected size; preserving it: %s\n' "${partial}" >&2
        return 1
      else
        printf 'resuming %s at byte %s\n' "${destination}" "${actual_size}"
      fi
    fi

    if [[ ! -f "${destination}" ]]; then
      range_url="https://huggingface.co/${repository}/resolve/${revision}/${source_path}?download=true"
      if (( expected_size > ranged_download_threshold_bytes )); then
        actual_size=0
        if [[ -f "${partial}" ]]; then
          actual_size="$(stat -c '%s' -- "${partial}")"
        fi
        while (( actual_size < expected_size )); do
          range_start="${actual_size}"
          range_end=$((range_start + ranged_download_chunk_bytes - 1))
          if (( range_end >= expected_size )); then
            range_end=$((expected_size - 1))
          fi
          range_path="${partial}.range.${range_start}"
          range_size=0

          if [[ -f "${range_path}" ]]; then
            range_size="$(stat -c '%s' -- "${range_path}")"
          elif [[ -e "${range_path}" ]]; then
            printf 'refusing to replace non-file range destination: %s\n' \
              "${range_path}" >&2
            return 1
          fi

          range_bytes=$((range_end - range_start + 1))
          if (( range_size > range_bytes )); then
            printf 'range partial exceeds requested size; preserving it: %s\n' \
              "${range_path}" >&2
            return 1
          fi

          if (( range_size < range_bytes )); then
            range_attempt=0
            while (( range_size < range_bytes )); do
              next_range_start=$((range_start + range_size))
              remaining_range_bytes=$((range_end - next_range_start + 1))
              previous_range_size="${range_size}"
              printf 'downloading bytes %s-%s for %s\n' \
                "${next_range_start}" "${range_end}" "${destination}"
              if curl \
                --fail \
                --location \
                --connect-timeout 20 \
                --speed-limit 1024 \
                --speed-time 60 \
                --max-filesize "${remaining_range_bytes}" \
                --range "${next_range_start}-${range_end}" \
                --output - \
                "${range_url}" >>"${range_path}"; then
                range_size="$(stat -c '%s' -- "${range_path}")"
                if (( range_size > range_bytes )); then
                  printf 'range partial exceeds requested size; preserving it: %s\n' \
                    "${range_path}" >&2
                  return 1
                fi
                if (( range_size == previous_range_size )); then
                  range_attempt=$((range_attempt + 1))
                  if (( range_attempt >= 10 )); then
                    printf 'range request made no progress after %s attempts; partial bytes are preserved in %s\n' \
                      "${range_attempt}" "${range_path}" >&2
                    return 1
                  fi
                  printf 'range request made no progress; retrying in 5 seconds (%s/10)\n' \
                    "${range_attempt}" >&2
                  sleep 5
                else
                  range_attempt=0
                fi
              else
                range_size="$(stat -c '%s' -- "${range_path}")"
                range_attempt=$((range_attempt + 1))
                if (( range_attempt >= 10 )); then
                  printf 'range download failed after %s attempts; partial bytes are preserved in %s\n' \
                    "${range_attempt}" "${range_path}" >&2
                  return 1
                fi
                printf 'range request failed; retrying in 5 seconds (%s/10)\n' \
                  "${range_attempt}" >&2
                sleep 5
              fi
            done
          fi

          if (( range_size != range_bytes )); then
            printf 'wrong size for range %s-%s: expected %s, got %s; partial preserved\n' \
              "${range_start}" "${range_end}" "${range_bytes}" "${range_size}" >&2
            return 1
          fi

          cat -- "${range_path}" >>"${partial}"
          rm -- "${range_path}"
          actual_size="$(stat -c '%s' -- "${partial}")"
        done
      else
        curl \
          --fail \
          --location \
          --retry 10 \
          --retry-all-errors \
          --retry-delay 5 \
          --connect-timeout 20 \
          --continue-at - \
          --output "${partial}" \
          "${range_url}"
      fi

      actual_size="$(stat -c '%s' -- "${partial}")"
      if [[ "${actual_size}" != "${expected_size}" ]]; then
        printf 'wrong size for %s: expected %s, got %s; partial preserved\n' \
          "${destination}" "${expected_size}" "${actual_size}" >&2
        return 1
      fi
      printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status || {
        printf 'wrong SHA-256 for %s; partial preserved\n' "${destination}" >&2
        return 1
      }
      cleanup_range_partials "${partial}"
      mv -- "${partial}" "${destination}"
      chmod 0440 "${destination}"
      printf 'downloaded and verified %s\n' "${destination}"
    fi
  fi

  printf '%s  %s\n' "${expected_sha256}" "${destination_path}" \
    >>"${root}/.SHA256SUMS.tmp"
}

download_text() {
  local model_id="$1"
  local url="$2"
  local destination_name="$3"
  local root="${model_root}/${model_id}"
  local destination="${root}/${destination_name}"
  local partial="${destination}.part"

  mkdir -p -- "${root}"
  if [[ -e "${destination}" ]]; then
    [[ -f "${destination}" && -s "${destination}" ]] || {
      printf 'refusing to replace metadata path: %s\n' "${destination}" >&2
      return 1
    }
    return
  fi
  curl \
    --fail \
    --location \
    --retry 10 \
    --retry-all-errors \
    --retry-delay 5 \
    --connect-timeout 20 \
    --output "${partial}" \
    "${url}"
  [[ -s "${partial}" ]] || {
    printf 'empty metadata download: %s\n' "${url}" >&2
    return 1
  }
  mv -- "${partial}" "${destination}"
  chmod 0440 "${destination}"
}

write_source_revision() {
  local model_id="$1"
  shift
  local root="${model_root}/${model_id}"
  local temporary="${root}/.SOURCE_REVISION.tmp"

  printf '%s\n' "$@" >"${temporary}"
  chmod 0440 "${temporary}"
  mv -- "${temporary}" "${root}/SOURCE_REVISION"
}

finish_model() {
  local model_id="$1"
  local root="${model_root}/${model_id}"
  (
    cd -- "${root}"
    sha256sum LICENSE MODEL_CARD.md SOURCE_REVISION >>.SHA256SUMS.tmp
    mv -- .SHA256SUMS.tmp SHA256SUMS
    chmod 0440 SHA256SUMS
    sha256sum --check SHA256SUMS
  )
}

prepare_manifest() {
  local model_id="$1"
  local root="${model_root}/${model_id}"
  mkdir -p -- "${root}"
  : >"${root}/.SHA256SUMS.tmp"
  chmod 0640 "${root}/.SHA256SUMS.tmp"
}

fetch_model_docs() {
  local model_id="$1"
  local repository="$2"
  local revision="$3"
  download_text "${model_id}" \
    "https://huggingface.co/${repository}/resolve/${revision}/README.md?download=true" \
    MODEL_CARD.md
}

install_apache_license() {
  download_text "$1" 'https://www.apache.org/licenses/LICENSE-2.0.txt' LICENSE
}

download_gemma() {
  local model_id="$1"
  local gguf_repository="$2"
  local gguf_revision="$3"
  local model_file="$4"
  local model_size="$5"
  local model_sha256="$6"
  local mmproj_file="$7"
  local mmproj_size="$8"
  local mmproj_sha256="$9"
  local source_repository="${10}"
  local source_revision="${11}"

  prepare_manifest "${model_id}"
  download_artifact "${model_id}" "${gguf_repository}" "${gguf_revision}" \
    "${model_file}" "${model_file}" "${model_size}" "${model_sha256}"
  download_artifact "${model_id}" "${gguf_repository}" "${gguf_revision}" \
    "${mmproj_file}" "${mmproj_file}" "${mmproj_size}" "${mmproj_sha256}"
  install_apache_license "${model_id}"
  fetch_model_docs "${model_id}" "${gguf_repository}" "${gguf_revision}"
  write_source_revision "${model_id}" \
    "gguf_repository=${gguf_repository}" \
    "gguf_revision=${gguf_revision}" \
    "source_model_repository=${source_repository}" \
    "source_model_revision=${source_revision}" \
    'quantization=Q4_0; Q8_0 vision projector' \
    'license=Apache-2.0'
  finish_model "${model_id}"
}

download_gemma gemma-4-e4b-it \
  ggml-org/gemma-4-E4B-it-GGUF \
  b8093469224f83f5c38f691eb906c380e9e63114 \
  gemma-4-E4B-it-Q4_0.gguf 4590807392 \
  a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2 \
  mmproj-gemma-4-E4B-it-Q8_0.gguf 559874816 \
  197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1 \
  google/gemma-4-E4B-it ee0ef6023621cff504d758262d4e04895a5af4a2

download_gemma gemma-4-12b-it \
  ggml-org/gemma-4-12B-it-GGUF \
  e3e681731089efaa3f0917336944ac64752db8ba \
  gemma-4-12B-it-Q4_0.gguf 7219673216 \
  3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70 \
  mmproj-gemma-4-12B-it-Q8_0.gguf 158987616 \
  59e62255435dda870e2d1de97cc031330b31a898bac12b38a182cecff9cd3738 \
  google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

download_gemma gemma-4-26b-a4b-it \
  ggml-org/gemma-4-26B-A4B-it-GGUF \
  bb4531cda34d1ea09d9814959ed4d5833cf2a4c8 \
  gemma-4-26B-A4B-it-Q4_0.gguf 14618145824 \
  d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03 \
  mmproj-gemma-4-26B-A4B-it-Q8_0.gguf 806408320 \
  cc4e855736da450bf1e162d8cccfe0ad685727d0c9e04ef7dd8d884f3121039b \
  google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

prepare_manifest qwen-image-2.1-uncensored
download_artifact qwen-image-2.1-uncensored \
  abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
  6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 \
  qwen-image-2.1-UC-Q4_K_M.gguf \
  diffusion_models/qwen-image-2.1-UC-Q4_K_M.gguf \
  4604558112 e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41
download_artifact qwen-image-2.1-uncensored \
  abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
  6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  9350798360 8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f
download_artifact qwen-image-2.1-uncensored \
  abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
  6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 \
  vae/qwen_image_2.1_vae_bf16.safetensors \
  vae/qwen_image_2.1_vae_bf16.safetensors \
  675509688 bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9
download_text qwen-image-2.1-uncensored \
  'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE?download=true' \
  LICENSE
fetch_model_docs qwen-image-2.1-uncensored \
  abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
  6b34e59458d3eb7ba6a6f86a116aed5253dc02c3
write_source_revision qwen-image-2.1-uncensored \
  'gguf_repository=abenzerps/Qwen-Image-2.1-Uncensored-GGUF' \
  'gguf_revision=6b34e59458d3eb7ba6a6f86a116aed5253dc02c3' \
  'source_model_repository=Qwen/Qwen-Image-2.1' \
  'source_model_revision=b3179ad355be050328e483a9dfdd9e60cd62adfa' \
  'text_encoder=Qwen3-VL-8B INT8 ConvRot' \
  'diffusion_model=Q4_K_M; VAE=BF16' \
  'license=Qwen Research License; non-commercial research/evaluation only' \
  'safety=the artifact has no built-in safety checker or content filter'
finish_model qwen-image-2.1-uncensored

prepare_manifest qwen3.6-35b-a3b
download_artifact qwen3.6-35b-a3b \
  ggml-org/Qwen3.6-35B-A3B-GGUF \
  baec3ebee244827cda0f4557eafa8b28f7545fa6 \
  Qwen3.6-35B-A3B-Q4_K_M.gguf \
  Qwen3.6-35B-A3B-Q4_K_M.gguf \
  20419565568 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7
download_artifact qwen3.6-35b-a3b \
  ggml-org/Qwen3.6-35B-A3B-GGUF \
  baec3ebee244827cda0f4557eafa8b28f7545fa6 \
  mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
  mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
  614194304 904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d
download_text qwen3.6-35b-a3b \
  'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/995ad96eacd98c81ed38be0c5b274b04031597b0/LICENSE?download=true' \
  LICENSE
fetch_model_docs qwen3.6-35b-a3b \
  ggml-org/Qwen3.6-35B-A3B-GGUF \
  baec3ebee244827cda0f4557eafa8b28f7545fa6
write_source_revision qwen3.6-35b-a3b \
  'gguf_repository=ggml-org/Qwen3.6-35B-A3B-GGUF' \
  'gguf_revision=baec3ebee244827cda0f4557eafa8b28f7545fa6' \
  'source_model_repository=Qwen/Qwen3.6-35B-A3B' \
  'source_model_revision=995ad96eacd98c81ed38be0c5b274b04031597b0' \
  'quantization=Q4_K_M; all experts retained in the GGUF' \
  'license=Apache-2.0'
finish_model qwen3.6-35b-a3b

prepare_manifest deepseek-v4.1-flash-q2
download_artifact deepseek-v4.1-flash-q2 \
  antirez/deepseek-v4.1-flash-gguf \
  dd8a266f7145edc19e2334b46e19b6821f221dc7 \
  DeepSeek-V4.1-Flash-Q2.gguf \
  DeepSeek-V4.1-Flash-Q2.gguf \
  365713686528 1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42
download_text deepseek-v4.1-flash-q2 \
  'https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/dd8a266f7145edc19e2334b46e19b6821f221dc7/LICENSE?download=true' \
  LICENSE
fetch_model_docs deepseek-v4.1-flash-q2 \
  antirez/deepseek-v4.1-flash-gguf \
  dd8a266f7145edc19e2334b46e19b6821f221dc7
write_source_revision deepseek-v4.1-flash-q2 \
  'repository=antirez/deepseek-v4.1-flash-gguf' \
  'revision=dd8a266f7145edc19e2334b46e19b6821f221dc7' \
  'source_model_repository=deepseek-ai/DeepSeek-V4.1-Flash' \
  'source_model_revision=df42c109f1defefcbfcedbe7d905718a12266e40' \
  'quantization=Q2 hybrid; 188.83 GiB FP8 Engram tables included' \
  'runtime_target=DwarfStar experimental DeepSeek-V4.1 path; not stock llama.cpp' \
  'license=MIT'
finish_model deepseek-v4.1-flash-q2

printf 'all evaluation model artifacts downloaded and SHA-256 verified\n'
