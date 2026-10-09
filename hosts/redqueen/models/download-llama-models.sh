#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to download model files as root\n' >&2
  exit 1
fi

download() {
  local model_root="$1"
  local url="$2"
  local relative_path="$3"
  local expected_sha256="$4"
  local destination="${model_root}/${relative_path}"
  local partial="${destination}.part"
  local http_status

  mkdir -p -- "$(dirname -- "${destination}")"
  if [[ -f "${destination}" ]] && printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status; then
    printf 'verified existing %s\n' "${destination}"
    return
  fi
  if [[ -L "${destination}" ]]; then
    printf 'linked destination differs from the pinned artifact; preserving source link: %s\n' \
      "${destination}" >&2
    return 1
  fi

  if [[ -f "${partial}" ]] && printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status; then
    mv -- "${partial}" "${destination}"
    chmod 0440 "${destination}"
    printf 'promoted verified partial %s\n' "${destination}"
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
  printf 'downloaded and verified %s\n' "${destination}"
}

install_metadata() {
  local source="$1"
  local destination="$2"
  if [[ -L "${destination}" ]]; then
    cmp -s -- "${source}" "${destination}" || {
      printf 'linked model metadata differs from its pinned source: %s\n' \
        "${destination}" >&2
      return 1
    }
    return 0
  fi
  install -m 0440 -- "${source}" "${destination}"
}

write_source_revision() {
  local model_root="$1"
  shift
  local destination="${model_root}/SOURCE_REVISION"
  local temporary
  temporary="$(mktemp "${model_root}/.SOURCE_REVISION.XXXXXXXX")"
  printf '%s\n' "$@" >"${temporary}"
  chmod 0440 "${temporary}"
  if [[ -L "${destination}" ]]; then
    cmp -s -- "${temporary}" "${destination}" || {
      rm -f -- "${temporary}"
      printf 'linked source revision differs from pinned source metadata: %s\n' \
        "${destination}" >&2
      return 1
    }
    rm -- "${temporary}"
  else
    mv -- "${temporary}" "${destination}"
  fi
}

download_qwen() {
  local model_root="${QWEN_MODEL_ROOT:-/srv/ai/models/qwen3.8-27b}"
  local gguf_revision="71bc7b627595dc8a91039addd9c791ae548d6747"
  local source_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
  local gguf_base="https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/${gguf_revision}"
  local source_base="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/${source_revision}"

  mkdir -p -- "${model_root}"
  exec 8>"${model_root}/.download.lock"
  flock -n 8 || {
    printf 'another Qwen3.8 download is already running\n' >&2
    return 1
  }

  download "${model_root}" "${gguf_base}/Qwen3.8-27B-Q4_K_M.gguf" \
    Qwen3.8-27B-Q4_K_M.gguf \
    c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747
  download "${model_root}" "${gguf_base}/mmproj-Qwen3.8-27B-Q8_0.gguf" \
    mmproj-Qwen3.8-27B-Q8_0.gguf \
    2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb
  download "${model_root}" "${source_base}/LICENSE" \
    LICENSE \
    bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
  download "${model_root}" "${source_base}/README.md" \
    QWEN_MODEL_CARD.md \
    57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641
  download "${model_root}" "${gguf_base}/README.md" \
    GGUF_MODEL_CARD.md \
    4a1ddebc85fd59b14650bb04757dea42cfefa63b75173e15fd5bc919d2861adb

  install_metadata "${script_dir}/qwen3.8-27b.sha256" "${model_root}/SHA256SUMS"
  write_source_revision "${model_root}" \
    'gguf_repository=ggml-org/Qwen3.8-27B-GGUF' \
    "gguf_revision=${gguf_revision}" \
    'source_repository=Qwen/Qwen3.8-27B' \
    "source_revision=${source_revision}"
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_jevk5() {
  local model_root="${JEVK5_MODEL_ROOT:-/srv/ai/models/jevk5-4b-v0.3}"
  local gguf_revision="ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a"
  local source_revision="f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a"
  local gguf_base="https://huggingface.co/alibiserikbay/JevK5-GGUF/resolve/${gguf_revision}"
  local source_base="https://raw.githubusercontent.com/allebee/jevk5/${source_revision}"

  mkdir -p -- "${model_root}"
  exec 7>"${model_root}/.download.lock"
  flock -n 7 || {
    printf 'another JevK5 download is already running\n' >&2
    return 1
  }

  download "${model_root}" "${gguf_base}/jevk5-4b-v0.3-Q8_0.gguf" \
    jevk5-4b-v0.3-Q8_0.gguf \
    aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30
  download "${model_root}" "${source_base}/LICENSE" \
    LICENSE \
    cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
  download "${model_root}" "${gguf_base}/README.md" \
    MODEL_CARD.md \
    5c75f91401539afe2ce722e87f8e81f1bbd23cb3cea16758180be4a316389fbe
  download "${model_root}" "${gguf_base}/SHA256SUMS" \
    UPSTREAM_SHA256SUMS \
    ad21b6c0b7a21fb6cc6980495a91bb2a1c97078beacfc065d4f19e934d6d9422

  install_metadata "${script_dir}/jevk5-4b-v0.3.sha256" "${model_root}/SHA256SUMS"
  write_source_revision "${model_root}" \
    'gguf_repository=alibiserikbay/JevK5-GGUF' \
    "gguf_revision=${gguf_revision}" \
    'source_repository=allebee/jevk5' \
    "source_revision=${source_revision}"
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_embeddinggemma() {
  local model_root="${EMBEDDINGGEMMA_MODEL_ROOT:-/srv/ai/models/embeddinggemma-2}"
  local gguf_revision="ba3888272494be64ed88c9eb536ddc61a1be73d5"
  local source_revision="914f7f89142e33e77833254d9c9b90c3cef7303b"
  local gguf_base="https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/${gguf_revision}"

  mkdir -p -- "${model_root}"
  exec 6>"${model_root}/.download.lock"
  flock -n 6 || {
    printf 'another EmbeddingGemma 2 download is already running\n' >&2
    return 1
  }

  download "${model_root}" "${gguf_base}/embeddinggemma-2-Q8_0.gguf" \
    embeddinggemma-2-Q8_0.gguf \
    6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc
  download "${model_root}" "${gguf_base}/README.md" \
    MODEL_CARD.md \
    e153a1a36202b8fb1f591a2aa593673fb0cff9bd2d81b582974d79a7a988b199
  if ! printf '%s  %s\n' \
    cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 \
    /usr/share/common-licenses/Apache-2.0 | sha256sum --check --status; then
    printf 'the host Apache-2.0 license text is missing or changed\n' >&2
    return 1
  fi
  install_metadata /usr/share/common-licenses/Apache-2.0 "${model_root}/LICENSE"

  install_metadata "${script_dir}/embeddinggemma-2.sha256" "${model_root}/SHA256SUMS"
  write_source_revision "${model_root}" \
    'gguf_repository=unsloth/embeddinggemma-2-GGUF' \
    "gguf_revision=${gguf_revision}" \
    'source_repository=google/embeddinggemma-2' \
    "source_revision=${source_revision}" \
    'quantization=Q8_0'
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_large_model_file() {
  local model_root="$1"
  local repository="$2"
  local revision="$3"
  local filename="$4"
  local expected_size="$5"
  local expected_sha256="$6"
  local destination="${model_root}/${filename}"
  local partial="${destination}.part"
  local actual_size=0 range_start range_end range_size range_expected range_path
  local url="https://huggingface.co/${repository}/resolve/${revision}/${filename}?download=true"

  if [[ -e "${destination}" ]]; then
    [[ -f "${destination}" ]] || {
      printf 'refusing to replace non-file model artifact: %s\n' "${destination}" >&2
      return 1
    }
    [[ "$(stat -c '%s' -- "${destination}")" == "${expected_size}" ]] && \
      printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status || {
        printf 'existing large model artifact has the wrong size or SHA-256; preserving it: %s\n' \
          "${destination}" >&2
        return 1
      }
    printf 'verified existing %s\n' "${destination}"
    return
  fi
  [[ ! -e "${partial}" || -f "${partial}" ]] || {
    printf 'refusing to replace non-file partial: %s\n' "${partial}" >&2
    return 1
  }
  if [[ -f "${partial}" ]]; then
    actual_size="$(stat -c '%s' -- "${partial}")"
    if (( actual_size > expected_size )); then
      printf 'partial exceeds expected size; preserving it: %s\n' "${partial}" >&2
      return 1
    fi
    if (( actual_size == expected_size )); then
      printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status || {
        printf 'complete partial has the wrong SHA-256; preserving it: %s\n' "${partial}" >&2
        return 1
      }
      mv -- "${partial}" "${destination}"
      chmod 0440 "${destination}"
      printf 'promoted verified partial %s\n' "${destination}"
      return
    fi
    printf 'resuming %s at byte %s\n' "${destination}" "${actual_size}"
  fi

  while (( actual_size < expected_size )); do
    range_start="${actual_size}"
    range_end=$((range_start + 1073741823))
    if (( range_end >= expected_size )); then
      range_end=$((expected_size - 1))
    fi
    range_expected=$((range_end - range_start + 1))
    range_path="${partial}.range.${range_start}"
    if [[ -f "${range_path}" ]]; then
      range_size="$(stat -c '%s' -- "${range_path}")"
    else
      range_size=0
    fi
    if (( range_size != range_expected )); then
      rm -f -- "${range_path}"
      curl \
        --fail \
        --location \
        --retry 10 \
        --retry-all-errors \
        --retry-delay 5 \
        --connect-timeout 20 \
        --range "${range_start}-${range_end}" \
        --output "${range_path}" \
        "${url}"
      range_size="$(stat -c '%s' -- "${range_path}")"
    fi
    if (( range_size != range_expected )); then
      rm -f -- "${range_path}"
      printf 'Hugging Face returned %s bytes for range %s-%s; expected %s\n' \
        "${range_size}" "${range_start}" "${range_end}" "${range_expected}" >&2
      return 1
    fi
    cat -- "${range_path}" >>"${partial}"
    rm -- "${range_path}"
    actual_size=$((actual_size + range_size))
    printf 'downloaded %s / %s bytes of %s\n' \
      "${actual_size}" "${expected_size}" "${filename}"
  done

  printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --status || {
    printf 'downloaded large model artifact has the wrong SHA-256; preserving it: %s\n' \
      "${partial}" >&2
    return 1
  }
  mv -- "${partial}" "${destination}"
  chmod 0440 "${destination}"
  printf 'downloaded and verified %s\n' "${destination}"
}

download_clef_flash() {
  local model_root="${CLEF_FLASH_MODEL_ROOT:-/srv/ai/models/clef-flash}"
  local gguf_revision="4a192915ef971886004b5b13294f2b4c7a7fc39d"
  local source_revision="17f0b0ad64efb65d273590632833508766b2aae6"
  local gguf_base="https://huggingface.co/ggml-org/Clef-Flash-GGUF/resolve/${gguf_revision}"
  local source_base="https://huggingface.co/Cloudflare/clef-flash/resolve/${source_revision}"

  mkdir -p -- "${model_root}"
  exec 5>"${model_root}/.download.lock"
  flock -n 5 || {
    printf 'another Clef-Flash download is already running\n' >&2
    return 1
  }

  download_large_model_file "${model_root}" ggml-org/Clef-Flash-GGUF "${gguf_revision}" \
    Clef-Flash-BF16.gguf 18164488352 \
    92ecea391fdf03b514dd43abc1ba9d62d984047092fed97d985b3e513e3a8be6
  download_large_model_file "${model_root}" ggml-org/Clef-Flash-GGUF "${gguf_revision}" \
    Clef-Flash-Q8_0.gguf 9657260192 \
    d7c352faf1bdd9ea24d0b9347e8eb1eb4bbadeff6c02383bf750215a74f2f1f1
  download_large_model_file "${model_root}" ggml-org/Clef-Flash-GGUF "${gguf_revision}" \
    Clef-Flash-Q4_K_M.gguf 6486448288 \
    fd3e90605e8103307dca37cb5a8cdb036267e2fe3cb2d908d80a8ceb9ec0638c
  download "${model_root}" "${gguf_base}/mmproj-Clef-Flash-Q8_0.gguf" \
    mmproj-Clef-Flash-Q8_0.gguf \
    3fbc646617c56c35ba48e06f0fbe8693a83bde49eb31e2a3bf59ba0a308c9e25
  download "${model_root}" "${source_base}/LICENSE" LICENSE \
    bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
  download "${model_root}" "${source_base}/README.md" SOURCE_MODEL_CARD.md \
    4aac13b2563900152eab8feaad0b310a91b55a96135abf4ad17ad4a302bd73e6
  download "${model_root}" "${gguf_base}/README.md" MODEL_CARD.md \
    30b244b7a978fe38d7d859c049b9b229470de276ec51ba918f758fec2a79ba80

  install_metadata "${script_dir}/clef-flash.sha256" "${model_root}/SHA256SUMS"
  write_source_revision "${model_root}" \
    'gguf_repository=ggml-org/Clef-Flash-GGUF' \
    "gguf_revision=${gguf_revision}" \
    'source_repository=Cloudflare/clef-flash' \
    "source_revision=${source_revision}" \
    'precision=BF16, Q8_0, Q4_K_M'
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_clef() {
  local model_root="${CLEF_MODEL_ROOT:-/srv/ai/models/clef}"
  local gguf_revision="63840a1a68cb7084c88610cffc328509356b04cb"
  local source_revision="2f3de3dd85f379784083b0814d997ab627200f0c"
  local gguf_base="https://huggingface.co/ggml-org/Clef-GGUF/resolve/${gguf_revision}"
  local source_base="https://huggingface.co/Cloudflare/clef/resolve/${source_revision}"

  mkdir -p -- "${model_root}"
  exec 5>"${model_root}/.download.lock"
  flock -n 5 || {
    printf 'another Clef download is already running\n' >&2
    return 1
  }

  download_large_model_file "${model_root}" ggml-org/Clef-GGUF "${gguf_revision}" \
    Clef-Q4_K_M.gguf 19232219200 \
    bd9b8ae24c5752a2bb5e2733d5168340d5f04457e02b8bf0d1bc9285ec3d4f61
  download "${model_root}" "${gguf_base}/mmproj-Clef-Q8_0.gguf" \
    mmproj-Clef-Q8_0.gguf \
    0d901ea999ae122ba0afeb4a3f4c4ac9a90baa07f1d120df9c350082db1190a8
  download "${model_root}" "${source_base}/LICENSE" LICENSE \
    bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
  download "${model_root}" "${source_base}/README.md" SOURCE_MODEL_CARD.md \
    b0211b6ca10038b3168dde51f1482088b8c6b0fdccbbd765fce237dbc24f5d76
  download "${model_root}" "${gguf_base}/README.md" MODEL_CARD.md \
    c3a5cc3e2092dd6dd6dd10bf8cd878b02c973333588a38ac22aefca1fbfe98f6

  install_metadata "${script_dir}/clef.sha256" "${model_root}/SHA256SUMS"
  write_source_revision "${model_root}" \
    'gguf_repository=ggml-org/Clef-GGUF' \
    "gguf_revision=${gguf_revision}" \
    'source_repository=Cloudflare/clef' \
    "source_revision=${source_revision}" \
    'quantization=Q4_K_M'
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_clm() {
  local model_root="${CLM_MODEL_ROOT:-/srv/ai/models/clm-v0.1-8b}"
  local revision="e939398d4556fcd9400c76fa8c5a513202f42b0a"
  local base="https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/${revision}"
  local encoder_revision="2473489dc243ccaffb4ce569c55bf1df66b2088f"
  local encoder_base="https://huggingface.co/ggml-org/Qwen3-8B-GGUF/resolve/${encoder_revision}"

  mkdir -p -- "${model_root}"
  exec 4>"${model_root}/.download.lock"
  flock -n 4 || {
    printf 'another CLM download is already running\n' >&2
    return 1
  }

  download "${model_root}" "${base}/CLM_v0.1-8B.pt" \
    CLM_v0.1-8B.pt \
    b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5
  download "${model_root}" "${base}/LICENSE" LICENSE \
    bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
  download_large_model_file "${model_root}" ggml-org/Qwen3-8B-GGUF \
    "${encoder_revision}" Qwen3-8B-f16.gguf 16388043968 \
    dbc8b6ba7073bf9543abfa30f33103170aba5c9ccda3d64944c7484b4839e2fd
  download "${model_root}" "${encoder_base}/LICENSE" ENCODER_LICENSE \
    bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a

  install_metadata "${script_dir}/clm-v0.1-8b.sha256" \
    "${model_root}/SHA256SUMS"
  install_metadata "${script_dir}/metadata/clm-v0.1-8b/MODEL_CARD.md" \
    "${model_root}/MODEL_CARD.md"
  write_source_revision "${model_root}" \
    'repository=Contrastive-LM/CLM-v0.1-8B' \
    "revision=${revision}" \
    'checkpoint=CLM_v0.1-8B.pt' \
    'checkpoint_sha256=b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5' \
    'package=contrastive-lm==0.1.0' \
    'encoder_gguf_repository=ggml-org/Qwen3-8B-GGUF' \
    "encoder_gguf_revision=${encoder_revision}" \
    'encoder_gguf_file=Qwen3-8B-f16.gguf' \
    'encoder_gguf_sha256=dbc8b6ba7073bf9543abfa30f33103170aba5c9ccda3d64944c7484b4839e2fd' \
    'encoder_gguf_size=16388043968' \
    'encoder_source_repository=Qwen/Qwen3-8B' \
    'encoder_source_revision=b968826d9c46dd6066d109eabc6255188de91218' \
    'encoder_model=qwen3-8b' \
    'encoder_pooling=last-token' \
    'encoder_max_tokens=2048'
  (cd -- "${model_root}" && sha256sum --check SHA256SUMS)
}

download_qwen
download_jevk5
download_embeddinggemma
download_clef_flash
download_clef
download_clm
