#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly script_path="$(readlink -f -- "$0")"
readonly host_root="$(cd -- "$(dirname -- "$script_path")" && pwd -P)"
readonly repo_root="$(cd -- "$host_root/../.." && pwd -P)"
backup_root=''

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this as ai, not root\n' >&2
  exit 1
fi
[[ -d "$repo_root/.git" ]] || {
  printf 'expected the inference Git checkout at %s\n' "$repo_root" >&2
  exit 1
}

backup_target() {
  local target="$1"
  local relative backup
  if [[ -z "$backup_root" ]]; then
    backup_root="$(mktemp -d /srv/ai/archive-state/.inference-link-backup.XXXXXXXX)"
  fi
  relative="$(printf '%s' "$target" | sed 's|^/||')"
  backup="$backup_root/$relative"
  mkdir -p -- "$(dirname -- "$backup")"
  mv -- "$target" "$backup"
  printf 'preserved prior path at %s\n' "$backup"
}

link_path() {
  local source="$1"
  local target="$2"
  local kind="$3"
  local replace_mismatch="$4"
  local resolved_source resolved_target temporary

  if [[ "$kind" == file && ! -f "$source" ]]; then
    printf 'source file is missing: %s\n' "$source" >&2
    return 1
  fi
  if [[ "$kind" == dir && ! -d "$source" ]]; then
    printf 'source directory is missing: %s\n' "$source" >&2
    return 1
  fi

  mkdir -p -- "$(dirname -- "$target")"
  if [[ -L "$target" ]]; then
    resolved_source="$(readlink -f -- "$source")"
    resolved_target="$(readlink -f -- "$target" || true)"
    if [[ "$resolved_target" == "$resolved_source" ]]; then
      return 0
    fi
  fi

  if [[ -e "$target" || -L "$target" ]]; then
    if [[ "$kind" == file && -f "$target" && ! -L "$target" ]] && \
      cmp -s -- "$source" "$target"; then
      rm -- "$target"
    else
      if [[ "$kind" == file && "$replace_mismatch" != yes ]]; then
        printf 'existing file differs from source; preserving it and stopping: %s\n' \
          "$target" >&2
        return 1
      fi
      backup_target "$target"
    fi
  fi

  temporary="$target.inference-link.$$"
  [[ ! -e "$temporary" && ! -L "$temporary" ]] || {
    printf 'temporary link path already exists: %s\n' "$temporary" >&2
    return 1
  }
  ln -s -- "$source" "$temporary"
  mv -Tf -- "$temporary" "$target"
  printf 'linked %s -> %s\n' "$target" "$source"
}

link_path "$repo_root" /srv/ai/config dir no

link_path "$host_root/adapters/redqueen_adapters" \
  /srv/ai/adapters/source/redqueen_adapters dir no
link_path "$host_root/adapters/workflows" /srv/ai/adapters/workflows dir no
link_path "$repo_root/LICENSE" /srv/ai/adapters/LICENSE file no
link_path "$host_root/adapters/NOTICE" /srv/ai/adapters/NOTICE file no
link_path "$host_root/adapters/requirements.lock" \
  /srv/ai/adapters/requirements.lock file no

link_path "$host_root/model-router/config.yaml" \
  /srv/ai/model-router/config.yaml file no
link_path "$host_root/model-router/scheduler.json" \
  /srv/ai/model-router/scheduler.json file no
link_path "$host_root/model-router/requirements.lock" \
  /srv/ai/model-router/requirements.lock file no
link_path "$host_root/model-router/comfyui-control.py" \
  /srv/ai/model-router/bin/comfyui-control.py file no
link_path "$host_root/model-router/glm-5.3-flash-abliterated-launcher.sh" \
  /srv/ai/model-router/bin/glm-5.3-flash-abliterated-launcher.sh file no
link_path "$host_root/model-router/THIRD-PARTY-LICENSE.md" \
  /srv/ai/model-router/THIRD-PARTY-LICENSE.md file no

# This MUST be a regular copy. A candidate checkout must not replace the
# helper that rolls back that candidate after SIGKILL, power loss or reboot.
# Successful deployment promotes the new helper; the linker only bootstraps.
/usr/bin/python3 "$host_root/adapters/redqueen_adapters/host_reliability.py" \
  install --source "$host_root/adapters/redqueen_adapters/host_reliability.py" \
  --bootstrap-only
/usr/bin/python3 "$host_root/adapters/redqueen_adapters/host_reliability.py" \
  unit --source "$host_root/systemd/redqueen-preflight.service" --bootstrap-only

link_path "$host_root/comfyui/download-qwen-image-2.1.sh" \
  /srv/ai/comfyui/download-qwen-image-2.1.sh file yes
link_path "$host_root/comfyui/extra-model-paths.yaml" \
  /srv/ai/comfyui/extra-model-paths.yaml file yes
link_path "$host_root/comfyui/custom_nodes/redqueen_admission_guard" \
  /srv/ai/comfyui/state/custom_nodes/redqueen_admission_guard dir no
link_path "$host_root/comfyui/prepare-qwen-image-2.1-workflows.sh" \
  /srv/ai/comfyui/prepare-qwen-image-2.1-workflows.sh file yes
link_path "$host_root/comfyui/qwen-image-2.1.sha256" \
  /srv/ai/comfyui/qwen-image-2.1.sha256 file no
link_path "$host_root/comfyui/qwen-image-2.1-workflows.sha256" \
  /srv/ai/comfyui/qwen-image-2.1-workflows.sha256 file no
link_path "$host_root/comfyui/requirements.lock" \
  /srv/ai/comfyui/requirements.lock file yes
link_path "$host_root/comfyui/versions.env" \
  /srv/ai/comfyui/versions.env file yes
link_path "$host_root/comfyui/workflows" /srv/ai/comfyui/workflows dir no

link_path "$host_root/archive/backup.sh" \
  /srv/ai/bin/redqueen-archive-backup.sh file no
link_path "$host_root/archive/passcommand.sh" \
  /srv/ai/bin/redqueen-archive-passcommand.sh file no

for name in clef-flash.sha256 clef.sha256 clm-v0.1-8b.sha256 \
  embeddinggemma-2.sha256 jevk5-4b-v0.3.sha256 qwen3.8-27b.sha256 versions.env \
  download-llama-models.sh; do
  replace=no
  [[ "$name" != download-llama-models.sh ]] || replace=yes
  link_path "$host_root/models/$name" "/srv/ai/$name" file "$replace"
done

for metadata in "$host_root"/models/metadata/*/*; do
  [[ -f "$metadata" ]] || continue
  model_id="$(basename -- "$(dirname -- "$metadata")")"
  name="$(basename -- "$metadata")"
  if [[ "$model_id" == deepseek-v4.1-flash-1017 && "$name" == SOURCE_REVISION ]]; then
    /usr/bin/python3 "$host_root/models/materialize_deepseek_revision.py" \
      "$metadata" "/srv/ai/models/$model_id/$name"
  else
    link_path "$metadata" "/srv/ai/models/$model_id/$name" file no
  fi
done

for runtime in \
  llama.cpp-v0.6.0-d812350 \
  llama.cpp-b11457-5ad1c5d; do
  for name in SOURCE_REVISION BUILD_OPTIONS; do
    link_path "$host_root/model-router/runtime-manifests/$runtime/$name" \
      "/srv/ai/runtimes/$runtime/$name" file no
  done
done

for unit in \
  ai-inference.slice \
  clm-api.service \
  clm-encoder.service \
  comfyui-control.service \
  comfyui.service \
  deepseek-hip.service \
  jevk5-adapter.service \
  jevk5.service \
  llama-swap.service \
  qwen-image-adapter.service \
  qwen38.service \
  redqueen-archive.service \
  redqueen-archive.timer \
  redqueen-supervisor.service \
  redqueen-scheduler.service; do
  replace=no
  [[ "$unit" != comfyui.service ]] || replace=yes
  link_path "$host_root/systemd/$unit" \
    "/home/ai/.config/systemd/user/$unit" file "$replace"
done

systemctl --user daemon-reload

for target in \
  /etc/nftables.d/redqueen-inference.nft \
  /etc/systemd/system/redqueen-inference-firewall.service; do
  if [[ -e "$target" && ! -L "$target" ]]; then
    printf 'root-owned source link still pending: %s\n' "$target" >&2
  fi
done

printf 'Redqueen user-owned runtime files now resolve to %s\n' "$repo_root"
