#!/usr/bin/env bash
set -euo pipefail

readonly script_path="$(readlink -f -- "$0")"
readonly host_root="$(cd -- "$(dirname -- "$script_path")" && pwd -P)"

if [[ "$(id -u)" -ne 0 ]]; then
  printf 'run this script with sudo; it only links root-owned inference files\n' >&2
  exit 1
fi

link_exact() {
  local source="$1"
  local target="$2"
  local source_real target_real temporary

  [[ -f "$source" && -f "$target" ]] || {
    printf 'expected source and target files: %s, %s\n' "$source" "$target" >&2
    return 1
  }
  if [[ -L "$target" ]]; then
    source_real="$(readlink -f -- "$source")"
    target_real="$(readlink -f -- "$target" || true)"
    [[ "$source_real" == "$target_real" ]] && return 0
    printf 'existing root link points elsewhere; preserving it: %s\n' "$target" >&2
    return 1
  fi
  cmp -s -- "$source" "$target" || {
    printf 'root-owned file differs from source; preserving it: %s\n' "$target" >&2
    return 1
  }
  rm -- "$target"
  temporary="$target.inference-link.$$"
  ln -s -- "$source" "$temporary"
  mv -Tf -- "$temporary" "$target"
  printf 'linked %s -> %s\n' "$target" "$source"
}

link_exact "$host_root/nftables.d/redqueen-inference.nft" \
  /etc/nftables.d/redqueen-inference.nft
link_exact "$host_root/systemd-system/redqueen-inference-firewall.service" \
  /etc/systemd/system/redqueen-inference-firewall.service
systemctl daemon-reload
