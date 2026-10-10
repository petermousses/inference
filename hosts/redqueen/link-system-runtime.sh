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

install_exact_copy() {
  local source="$1"
  local target="$2"
  local source_real target_real temporary

  [[ -f "$source" ]] || {
    printf 'expected source file: %s\n' "$source" >&2
    return 1
  }
  if [[ -L "$target" ]]; then
    source_real="$(readlink -f -- "$source")"
    target_real="$(readlink -f -- "$target" || true)"
    [[ "$source_real" == "$target_real" ]] || {
      printf 'existing root link points elsewhere; preserving it: %s\n' "$target" >&2
      return 1
    }
  elif [[ -e "$target" ]]; then
    [[ -f "$target" ]] || {
      printf 'existing root path is not a regular file; preserving it: %s\n' "$target" >&2
      return 1
    }
    cmp -s -- "$source" "$target" || {
      printf 'root-owned file differs from source; preserving it: %s\n' "$target" >&2
      return 1
    }
  fi

  temporary="$target.inference-copy.$$"
  [[ ! -e "$temporary" && ! -L "$temporary" ]] || {
    printf 'temporary install path already exists; preserving it: %s\n' "$temporary" >&2
    return 1
  }
  install -o root -g root -m 0644 -- "$source" "$temporary"
  mv -Tf -- "$temporary" "$target"
  printf 'installed root-readable copy at %s from %s\n' "$target" "$source"
}

install_exact_copy "$host_root/nftables.d/redqueen-inference.nft" \
  /etc/nftables.d/redqueen-inference.nft
link_exact "$host_root/systemd-system/redqueen-inference-firewall.service" \
  /etc/systemd/system/redqueen-inference-firewall.service
systemctl daemon-reload
