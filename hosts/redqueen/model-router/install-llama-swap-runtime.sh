#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly revision=fcefa7b7bebc326790937bbebc32ae87a958747c
readonly patch_sha256=92c47a1c5111ce5341c5debe501566c46c94718d2bea9bec7465145c4c759338
readonly target=/srv/ai/runtimes/llama-swap-v260-unbounded
readonly patch_file="$script_dir/patches/llama-swap-v260-unbounded-load.patch"
readonly prebuilt="${1:-}"
[[ "$(id -u)" -ne 0 ]] || { printf 'run this as ai\n' >&2; exit 1; }
printf '%s  %s\n' "$patch_sha256" "$patch_file" | sha256sum --check
if [[ -e "$target" ]]; then
  [[ "$(cat "$target/SOURCE_REVISION")" == "$revision" && \
    "$(cat "$target/PATCH_SHA256")" == "$patch_sha256" ]] || {
    printf 'existing router runtime does not match the pinned source and patch\n' >&2; exit 1;
  }
  (cd -- "$target" && sha256sum --check BINARY_SHA256)
  exit 0
fi

install -d -m 0755 /srv/ai/runtimes
readonly temporary="$(mktemp -d /srv/ai/runtimes/.llama-swap-stage.XXXXXXXX)"
trap 'rm -rf -- "$temporary"' EXIT
mkdir -p "$temporary/runtime/bin"
if [[ -n "$prebuilt" ]]; then
  # The prebuilt directory contains a reviewed cross-build and its manifest.
  # Never accept a naked binary or a manifest for another source/patch.
  [[ "$(cat "$prebuilt/SOURCE_REVISION")" == "$revision" && \
    "$(cat "$prebuilt/PATCH_SHA256")" == "$patch_sha256" ]] || {
    printf 'prebuilt router manifest does not match the pinned source and patch\n' >&2; exit 1;
  }
  (cd -- "$prebuilt" && sha256sum --check BINARY_SHA256)
  install -m 0755 "$prebuilt/bin/llama-swap" "$temporary/runtime/bin/llama-swap"
else
  [[ "$(go version)" == 'go version go1.27.1 linux/amd64' ]] || {
    printf 'build requires Go 1.27.1 linux/amd64, or pass a verified prebuilt directory\n' >&2; exit 1;
  }
  git clone --quiet --depth 1 --branch v260 https://github.com/mostlygeek/llama-swap.git "$temporary/source"
  [[ "$(git -C "$temporary/source" rev-parse HEAD)" == "$revision" ]] || {
    printf 'llama-swap tag does not match pinned source revision\n' >&2; exit 1;
  }
  git -C "$temporary/source" apply --check "$patch_file"
  git -C "$temporary/source" apply "$patch_file"
  (
    cd -- "$temporary/source"
    export CGO_ENABLED=0
    go build -o build/simple-responder ./cmd/simple-responder
    go test ./internal/config ./internal/process ./internal/router -short -count=1
    go build -trimpath \
      -ldflags='-s -w -X main.version=v260-redqueen1 -X main.commit=fcefa7b7-rq-unbounded -X main.date=2026-10-15T00:00:00Z' \
      -o "$temporary/runtime/bin/llama-swap" .
  )
fi
printf '%s\n' "$revision" > "$temporary/runtime/SOURCE_REVISION"
printf '%s\n' "$patch_sha256" > "$temporary/runtime/PATCH_SHA256"
printf 'Go 1.27.1; CGO_ENABLED=0; linux/amd64; v260-redqueen1; unbounded loading, bounded health probes\n' \
  > "$temporary/runtime/BUILD_OPTIONS"
(cd -- "$temporary/runtime" && sha256sum bin/llama-swap > BINARY_SHA256)
"$temporary/runtime/bin/llama-swap" -version
mv -T -- "$temporary/runtime" "$target"
printf 'installed %s; no services were restarted\n' "$target"
