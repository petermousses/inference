#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly workflow_root="${WORKFLOW_ROOT:-/srv/ai/cache/step2-gates}"
readonly state_root="${COMFY_STATE_ROOT:-/srv/ai/comfyui/state}"
readonly api_url="${COMFY_API_URL:-http://127.0.0.1:8189}"
readonly python_bin="${COMFY_PYTHON:-/srv/ai/comfyui/venv/bin/python}"
readonly run_id="$(date -u +%Y%m%dT%H%M%SZ)"
readonly run_dir="${RUN_BASE:-/srv/ai/cache/step2-image-gates}/${run_id}"

if [[ "$(id -u)" -eq 0 ]]; then
  printf 'refusing to run the image gates as root\n' >&2
  exit 1
fi
for command in curl jq systemctl sha256sum; do
  if ! command -v "${command}" >/dev/null; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  printf 'ComfyUI Python is not executable: %s\n' "${python_bin}" >&2
  exit 1
fi

declare -A workflows=(
  [baseline]="${workflow_root}/qwen-image-2.1-t2i-smoke-api.json"
  [transparency]="${workflow_root}/qwen-image-2.1-transparency-smoke-api.json"
  [multiref]="${workflow_root}/qwen-image-2.1-multiref-smoke-api.json"
  [edit_1024]="${workflow_root}/qwen-image-2.1-edit-1024-smoke-api.json"
  [t2i_2k]="${workflow_root}/qwen-image-2.1-t2i-2k-smoke-api.json"
)
for gate in baseline transparency multiref edit_1024 t2i_2k; do
  workflow="${workflows[${gate}]}"
  if [[ ! -r "${workflow}" ]] || ! jq -e . "${workflow}" >/dev/null; then
    printf 'workflow is missing, unreadable or invalid: %s\n' "${workflow}" >&2
    exit 1
  fi
done

mkdir -p -- "${run_dir}" "${state_root}/input/smoke"
systemctl --user restart comfyui.service
for _ in $(seq 1 90); do
  if curl -fsS "${api_url}/system_stats" >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "${api_url}/system_stats" >/dev/null

readonly starting_restarts="$(systemctl --user show comfyui.service -p NRestarts --value)"

submit_workflow() {
  local gate="$1"
  local workflow="$2"
  local response_file="${run_dir}/${gate}.response.json"
  local history_file="${run_dir}/${gate}.history.json"
  local prompt_id image_json filename subfolder image_type

  jq -c '{prompt:.}' "${workflow}" \
    | curl -fsS \
        --connect-timeout 5 \
        --max-time 30 \
        -H 'Content-Type: application/json' \
        --data-binary @- \
        --output "${response_file}" \
        "${api_url}/prompt"
  prompt_id="$(jq -er '.prompt_id' "${response_file}")"
  printf '%s prompt_id=%s start=%s\n' \
    "${gate}" "${prompt_id}" "$(date --iso-8601=seconds)" >&2

  for _ in $(seq 1 600); do
    curl -fsS \
      --connect-timeout 5 \
      --max-time 30 \
      --output "${history_file}" \
      "${api_url}/history/${prompt_id}"
    if jq -e --arg id "${prompt_id}" '.[ $id ]' "${history_file}" >/dev/null; then
      if ! jq -e --arg id "${prompt_id}" \
          '.[ $id ].status.status_str == "success"' "${history_file}" >/dev/null; then
        jq --arg id "${prompt_id}" '.[ $id ].status' "${history_file}" >&2
        return 1
      fi
      image_json="$(jq -cer --arg id "${prompt_id}" \
        '[.[ $id ].outputs[]?.images[]?][0]' "${history_file}")"
      filename="$(jq -er '.filename' <<<"${image_json}")"
      subfolder="$(jq -er '.subfolder' <<<"${image_json}")"
      image_type="$(jq -er '.type' <<<"${image_json}")"
      if [[ "${image_type}" != output \
        || "${subfolder}" != smoke \
        || "${filename}" == */* \
        || "${filename}" == *..* ]]; then
        printf 'unsafe or unexpected ComfyUI output descriptor: %s\n' "${image_json}" >&2
        return 1
      fi
      printf '%s/%s/%s\n' "${state_root}/output" "${subfolder}" "${filename}"
      return 0
    fi
    sleep 2
  done
  printf 'timed out waiting for %s (%s)\n' "${gate}" "${prompt_id}" >&2
  return 1
}

validate_png() {
  local path="$1"
  local width="$2"
  local height="$3"
  local gate="$4"
  local reference_path="${5:-}"
  "${python_bin}" - \
    "${path}" "${width}" "${height}" "${gate}" "${reference_path}" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageStat

path = Path(sys.argv[1])
expected_size = (int(sys.argv[2]), int(sys.argv[3]))
gate = sys.argv[4]
reference_path = Path(sys.argv[5]) if sys.argv[5] else None
with Image.open(path) as image:
    image.load()
    if image.format != "PNG":
        raise SystemExit(f"{gate}: expected PNG, got {image.format}")
    if image.size != expected_size:
        raise SystemExit(f"{gate}: expected {expected_size}, got {image.size}")
    rgba = image.convert("RGBA")
    extrema = rgba.convert("RGB").getextrema()
    if not any(low < high for low, high in extrema):
        raise SystemExit(f"{gate}: output is a constant-color image")

    center = rgba.convert("RGB").crop(
        (expected_size[0] // 4, expected_size[1] // 4,
         expected_size[0] * 3 // 4, expected_size[1] * 3 // 4)
    )
    center_mean = ImageStat.Stat(center).mean
    if gate == "baseline" and not (
        center_mean[0] > center_mean[1] * 1.05
        and center_mean[0] > center_mean[2] * 1.05
    ):
        raise SystemExit(f"baseline: center is not red-dominant: {center_mean}")
    if gate == "transparency":
        alpha = rgba.getchannel("A")
        histogram = alpha.histogram()
        if histogram[0] == 0 or histogram[255] == 0 or sum(histogram[1:255]) == 0:
            raise SystemExit(
                "transparency: expected fully transparent, partially transparent, "
                "and opaque pixels"
            )
        print(
            f"alpha_zero={histogram[0]} "
            f"alpha_partial={sum(histogram[1:255])} "
            f"alpha_opaque={histogram[255]}"
        )
    if gate in {"multiref", "edit_1024"} and not (
        center_mean[1] > center_mean[0] * 1.05
        and center_mean[1] > center_mean[2] * 1.05
    ):
        raise SystemExit(f"{gate}: center did not acquire the green appearance: {center_mean}")
    if gate == "multiref":
        swatch = (24.0, 180.0, 72.0)
        swatch_total = sum(swatch)
        center_total = sum(center_mean)
        center_chromaticity = tuple(value / center_total for value in center_mean)
        swatch_chromaticity = tuple(value / swatch_total for value in swatch)
        color_distance = sum(
            (observed - expected) ** 2
            for observed, expected in zip(center_chromaticity, swatch_chromaticity)
        ) ** 0.5
        saturation = (
            (max(center_mean) - min(center_mean)) / max(center_mean)
            if max(center_mean) else 0.0
        )
        if color_distance > 0.12 or saturation < 0.35:
            raise SystemExit(
                f"multiref: center does not match swatch chromaticity: "
                f"distance={color_distance:.4f} saturation={saturation:.4f}"
            )
        print(
            f"gate=multiref swatch_distance={color_distance:.4f} "
            f"saturation={saturation:.4f}"
        )
        corner_size = max(8, expected_size[0] // 16)
        corners = [
            (0, 0, corner_size, corner_size),
            (expected_size[0] - corner_size, 0, expected_size[0], corner_size),
            (0, expected_size[1] - corner_size, corner_size, expected_size[1]),
            (expected_size[0] - corner_size, expected_size[1] - corner_size,
             expected_size[0], expected_size[1]),
        ]
        corner_mean = sum(
            sum(ImageStat.Stat(rgba.convert("RGB").crop(box)).mean) / 3
            for box in corners
        ) / len(corners)
        if corner_mean < 200:
            raise SystemExit(
                f"multiref: background did not preserve image-1 composition: {corner_mean}"
            )
    if reference_path is not None:
        with Image.open(reference_path) as reference_image:
            reference = reference_image.convert("RGB").resize(
                expected_size, Image.Resampling.LANCZOS
            )
        output = rgba.convert("RGB")
        reference_mask = [min(pixel) < 230 for pixel in reference.getdata()]
        output_mask = [min(pixel) < 230 for pixel in output.getdata()]
        intersection = sum(
            left and right for left, right in zip(reference_mask, output_mask)
        )
        union = sum(left or right for left, right in zip(reference_mask, output_mask))
        structure_iou = intersection / union if union else 0.0
        minimum_iou = 0.85 if gate == "multiref" else 0.90
        if structure_iou < minimum_iou:
            raise SystemExit(
                f"{gate}: source-structure IoU {structure_iou:.4f} is below "
                f"{minimum_iou:.2f}"
            )
        print(f"gate={gate} structure_iou={structure_iou:.4f}")
    print(
        f"gate={gate} path={path} mode={image.mode} size={image.size} "
        f"center_mean={tuple(round(value, 2) for value in center_mean)}"
    )
PY
}

baseline_output="$(submit_workflow baseline "${workflows[baseline]}")"
validate_png "${baseline_output}" 512 512 baseline
install -m 0640 -- "${baseline_output}" \
  "${state_root}/input/smoke/qwen-image-2.1-red-cube.png"

transparency_output="$(submit_workflow transparency "${workflows[transparency]}")"
validate_png "${transparency_output}" 512 512 transparency
"${python_bin}" - "${state_root}/input/smoke/qwen-image-2.1-style-reference.png" <<'PY'
import sys
from PIL import Image

Image.new("RGB", (512, 512), (24, 180, 72)).save(sys.argv[1], format="PNG")
PY
chmod 0640 "${state_root}/input/smoke/qwen-image-2.1-style-reference.png"

multiref_output="$(submit_workflow multiref "${workflows[multiref]}")"
validate_png "${multiref_output}" 512 512 multiref "${baseline_output}"

edit_1024_output="$(submit_workflow edit_1024 "${workflows[edit_1024]}")"
validate_png "${edit_1024_output}" 1024 1024 edit_1024 "${baseline_output}"

t2i_2k_output="$(submit_workflow t2i_2k "${workflows[t2i_2k]}")"
validate_png "${t2i_2k_output}" 2048 2048 t2i_2k

for output in \
  "${baseline_output}" \
  "${transparency_output}" \
  "${multiref_output}" \
  "${edit_1024_output}" \
  "${t2i_2k_output}"; do
  sha256sum -- "${output}"
done

if [[ "$(systemctl --user show comfyui.service -p NRestarts --value)" != "${starting_restarts}" ]]; then
  printf 'ComfyUI restarted during image gates\n' >&2
  exit 1
fi
for service in qwen38.service jevk5.service comfyui.service; do
  if [[ "$(systemctl --user is-active "${service}")" != active ]]; then
    printf 'service is not active after image gates: %s\n' "${service}" >&2
    exit 1
  fi
done
printf 'all step-2 image gates passed; artifacts: %s\n' "${run_dir}"
