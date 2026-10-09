#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly host_root="$(cd -- "${source_root}/.." && pwd)"
readonly repo_root="$(cd -- "${host_root}/../.." && pwd)"
readonly release_file="${repo_root}/hosts/borg-release.env"
readonly target_root=/srv/ai
readonly target_bin="${target_root}/bin"
readonly borg_target="${target_bin}/borg"
readonly passphrase_file="${target_root}/secrets/redqueen-archive-passphrase"
readonly passcommand_target="${target_bin}/redqueen-archive-passcommand.sh"
readonly state_root="${target_root}/archive-state/borg"
readonly ssh_key="${HOME}/.ssh/redqueen-nas-archive"
readonly known_hosts="${HOME}/.ssh/redqueen-nas-archive.known_hosts"
readonly user_unit_dir="${HOME}/.config/systemd/user"
readonly env_target="${HOME}/.config/redqueen-archive.env"
readonly unit_source="${host_root}/systemd/redqueen-archive.service"
readonly timer_source="${host_root}/systemd/redqueen-archive.timer"
readonly unit_target="${user_unit_dir}/redqueen-archive.service"
readonly timer_target="${user_unit_dir}/redqueen-archive.timer"
if [[ "$(id -u)" -eq 0 ]]; then
  printf 'run this installer as ai, not root\n' >&2
  exit 1
fi
for command in awk cat chmod cmp curl date find gpg grep install mktemp mv openssl rm \
  sha256sum ssh ssh-keygen systemctl; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'required command not found: %s; have the node owner install its package, then retry\n' \
      "${command}" >&2
    exit 1
  fi
done
[[ -r "${release_file}" ]] || {
  printf 'missing pinned Borg release metadata: %s\n' "${release_file}" >&2
  exit 1
}
# The release metadata is a versioned shell fragment containing only constants.
# shellcheck disable=SC1090
source "${release_file}"

install -d -m 0755 -- "${target_bin}"
install -d -m 0700 -- "${target_root}/secrets" "${target_root}/archive-state" \
  "${state_root}" "${HOME}/.ssh" "${user_unit_dir}" "${HOME}/.config"

readonly temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/redqueen-archive-install.XXXXXXXX")"
cleanup() {
  rm -rf -- "${temporary_root}"
}
trap cleanup EXIT
readonly gpg_home="${temporary_root}/gnupg"
install -d -m 0700 -- "${gpg_home}"
readonly asset_url="https://github.com/borgbackup/borg/releases/download/${BORG_RELEASE}/${BORG_ASSET}"
readonly signature_url="${asset_url}.asc"
readonly asset_path="${temporary_root}/${BORG_ASSET}"
readonly signature_path="${asset_path}.asc"

if [[ -e "${borg_target}" ]]; then
  if [[ ! -x "${borg_target}" ]] || ! "${borg_target}" --version | grep -Fq "${BORG_RELEASE}"; then
    printf 'existing Borg binary differs from pinned %s; preserving it for review\n' \
      "${BORG_RELEASE}" >&2
    exit 1
  fi
else
  curl --fail --location --retry 3 --connect-timeout 20 --output "${asset_path}" "${asset_url}"
  curl --fail --location --retry 3 --connect-timeout 20 --output "${signature_path}" "${signature_url}"
  printf '%s  %s\n' "${BORG_ASSET_SHA256}" "${asset_path}" | sha256sum --check
  gpg --batch --homedir "${gpg_home}" --keyserver hkps://keyserver.ubuntu.com \
    --recv-keys "${BORG_SIGNING_FINGERPRINT}"
  readonly actual_fingerprint="$(gpg --batch --homedir "${gpg_home}" \
    --with-colons --fingerprint "${BORG_SIGNING_FINGERPRINT}" \
    | awk -F: '$1 == "fpr" { print toupper($10); exit }')"
  [[ "${actual_fingerprint}" == "${BORG_SIGNING_FINGERPRINT}" ]] || {
    printf 'Borg signing-key fingerprint did not match the pinned primary key\n' >&2
    exit 1
  }
  gpg --batch --homedir "${gpg_home}" --verify "${signature_path}" "${asset_path}"
  chmod 0755 "${asset_path}"
  install -m 0755 -- "${asset_path}" "${borg_target}"
fi

if [[ ! -e "${passphrase_file}" ]]; then
  readonly passphrase_temporary="$(mktemp "${target_root}/secrets/.redqueen-archive-passphrase.XXXXXXXX")"
  trap 'rm -rf -- "${temporary_root}"; rm -f -- "${passphrase_temporary}"' EXIT
  openssl rand -hex 32 >"${passphrase_temporary}"
  chmod 0600 "${passphrase_temporary}"
  mv -- "${passphrase_temporary}" "${passphrase_file}"
  trap cleanup EXIT
fi
[[ -f "${passphrase_file}" && ! -L "${passphrase_file}" && -s "${passphrase_file}" && \
  -r "${passphrase_file}" ]] || {
  printf 'archive passphrase must be a non-empty regular file readable by ai: %s\n' \
    "${passphrase_file}" >&2
  exit 1
}
chmod 0600 "${passphrase_file}"

if [[ -e "${ssh_key}" || -e "${ssh_key}.pub" ]]; then
  [[ -f "${ssh_key}" && ! -L "${ssh_key}" && -f "${ssh_key}.pub" && \
    ! -L "${ssh_key}.pub" && -s "${ssh_key}" && -s "${ssh_key}.pub" ]] || {
    printf 'incomplete archive SSH keypair exists; preserving it for review: %s\n' \
      "${ssh_key}" >&2
    exit 1
  }
else
  ssh-keygen -q -t ed25519 -N '' -C redqueen-ai-archive -f "${ssh_key}"
fi
chmod 0600 "${ssh_key}"
chmod 0644 "${ssh_key}.pub"

readonly expected_env="${temporary_root}/redqueen-archive.env"
# The archive's SSH identity and trust policy are pinned below, so skip unrelated
# distribution-wide SSH config fragments that can reject this service account.
cat >"${expected_env}" <<EOF
BORG_BASE_DIR=${state_root}
BORG_REPO=ssh://redqueen-archive@10.9.20.14/filesystem/k3s/backups/redqueen-ai/repo
BORG_REMOTE_PATH=/usr/local/bin/borg
BORG_RSH="ssh -F /dev/null -i ${ssh_key} -o IdentitiesOnly=yes -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o StrictHostKeyChecking=yes -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=${known_hosts} -o GlobalKnownHostsFile=/dev/null -o ServerAliveInterval=10 -o ServerAliveCountMax=30"
BORG_PASSCOMMAND=${passcommand_target}
BORG_LOCK_WAIT=600
EOF
if [[ -e "${env_target}" ]]; then
  if ! cmp -s -- "${expected_env}" "${env_target}"; then
    printf 'existing archive environment differs from the pinned setup; preserving it for review: %s\n' \
      "${env_target}" >&2
    exit 1
  fi
else
  install -m 0600 -- "${expected_env}" "${env_target}"
fi

install -m 0755 -- "${source_root}/passcommand.sh" "${passcommand_target}"
install -m 0755 -- "${source_root}/backup.sh" "${target_bin}/redqueen-archive-backup.sh"
install -m 0644 -- "${unit_source}" "${unit_target}"
install -m 0644 -- "${timer_source}" "${timer_target}"
systemctl --user daemon-reload

printf 'Borg %s installed and signature-verified; archive job installed but not enabled\n' \
  "${BORG_RELEASE}"
printf 'save the passphrase from %s in your password manager; its contents were not printed\n' \
  "${passphrase_file}"
printf 'verify the NAS SSH host key and save it to %s before using Borg\n' "${known_hosts}"
printf 'add this public key on the NAS with its restricted borg serve command:\n'
cat -- "${ssh_key}.pub"
