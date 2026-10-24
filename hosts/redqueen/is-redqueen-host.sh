#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname -s 2>/dev/null || true)" == amd-halo ]] || exit 1
readonly redqueen_address="$(getent ahostsv4 redqueen.mousses.xyz | awk 'NR == 1 { print $1 }')"
[[ -n "$redqueen_address" ]] || exit 1
readonly local_addresses=" $(hostname -I 2>/dev/null || true) "
[[ "$local_addresses" == *" $redqueen_address "* ]]
