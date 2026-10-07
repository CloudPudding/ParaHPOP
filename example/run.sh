#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-cpu}"
profile="${2:-full}"
case "$backend" in cpu|gpu) ;; *) echo 'Backend must be cpu or gpu.' >&2; exit 1 ;; esac
case "$profile" in full|central) ;; *) echo 'Profile must be full or central.' >&2; exit 1 ;; esac
build_dir="${BUILD_DIR:-$root/build}"
output_dir="${3:-$root/output/envisat_${profile}_${backend}_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE
"$build_dir/bin/parahpop_$backend" "$root/example/config/envisat_$profile.json" \
    "$root/example/input/envisat.json" "$output_dir"
printf 'Output: %s\n' "$output_dir"
