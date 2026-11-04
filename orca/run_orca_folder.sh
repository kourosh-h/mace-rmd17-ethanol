#!/usr/bin/env bash
# Run every ORCA input (*.inp) below a folder, several at the same time.
#
# For a molecule of nine atoms, one serial ORCA calculation per CPU core is faster than one parallel (MPI)
# calculation, and it needs no MPI installation. Inputs that already have a finished output are skipped.
#
# Run:  bash orca/run_orca_folder.sh orca/calculations/rmd17_label_check 6     (6 calculations at a time)
set -euo pipefail

FOLDER=${1:?give the folder with the ORCA inputs}
JOBS=${2:-4}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

to_run=()
while IFS= read -r input; do
    output="${input%.inp}.out"
    if [ -f "$output" ] && grep -q "ORCA TERMINATED NORMALLY" "$output"; then
        continue
    fi
    to_run+=("$input")
done < <(find "$FOLDER" -name '*.inp' | sort)

echo "${#to_run[@]} calculations to run, $JOBS at a time"
if [ "${#to_run[@]}" -gt 0 ]; then
    printf '%s\n' "${to_run[@]}" | xargs -P "$JOBS" -I {} bash "$SCRIPT_DIR/run_orca.sh" {}
fi
