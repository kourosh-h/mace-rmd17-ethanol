#!/usr/bin/env bash
# Run one ORCA calculation: the output is written next to the input, with the same name and .out.
#
# ORCA must be started with its full path, because it starts its sub-programs (orca_scf, orca_grad, ...) from
# the folder of the executable. Set ORCA_EXE, or put the ORCA folder on the PATH.
#
# Run:  bash orca/run_orca.sh orca/example_ethanol_single_point.inp
set -euo pipefail

INPUT=${1:?give the ORCA input file, for example orca/example_ethanol_single_point.inp}
ORCA_EXE=${ORCA_EXE:-$(command -v orca || true)}
if [ -z "$ORCA_EXE" ]; then
    echo "ORCA was not found: set ORCA_EXE=/full/path/to/orca"
    exit 1
fi

folder=$(cd "$(dirname "$INPUT")" && pwd)
name=$(basename "$INPUT" .inp)
cd "$folder"

start=$(date +%s)
"$ORCA_EXE" "$name.inp" > "$name.out" 2>&1 || true
seconds=$(( $(date +%s) - start ))

if grep -q "ORCA TERMINATED NORMALLY" "$name.out"; then
    energy=$(grep "FINAL SINGLE POINT ENERGY" "$name.out" | tail -1 | awk '{print $5}')
    echo "$name: E = $energy Hartree ($seconds s)"
    # the wave function and the scratch files are large and are not needed afterwards
    rm -f "$name.gbw" "$name.densities" "$name".*.tmp
else
    echo "$name: ORCA did not finish, see $folder/$name.out"
    exit 1
fi
