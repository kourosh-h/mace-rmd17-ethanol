#!/usr/bin/env bash
# Step 3b: train one MACE model for each training-set size, one after the other.
#
# By default the two sizes of Table 1 of Batatia et al. (2022) are trained: n = 50 and n = 1000. Smaller sets
# get more epochs, because one epoch of a small set is only a few optimizer steps. On a laptop CPU (4 cores)
# one epoch of 950 configurations takes about 45 s: n = 50 takes about 40 min, n = 1000 about 3.4 h.
# The full learning curve (about 8 h):  SIZES="50 100 250 500 1000" bash scripts/step3_learning_curve.sh
#
# Run:  bash scripts/step3_learning_curve.sh          (the variables of step3_train_mace.sh apply)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/mace

SIZES=${SIZES:-"50 1000"}

epochs_for_size() {
    # number of epochs for a training-set size
    case "$1" in
        50) echo 1000 ;;
        100) echo 800 ;;
        250) echo 500 ;;
        500) echo 350 ;;
        *) echo 250 ;;
    esac
}

for n_train in $SIZES; do
    if [ -f "models/ethanol_N${n_train}.model" ]; then
        echo "models/ethanol_N${n_train}.model exists, skipped"
        continue
    fi
    epochs=$(epochs_for_size "$n_train")
    echo "training on ${n_train} configurations for ${epochs} epochs"
    MAX_EPOCHS=$epochs bash scripts/step3_train_mace.sh "$n_train" > "work/mace/train_N${n_train}.log" 2>&1
done
