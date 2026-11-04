#!/usr/bin/env bash
# Step 3: train one MACE model on n rMD17 ethanol configurations.
#
# The settings follow Appendix A.5 of Batatia et al. (NeurIPS 2022) for rMD17: two interaction layers,
# correlation order 3, l_max = 3 for the spherical harmonics, 5 Å cutoff, 8 Bessel functions with a polynomial
# envelope (p = 5), radial MLP [64, 64, 64], 16x0e readout, loss weights 1 (energy) and 1000 (forces), AMSGrad
# with learning rate 0.01, batch size 5, plateau scheduler (patience 50, factor 0.8), EMA 0.99, weight decay
# 5e-7, float32, per-atom shift from the average energy of the training set (E0s = average).
#
# Changed for a laptop (set by the environment variables below): 64 channels and L = 1 messages instead of
# 256 channels, and a fixed budget of epochs. 1000 configurations are split 950 / 50 as in the paper; smaller
# sets keep 10 % for validation.
#
# Added: the second training stage of mace (--swa) for the last 40 % of the epochs, with loss weights 1000
# (energy) and 100 (forces). With the force weight of the first stage, the constant energy offset of the
# untrained model (a few eV) decays only slowly when an epoch has few optimizer steps (n = 50: 9 steps).
#
# Output: models/ethanol_N<n>_stagetwo.model (the final model) and models/ethanol_N<n>.model, logs in work/mace/.
# The Apple GPU (DEVICE=mps) does not work with mace 0.3.16: its evaluation step needs float64.
#
# Run:  bash scripts/step3_train_mace.sh 1000        (n = 50, 100, 250, 500 or 1000)
#       MAX_EPOCHS=100 NUM_CHANNELS=32 bash scripts/step3_train_mace.sh 1000
set -euo pipefail
cd "$(dirname "$0")/.."

N_TRAIN=${1:?give the number of training configurations, for example 1000}
SEED=${SEED:-123}
DEVICE=${DEVICE:-cpu}                 # cpu (or cuda on a cluster)
NUM_CHANNELS=${NUM_CHANNELS:-64}      # paper: 256
MAX_L=${MAX_L:-1}                     # equivariance of the messages; paper: up to 3
MAX_EPOCHS=${MAX_EPOCHS:-250}

if [ "$N_TRAIN" -ge 1000 ]; then
    VALID_FRACTION=0.05               # 950 training + 50 validation, as in the paper
else
    VALID_FRACTION=0.10
fi

# the second stage starts after 60 % of the epochs
START_STAGE_TWO=$(( MAX_EPOCHS * 6 / 10 ))

NAME="ethanol_N${N_TRAIN}"
mkdir -p models work/mace

mace_run_train \
    --name="$NAME" \
    --train_file="data/ethanol_train_N${N_TRAIN}.xyz" \
    --valid_fraction="$VALID_FRACTION" \
    --test_file="data/ethanol_test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels="$NUM_CHANNELS" \
    --max_L="$MAX_L" \
    --max_ell=3 \
    --correlation=3 \
    --r_max=5.0 \
    --num_radial_basis=8 \
    --num_cutoff_basis=5 \
    --radial_MLP="[64, 64, 64]" \
    --MLP_irreps="16x0e" \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=1000.0 \
    --optimizer="adam" \
    --amsgrad \
    --lr=0.01 \
    --weight_decay=5e-7 \
    --scheduler="ReduceLROnPlateau" \
    --scheduler_patience=50 \
    --lr_factor=0.8 \
    --ema \
    --ema_decay=0.99 \
    --swa \
    --start_swa="$START_STAGE_TWO" \
    --swa_energy_weight=1000.0 \
    --swa_forces_weight=100.0 \
    --swa_lr=0.001 \
    --batch_size=5 \
    --valid_batch_size=20 \
    --max_num_epochs="$MAX_EPOCHS" \
    --patience=1000 \
    --eval_interval=5 \
    --error_table="TotalMAE" \
    --default_dtype="float32" \
    --device="$DEVICE" \
    --seed="$SEED" \
    --model_dir="models" \
    --checkpoints_dir="work/mace/checkpoints" \
    --log_dir="work/mace/logs" \
    --results_dir="work/mace/results" \
    --save_cpu \
    --restart_latest
