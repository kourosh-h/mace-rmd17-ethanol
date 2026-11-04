#!/usr/bin/env bash
# Run the whole study from the download of rMD17 to the figures.
#
#   1  download rMD17 and prepare the ethanol training and test sets          (minutes, 1 GB download)
#   2  recompute 12 test configurations with ORCA (PBE/def2-SVP)              (~1 min with 6 cores)
#   3  train MACE on 50 and on 1000 configurations, as in Table 1 of the paper (about 4 h on a laptop CPU)
#   4  errors on the 1000 test configurations, compared with the paper
#   5  molecular dynamics with MACE: 50 ps NVT at 500 K and 10 ps NVE
#   6  recompute 20 MD frames with ORCA; torsion distributions; energy drift
#   7  figures
#
# ORCA must be called with its full path; set ORCA_EXE, for example
#   export ORCA_EXE=$HOME/orca_5_0_4/orca
# Each step skips the work that is already done, so the script can be run again after an interruption.
#
# Run:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

: "${ORCA_EXE:?set ORCA_EXE to the full path of the orca executable}"
PYTHON=${PYTHON:-python}

bash scripts/download_rmd17.sh
$PYTHON scripts/step1_prepare_rmd17.py
$PYTHON scripts/step2_check_labels_orca.py
bash scripts/step3_learning_curve.sh
$PYTHON scripts/step4_test_errors.py
$PYTHON scripts/step5_md_mace.py
$PYTHON scripts/step6_check_md.py
$PYTHON scripts/make_figures.py
