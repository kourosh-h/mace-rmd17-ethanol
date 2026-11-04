"""
Step 2: recompute a few rMD17 configurations with ORCA and compare with the published energies and forces.

This checks that we can reproduce the reference data before training on it, and that the ORCA settings of
orca_tools.RMD17_METHOD (PBE/def2-SVP) are the level of theory of rMD17. The same settings are used in step 6
to check the MACE molecular dynamics.

Reads data/ethanol_test.xyz; runs ORCA in orca/calculations/rmd17_label_check/ (kept in the repository); writes

    results/orca_label_check.csv          energies of rMD17 and ORCA for every configuration (eV)
    results/orca_label_check_forces.csv   all force components of rMD17 and ORCA (eV/Å)
    results/orca_label_check.json         summary: mean absolute differences

Energies are compared after removing the mean of each set: a constant offset (for example from a different
integration grid or ORCA version) does not change forces or relative energies, which are what MACE learns.

Run:  ORCA_EXE=/full/path/to/orca python scripts/step2_check_labels_orca.py [--n-configurations 12]
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_style import DATA_DIR, ORCA_DIR, RESULTS_DIR  # noqa: E402
from orca_tools import RMD17_METHOD, run_single_point  # noqa: E402

SEED = 2020   # random choice of the test configurations that are recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--n-configurations', type=int, default=12)
    parser.add_argument('--workers', type=int, default=6, help='ORCA calculations at the same time')
    args = parser.parse_args()

    test = read(DATA_DIR / 'ethanol_test.xyz', ':')
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(test), args.n_configurations, replace=False))
    print(f'recomputing {len(chosen)} test configurations with ORCA: {RMD17_METHOD}')

    # one serial ORCA calculation per configuration, several at the same time
    futures = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i in chosen:
            folder = ORCA_DIR / 'rmd17_label_check' / f"rmd17_{test[i].info['rmd17_index']:06d}"
            futures.append(pool.submit(run_single_point, test[i].get_chemical_symbols(), test[i].positions,
                                       folder))
        orca_results = []
        for future in futures:
            orca_results.append(future.result())

    # ------------------------------------------------------------------
    # Compare with the published labels
    # ------------------------------------------------------------------
    rmd17_energies = np.array([test[i].info['REF_energy'] for i in chosen])
    orca_energies = np.array([result[0] for result in orca_results])
    rmd17_forces = np.concatenate([test[i].arrays['REF_forces'] for i in chosen]).ravel()
    orca_forces = np.concatenate([result[1] for result in orca_results]).ravel()

    relative_rmd17 = rmd17_energies - rmd17_energies.mean()
    relative_orca = orca_energies - orca_energies.mean()
    summary = {
        'orca_method': RMD17_METHOD,
        'n_configurations': len(chosen),
        'mean_energy_offset_meV': float((orca_energies - rmd17_energies).mean() * 1000),
        'relative_energy_mae_meV': float(np.abs(relative_orca - relative_rmd17).mean() * 1000),
        'force_mae_meV_per_A': float(np.abs(orca_forces - rmd17_forces).mean() * 1000),
        'force_max_abs_difference_meV_per_A': float(np.abs(orca_forces - rmd17_forces).max() * 1000),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / 'orca_label_check.csv', 'w', newline='') as table:
        writer = csv.writer(table)
        writer.writerow(['rmd17_index', 'rmd17_energy_eV', 'orca_energy_eV'])
        for k, i in enumerate(chosen):
            writer.writerow([test[i].info['rmd17_index'], f'{rmd17_energies[k]:.8f}', f'{orca_energies[k]:.8f}'])
    with open(RESULTS_DIR / 'orca_label_check_forces.csv', 'w', newline='') as table:
        writer = csv.writer(table)
        writer.writerow(['rmd17_force_eV_per_A', 'orca_force_eV_per_A'])
        for rmd17_value, orca_value in zip(rmd17_forces, orca_forces):
            writer.writerow([f'{rmd17_value:.8f}', f'{orca_value:.8f}'])
    (RESULTS_DIR / 'orca_label_check.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
