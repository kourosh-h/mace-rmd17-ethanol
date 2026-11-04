"""
Step 4: errors of the trained MACE models on the 1000 test configurations of rMD17 split 1.

For every model models/ethanol_N<n>_stagetwo.model, the energy and the forces of every test configuration are predicted
and compared with the rMD17 labels. The mean absolute errors are those of Table 1 of Batatia et al. (2022):
energy in meV per molecule, force components in meV/Å.

Writes
    results/test_errors.csv                       one row per model: n, energy and force MAE and RMSE
    results/test_predictions_N<n>.csv             energy of every test configuration, rMD17 and MACE (eV)
    results/test_force_predictions_N<n>.csv       force components, rMD17 and MACE (eV/Å), for n = 1000 only

Run:  python scripts/step4_test_errors.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_style import DATA_DIR, MODEL_DIR, PAPER_ETHANOL, RESULTS_DIR  # noqa: E402

TRAINING_SET_SIZES = [50, 100, 250, 500, 1000]


def model_file(n_train):
    """The model file of one training-set size: the final model of the second training stage, if there is one."""
    for suffix in ['_stagetwo.model', '.model']:
        path = MODEL_DIR / f'ethanol_N{n_train}{suffix}'
        if path.exists():
            return path
    return None


def predict(calculator, configurations):
    """Energies (eV, one per configuration) and forces (eV/Å, configurations x atoms x 3) of the model."""
    energies = []
    forces = []
    for atoms in configurations:
        atoms = atoms.copy()
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    return np.array(energies), np.array(forces)


def main():
    test = read(DATA_DIR / 'ethanol_test.xyz', ':')
    reference_energies = np.array([atoms.info['REF_energy'] for atoms in test])
    reference_forces = np.array([atoms.arrays['REF_forces'] for atoms in test])
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for n_train in TRAINING_SET_SIZES:
        path = model_file(n_train)
        if path is None:
            print(f'no model for n = {n_train}, skipped')
            continue
        calculator = MACECalculator(model_paths=str(path), device='cpu', default_dtype='float32')
        energies, forces = predict(calculator, test)

        energy_error = (energies - reference_energies) * 1000       # meV per molecule
        force_error = (forces - reference_forces).ravel() * 1000    # meV/Å per component
        row = {'n_train': n_train,
               'energy_mae_meV': np.abs(energy_error).mean(),
               'energy_rmse_meV': np.sqrt(np.mean(energy_error ** 2)),
               'force_mae_meV_per_A': np.abs(force_error).mean(),
               'force_rmse_meV_per_A': np.sqrt(np.mean(force_error ** 2))}
        rows.append(row)
        paper = PAPER_ETHANOL.get(n_train)
        message = f"n = {n_train:4d}: energy MAE {row['energy_mae_meV']:.2f} meV, " \
                  f"force MAE {row['force_mae_meV_per_A']:.2f} meV/Å"
        if paper:
            message += f"  (paper: {paper['energy_mae_meV']} meV, {paper['force_mae_meV_per_A']} meV/Å)"
        print(message)

        with open(RESULTS_DIR / f'test_predictions_N{n_train}.csv', 'w', newline='') as table:
            writer = csv.writer(table)
            writer.writerow(['rmd17_index', 'rmd17_energy_eV', 'mace_energy_eV'])
            for atoms, reference, predicted in zip(test, reference_energies, energies):
                writer.writerow([atoms.info['rmd17_index'], f'{reference:.8f}', f'{predicted:.8f}'])
        if n_train == 1000:
            with open(RESULTS_DIR / 'test_force_predictions_N1000.csv', 'w', newline='') as table:
                writer = csv.writer(table)
                writer.writerow(['element', 'rmd17_force_eV_per_A', 'mace_force_eV_per_A'])
                for atoms, reference, predicted in zip(test, reference_forces, forces):
                    for symbol, reference_atom, predicted_atom in zip(atoms.get_chemical_symbols(), reference,
                                                                      predicted):
                        for component in range(3):
                            writer.writerow([symbol, f'{reference_atom[component]:.6f}',
                                             f'{predicted_atom[component]:.6f}'])

    with open(RESULTS_DIR / 'test_errors.csv', 'w', newline='') as table:
        writer = csv.writer(table)
        writer.writerow(['n_train', 'energy_mae_meV', 'energy_rmse_meV', 'force_mae_meV_per_A',
                         'force_rmse_meV_per_A'])
        for row in rows:
            writer.writerow([row['n_train'], f"{row['energy_mae_meV']:.4f}", f"{row['energy_rmse_meV']:.4f}",
                             f"{row['force_mae_meV_per_A']:.4f}", f"{row['force_rmse_meV_per_A']:.4f}"])
    print('wrote results/test_errors.csv')


if __name__ == '__main__':
    main()
