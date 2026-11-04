"""
Step 1: prepare the ethanol data of the revised MD17 dataset (rMD17) for MACE.

Reads work/download/rmd17/npz_data/rmd17_ethanol.npz and the index files of split 1 (index_train_01.csv,
index_test_01.csv), converts the energies and forces from kcal/mol to eV, and writes extended-XYZ files with the
keys REF_energy (eV) and REF_forces (eV/Å):

    data/ethanol_train_N1000.xyz          the 1000 training configurations of split 1
    data/ethanol_train_N<n>.xyz           the first n of them, n = 50, 100, 250, 500 (learning curve)
    data/ethanol_test.xyz                 the 1000 test configurations of split 1

It also writes the two torsion angles of every one of the 100 000 configurations (the 500 K ab initio molecular
dynamics of the original MD17), to compare with the MACE molecular dynamics in step 6:

    data/rmd17_ethanol_torsions.npz       hydroxyl torsion H-O-C-C and methyl torsion H-C-C-O (degrees)

Run:  python scripts/step1_prepare_rmd17.py   (after download_rmd17.sh)
"""
import csv
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_style import DATA_DIR, WORK_DIR  # noqa: E402

RMD17_DIR = WORK_DIR / 'download' / 'rmd17'
KCAL_PER_MOL_TO_EV = 4.184 / 96.48533212   # 1 kcal/mol in eV (kJ/mol per eV from CODATA 2018)
SPLIT = 1                                  # rMD17 provides five splits; the paper table uses one per model
LEARNING_CURVE_SIZES = [50, 100, 250, 500, 1000]


def read_indices(path):
    """Indices of the configurations listed in one split file of rMD17 (one integer per line)."""
    indices = []
    with open(path) as split_file:
        for row in csv.reader(split_file):
            if row:
                indices.append(int(float(row[0])))
    return indices


def make_configurations(dataset, indices):
    """ASE Atoms of the given configurations, with the rMD17 energy and forces converted to eV and eV/Å."""
    numbers = dataset['nuclear_charges']
    configurations = []
    for index in indices:
        atoms = Atoms(numbers=numbers, positions=dataset['coords'][index])
        atoms.info['REF_energy'] = float(dataset['energies'][index]) * KCAL_PER_MOL_TO_EV
        atoms.arrays['REF_forces'] = dataset['forces'][index] * KCAL_PER_MOL_TO_EV
        atoms.info['rmd17_index'] = int(index)
        configurations.append(atoms)
    return configurations


def torsion_atoms(numbers, positions):
    """
    Indices of the atoms that define the two torsions of ethanol, found from the distances of one configuration:
    O, the carbon bonded to O (C1), the methyl carbon (C2), the hydroxyl hydrogen and the three methyl hydrogens.
    """
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    oxygen = int(np.where(numbers == 8)[0][0])
    carbons = np.where(numbers == 6)[0]
    hydrogens = np.where(numbers == 1)[0]
    # C1 is the carbon nearest to O, C2 the other one
    carbon_1 = int(carbons[np.argmin(distances[oxygen, carbons])])
    carbon_2 = int(carbons[np.argmax(distances[oxygen, carbons])])
    # the hydroxyl hydrogen is the hydrogen nearest to O; the methyl hydrogens are the three nearest to C2
    hydroxyl_hydrogen = int(hydrogens[np.argmin(distances[oxygen, hydrogens])])
    methyl_hydrogens = [int(h) for h in hydrogens[np.argsort(distances[carbon_2, hydrogens])[:3]]]
    return oxygen, carbon_1, carbon_2, hydroxyl_hydrogen, methyl_hydrogens


def dihedral_angles(positions, a, b, c, d):
    """Dihedral angle a-b-c-d (degrees, -180 ... 180) for every configuration of positions (frames, atoms, 3)."""
    b0 = positions[:, a] - positions[:, b]
    b1 = positions[:, c] - positions[:, b]
    b2 = positions[:, d] - positions[:, c]
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    # components of b0 and b2 perpendicular to the central bond
    v = b0 - np.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=1)[:, None] * b1
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def main():
    dataset = np.load(RMD17_DIR / 'npz_data' / 'rmd17_ethanol.npz')
    print(f"rMD17 ethanol: {len(dataset['energies'])} configurations, nuclear charges {dataset['nuclear_charges']}")

    train_indices = read_indices(RMD17_DIR / 'splits' / f'index_train_{SPLIT:02d}.csv')
    test_indices = read_indices(RMD17_DIR / 'splits' / f'index_test_{SPLIT:02d}.csv')
    print(f'split {SPLIT}: {len(train_indices)} training and {len(test_indices)} test configurations')

    DATA_DIR.mkdir(exist_ok=True)
    train = make_configurations(dataset, train_indices)
    for size in LEARNING_CURVE_SIZES:
        # the smaller training sets are the first entries of the (already random) split order
        write(DATA_DIR / f'ethanol_train_N{size}.xyz', train[:size])
    write(DATA_DIR / 'ethanol_test.xyz', make_configurations(dataset, test_indices))

    # ------------------------------------------------------------------
    # Torsions of the whole 500 K trajectory, the reference of step 6
    # ------------------------------------------------------------------
    numbers = dataset['nuclear_charges']
    coords = dataset['coords']
    oxygen, carbon_1, carbon_2, hydroxyl_hydrogen, methyl_hydrogens = torsion_atoms(numbers, coords[0])
    hydroxyl = dihedral_angles(coords, hydroxyl_hydrogen, oxygen, carbon_1, carbon_2)
    # the three methyl hydrogens are equivalent: the methyl torsion is taken modulo 120 degrees
    methyl = dihedral_angles(coords, methyl_hydrogens[0], carbon_2, carbon_1, oxygen) % 120.0
    np.savez_compressed(DATA_DIR / 'rmd17_ethanol_torsions.npz', hydroxyl_deg=hydroxyl.astype(np.float32),
                        methyl_deg=methyl.astype(np.float32))
    trans_fraction = np.mean(np.abs(hydroxyl) > 120.0)
    print(f'hydroxyl torsion: trans fraction {trans_fraction:.3f} (|H-O-C-C| > 120 degrees)')
    print('wrote data/ethanol_train_N*.xyz, data/ethanol_test.xyz, data/rmd17_ethanol_torsions.npz')


if __name__ == '__main__':
    main()
