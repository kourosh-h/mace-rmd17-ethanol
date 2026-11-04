"""
Small driver for ORCA: write the input of one single-point calculation, run it, read energy and forces.

Used by step 2 (check of the rMD17 labels) and step 6 (check of the MACE molecular dynamics). Each calculation
runs serially in its own folder; several calculations run at the same time, one per CPU core, which for a
molecule of nine atoms is faster than one parallel (MPI) calculation.

The ORCA executable is taken from the environment variable ORCA_EXE, or else from the PATH. It must be called
with its full path, because ORCA starts its sub-programs from the folder of the executable.
Energies are returned in eV and forces in eV/Å.
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988     # CODATA 2018
BOHR_TO_ANGSTROM = 0.529177210903   # CODATA 2018

# Level of theory of rMD17 (Christensen and von Lilienfeld, 2020, section 4.3.1): ORCA 4.0.1, PBE/def2-SVP,
# RI approximation for the Coulomb integrals, VeryTightSCF, Grid7 NoFinalGrid. ORCA 5 no longer has the GridN
# keywords; DefGrid3 is its densest standard grid. RI-J with def2/J is the ORCA default for PBE.
RMD17_METHOD = 'PBE def2-SVP def2/J VeryTightSCF DefGrid3'


def orca_executable():
    """Full path of the ORCA executable (ORCA_EXE, or else the first 'orca' on the PATH)."""
    executable = os.environ.get('ORCA_EXE') or shutil.which('orca')
    if executable is None:
        raise RuntimeError('ORCA was not found: set ORCA_EXE=/full/path/to/orca')
    return str(Path(executable).resolve())


def write_input(path, symbols, positions, method, charge=0, multiplicity=1):
    """Write an ORCA input for one energy and gradient calculation (EnGrad); positions in Å."""
    lines = [f'! {method} EnGrad', f'* xyz {charge} {multiplicity}']
    for symbol, position in zip(symbols, positions):
        lines.append(f'{symbol:2s} {position[0]:16.10f} {position[1]:16.10f} {position[2]:16.10f}')
    lines.append('*')
    Path(path).write_text('\n'.join(lines) + '\n')


def read_engrad(path, n_atoms):
    """
    Energy (eV) and forces (eV/Å) from the .engrad file of ORCA. The file holds, after comment lines, the
    number of atoms, the energy (Hartree) and the 3N gradient components (Hartree/Bohr).
    """
    values = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            values.append(line.strip())
    energy = float(values[1]) * HARTREE_TO_EV
    gradient = np.array([float(value) for value in values[2:2 + 3 * n_atoms]]).reshape(n_atoms, 3)
    # the force is minus the gradient of the energy
    forces = -gradient * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    return energy, forces


def run_single_point(symbols, positions, folder, method=RMD17_METHOD):
    """
    Run one ORCA energy and gradient calculation in `folder` and return (energy in eV, forces in eV/Å).
    A folder that already holds a finished calculation is read and not run again.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    engrad_file = folder / 'orca.engrad'
    output_file = folder / 'orca.out'
    finished = output_file.exists() and 'ORCA TERMINATED NORMALLY' in output_file.read_text()
    if not finished:
        write_input(folder / 'orca.inp', symbols, positions, method)
        with open(output_file, 'w') as output:
            subprocess.run([orca_executable(), 'orca.inp'], cwd=folder, stdout=output,
                           stderr=subprocess.STDOUT, check=False)
        if 'ORCA TERMINATED NORMALLY' not in output_file.read_text():
            raise RuntimeError(f'ORCA did not finish in {folder}')
        # the wave-function and scratch files are large and not needed afterwards
        for scratch_file in folder.iterdir():
            if scratch_file.suffix in ['.gbw', '.tmp', '.densities']:
                scratch_file.unlink()
    return read_engrad(engrad_file, len(symbols))
