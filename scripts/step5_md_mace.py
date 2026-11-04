"""
Step 5: molecular dynamics of one ethanol molecule with the MACE model trained on 1000 configurations.

Two runs, both with a time step of 0.5 fs:
    NVT  Langevin thermostat at 500 K, the temperature of the MD17 trajectory, 50 ps (the first 10 ps are
         equilibration); used for the torsion distributions of step 6
    NVE  10 ps without thermostat, started from the end of the NVT run; the total energy must stay constant
         if the forces are the derivative of a smooth energy

Reads models/ethanol_N1000_stagetwo.model and the first test configuration as the starting structure. Writes
    work/md/nvt_500K.traj, work/md/nve.traj      trajectories (every 10 steps = 5 fs)
    results/md_nvt_500K.csv, results/md_nve.csv  time (ps), potential, kinetic and total energy (eV), T (K)

Run:  python scripts/step5_md_mace.py [--nvt-ps 50] [--nve-ps 10]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_style import DATA_DIR, MODEL_DIR, RESULTS_DIR, WORK_DIR  # noqa: E402

TEMPERATURE_K = 500.0      # temperature of the MD17 ab initio trajectory
TIME_STEP_FS = 0.5         # time step of MD17
FRICTION_PER_FS = 0.01     # Langevin friction (1/fs): a weak coupling, 100 fs relaxation time
SAVE_INTERVAL = 10         # save every 10 steps = 5 fs
SEED = 7


def run_and_record(dynamics, atoms, n_steps, trajectory_path, table_path, start_time_ps):
    """
    Run n_steps of the dynamics, save every SAVE_INTERVAL steps to the trajectory, and write the table of
    energies and temperature. Returns the time (ps) at the end of the run.
    """
    trajectory = Trajectory(trajectory_path, 'w', atoms)
    rows = []

    def record():
        """Save the current frame and its energies (called by ASE every SAVE_INTERVAL steps)."""
        potential = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        time_ps = start_time_ps + dynamics.get_number_of_steps() * TIME_STEP_FS / 1000.0
        rows.append([time_ps, potential, kinetic, potential + kinetic, atoms.get_temperature()])
        trajectory.write(atoms)

    dynamics.attach(record, interval=SAVE_INTERVAL)
    wall_start = time.time()
    dynamics.run(n_steps)
    trajectory.close()
    steps_per_second = n_steps / (time.time() - wall_start)

    with open(table_path, 'w', newline='') as table:
        writer = csv.writer(table)
        writer.writerow(['time_ps', 'potential_energy_eV', 'kinetic_energy_eV', 'total_energy_eV',
                         'temperature_K'])
        for row in rows:
            writer.writerow([f'{row[0]:.4f}', f'{row[1]:.6f}', f'{row[2]:.6f}', f'{row[3]:.6f}', f'{row[4]:.2f}'])
    print(f'{n_steps} steps, {steps_per_second:.0f} steps/s')
    return rows[-1][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--nvt-ps', type=float, default=50.0)
    parser.add_argument('--nve-ps', type=float, default=10.0)
    parser.add_argument('--model', default=str(MODEL_DIR / 'ethanol_N1000_stagetwo.model'))
    args = parser.parse_args()

    calculator = MACECalculator(model_paths=args.model, device='cpu', default_dtype='float32')
    atoms = read(DATA_DIR / 'ethanol_test.xyz', 0)
    atoms.calc = calculator
    # The center of mass is held fixed with a constraint. The older option Langevin(fixcm=True) is exact only for
    # atoms of equal mass; for this 9-atom molecule with H, C and O it gave about 650 K instead of 500 K (the ASE
    # documentation warns that it is not correct for small systems).
    atoms.set_constraint(FixCom())

    rng = np.random.default_rng(SEED)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)       # no motion of the center of mass
    ZeroRotation(atoms)     # no overall rotation

    md_dir = WORK_DIR / 'md'
    md_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f'NVT at {TEMPERATURE_K:.0f} K for {args.nvt_ps} ps')
    nvt = Langevin(atoms, TIME_STEP_FS * units.fs, temperature_K=TEMPERATURE_K,
                   friction=FRICTION_PER_FS / units.fs, fixcm=False, rng=rng)
    nvt_steps = int(round(args.nvt_ps * 1000 / TIME_STEP_FS))
    end_time = run_and_record(nvt, atoms, nvt_steps, md_dir / 'nvt_500K.traj', RESULTS_DIR / 'md_nvt_500K.csv', 0.0)

    print(f'NVE for {args.nve_ps} ps')
    nve = VelocityVerlet(atoms, TIME_STEP_FS * units.fs)
    nve_steps = int(round(args.nve_ps * 1000 / TIME_STEP_FS))
    run_and_record(nve, atoms, nve_steps, md_dir / 'nve.traj', RESULTS_DIR / 'md_nve.csv', end_time)


if __name__ == '__main__':
    main()
