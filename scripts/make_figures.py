"""
Figures of the study, drawn from the tables in data/ and results/ (no model or ORCA calculation is run here).

    Figure 1  the data: ethanol and its two torsions, energies of the training and test sets, torsion map of the
              500 K trajectory with the training configurations
    Figure 2  the reference: rMD17 energies and forces recomputed with ORCA
    Figure 3  accuracy of MACE: learning curves compared with Batatia et al. (2022), force parity on the test set
    Figure 4  molecular dynamics with MACE: temperature, energy conservation, torsion distributions compared with
              MD17, and forces of MACE frames recomputed with ORCA
    banner    the picture at the top of the README (docs/images/banner_light.png and banner_dark.png)
    movie     animation of 2 ps of the MACE run, with the hydroxyl torsion (docs/images/md_ethanol.gif); needs the
              trajectory of step 5 in work/md/

Each figure is written to figures/ as PDF and PNG. A figure whose input tables are missing is skipped.

Run:  python scripts/make_figures.py [1 2 3 4 banner movie]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_style import (ACCENT_COLORS, DATA_DIR, ELEMENT_COLORS, FIG_WIDTH, MACE_COLOR, PAPER_COLOR,  # noqa: E402
                        PAPER_ETHANOL, REFERENCE_COLOR, RESULTS_DIR, SHADE_ALPHA, WORK_DIR, new_figure,
                        panel_letter, save)
from step1_prepare_rmd17 import dihedral_angles, torsion_atoms  # noqa: E402

ATOM_RADII = {'H': 0.22, 'C': 0.36, 'O': 0.34}   # drawn radii (Å), not physical radii
BOND_LENGTH_HEAVY = 1.75                        # C-C and C-O closer than this (Å) are drawn as bonded
BOND_LENGTH_HYDROGEN = 1.3                      # X-H closer than this (Å) are drawn as bonded
VIEW_TILT_DEG = 35.0                            # tilt of the molecule drawing out of the C-C-O plane
DOCS_IMAGES = Path(__file__).resolve().parent.parent / 'docs' / 'images'   # images of the README
MOVIE_PS = 2.0                                  # length of the animated piece of the MD run (ps)
MOVIE_STEP = 2                                  # draw every second saved frame (10 fs of MD per picture)
MOVIE_FRAME_MS = 50                             # display time of one picture (ms), 20 pictures per second
SAVED_FRAME_FS = 5.0                            # time between saved frames of step 5 (fs)


def read_table(path):
    """Columns of a CSV table as a dictionary of arrays (numbers where possible, else strings)."""
    with open(path) as table:
        rows = list(csv.DictReader(table))
    columns = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        try:
            columns[name] = np.array([float(value) for value in values])
        except ValueError:
            columns[name] = np.array(values)
    return columns


def parity_axes(ax, x, y, color, label_x, label_y, size=3.0, alpha=0.35):
    """Scatter y against x with the diagonal y = x and equal axis ranges."""
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    margin = 0.05 * (high - low)
    ax.plot([low - margin, high + margin], [low - margin, high + margin], color='0.6', lw=0.9, ls='--', zorder=1)
    ax.scatter(x, y, s=size, color=color, alpha=alpha, edgecolors='none', zorder=2, rasterized=True)
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)


def draw_molecule(ax, atoms, colors=ELEMENT_COLORS, labels=True, highlight=None):
    """
    Ethanol seen along the axis of smallest spread, with bonds as gray lines and atoms as filled circles colored
    by colors (element -> color). With labels, the atoms of the two torsions are labeled. highlight is an
    optional (atom index, color) pair for one atom that should stand out. The view is a projection; depth sets
    the drawing order.
    """
    positions = atoms.positions - atoms.positions.mean(axis=0)
    oxygen, carbon_1, carbon_2, hydroxyl_hydrogen, methyl_hydrogens = torsion_atoms(atoms.numbers, atoms.positions)
    # view on the plane of the heavy atoms C2-C1-O, tilted by VIEW_TILT_DEG about the C2-C1 axis, so that the two
    # hydrogens of C1 (one above and one below that plane) do not fall on top of each other
    along_bond = positions[carbon_1] - positions[carbon_2]
    along_bond = along_bond / np.linalg.norm(along_bond)
    normal = np.cross(along_bond, positions[oxygen] - positions[carbon_1])
    normal = normal / np.linalg.norm(normal)
    in_plane = np.cross(normal, along_bond)
    tilt = np.radians(VIEW_TILT_DEG)
    screen_y = np.cos(tilt) * in_plane + np.sin(tilt) * normal
    screen_depth = -np.sin(tilt) * in_plane + np.cos(tilt) * normal
    projected = np.stack([positions @ along_bond, positions @ screen_y, positions @ screen_depth], axis=1)
    symbols = atoms.get_chemical_symbols()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = np.linalg.norm(positions[i] - positions[j])
            # bonds to hydrogen are about 1.1 Å, C-C and C-O bonds 1.4-1.5 Å
            if 'H' in (symbols[i], symbols[j]):
                bonded = distance < BOND_LENGTH_HYDROGEN
            else:
                bonded = distance < BOND_LENGTH_HEAVY
            if bonded:
                ax.plot(projected[[i, j], 0], projected[[i, j], 1], color='0.55', lw=2.2, zorder=1,
                        solid_capstyle='round')
    order = np.argsort(projected[:, 2])
    for i in order:
        color = colors[symbols[i]]
        if highlight is not None and i == highlight[0]:
            color = highlight[1]
        circle = plt.Circle(projected[i, :2], ATOM_RADII[symbols[i]], facecolor=color,
                            edgecolor='black', lw=0.6, zorder=2 + projected[i, 2])
        ax.add_patch(circle)
    # label text and its offset from the atom (points), placed away from the bonds
    atom_labels = {oxygen: ('O', (8, 6)), carbon_1: ('C1', (-6, -16)), carbon_2: ('C2', (-6, 10)),
                   hydroxyl_hydrogen: ('H$_O$', (7, 6))}
    if not labels:
        atom_labels = {}
    for index, (text, offset) in atom_labels.items():
        ax.annotate(text, projected[index, :2], xytext=offset, textcoords='offset points', fontsize=7,
                    zorder=20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(projected[:, 0].min() - 0.7, projected[:, 0].max() + 0.7)
    ax.set_ylim(projected[:, 1].min() - 0.7, projected[:, 1].max() + 0.7)


# ----------------------------------------------------------------------
# Figure 1: the data
# ----------------------------------------------------------------------

def figure_1():
    """Ethanol, energies of the training and test sets, and the torsion map of the MD17 trajectory."""
    train = read(DATA_DIR / 'ethanol_train_N1000.xyz', ':')
    test = read(DATA_DIR / 'ethanol_test.xyz', ':')
    torsions = np.load(DATA_DIR / 'rmd17_ethanol_torsions.npz')

    fig = plt.figure(figsize=(FIG_WIDTH, 2.45))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.0, 1.15], left=0.03, right=0.92, bottom=0.18,
                    top=0.9, wspace=0.42)

    ax = fig.add_subplot(grid[0])
    draw_molecule(ax, train[0])
    handles = []
    for symbol in ['C', 'O', 'H']:
        handles.append(Line2D([], [], marker='o', ls='none', markerfacecolor=ELEMENT_COLORS[symbol],
                              markeredgecolor='black', markersize=6, label=symbol))
    ax.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12), handletextpad=0.1)
    panel_letter(ax, 'a', x=0.0)

    # energies relative to the lowest energy of the two sets (meV)
    ax = fig.add_subplot(grid[1])
    train_energies = np.array([atoms.info['REF_energy'] for atoms in train])
    test_energies = np.array([atoms.info['REF_energy'] for atoms in test])
    lowest = min(train_energies.min(), test_energies.min())
    bins = np.linspace(0, max(train_energies.max(), test_energies.max()) - lowest, 40) * 1000
    ax.hist((train_energies - lowest) * 1000, bins=bins, color=MACE_COLOR, alpha=0.55, label='training (1000)')
    ax.hist((test_energies - lowest) * 1000, bins=bins, histtype='step', color=REFERENCE_COLOR, lw=1.3,
            label='test (1000)')
    ax.set_xlabel('energy above minimum (meV)')
    ax.set_ylabel('configurations')
    # room above the histograms for the legend
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc='upper right', borderaxespad=0.0)
    panel_letter(ax, 'b')

    # torsion map of the 100 000 configurations, with the training configurations on top
    ax = fig.add_subplot(grid[2])
    counts, x_edges, y_edges = np.histogram2d(torsions['hydroxyl_deg'], torsions['methyl_deg'],
                                              bins=[72, 24], range=[[-180, 180], [0, 120]])
    # density relative to the most visited cell (1 = most visited)
    image = ax.pcolormesh(x_edges, y_edges, counts.T / counts.max(), cmap='Blues', shading='flat',
                          rasterized=True, vmin=0, vmax=1)
    oxygen, carbon_1, carbon_2, hydroxyl_hydrogen, methyl_hydrogens = torsion_atoms(train[0].numbers,
                                                                                    train[0].positions)
    train_hydroxyl = []
    train_methyl = []
    for atoms in train:
        train_hydroxyl.append(atoms.get_dihedral(hydroxyl_hydrogen, oxygen, carbon_1, carbon_2, mic=False))
        train_methyl.append(atoms.get_dihedral(methyl_hydrogens[0], carbon_2, carbon_1, oxygen) % 120.0)
    train_hydroxyl = (np.array(train_hydroxyl) + 180.0) % 360.0 - 180.0
    ax.scatter(train_hydroxyl, train_methyl, s=1.2, color=ACCENT_COLORS[1], alpha=0.7, edgecolors='none',
               label='training', rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 120)
    ax.set_xticks([-180, -60, 60, 180])
    ax.set_yticks([0, 60, 120])
    ax.set_xlabel('hydroxyl torsion H$_O$–O–C1–C2 (°)')
    ax.set_ylabel('methyl torsion H–C2–C1–O (°)')
    colorbar = fig.colorbar(image, ax=ax, pad=0.03, fraction=0.06)
    colorbar.set_label('relative density, MD17 at 500 K', fontsize=7)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=7)
    ax.legend(loc='upper left', markerscale=4, handletextpad=0.1, borderaxespad=0.1, labelcolor=ACCENT_COLORS[1])
    panel_letter(ax, 'c')
    save(fig, 'Figure_1_data')


# ----------------------------------------------------------------------
# Figure 2: the ORCA check of the reference data
# ----------------------------------------------------------------------

def figure_2():
    """rMD17 energies and forces against the same configurations recomputed with ORCA."""
    energies = read_table(RESULTS_DIR / 'orca_label_check.csv')
    forces = read_table(RESULTS_DIR / 'orca_label_check_forces.csv')
    summary = json.loads((RESULTS_DIR / 'orca_label_check.json').read_text())

    fig, axes = new_figure(2.4, ncols=3, gridspec_kw={'wspace': 0.55, 'left': 0.08, 'right': 0.98,
                                                      'bottom': 0.19, 'top': 0.9})
    relative_rmd17 = (energies['rmd17_energy_eV'] - energies['rmd17_energy_eV'].mean()) * 1000
    relative_orca = (energies['orca_energy_eV'] - energies['orca_energy_eV'].mean()) * 1000
    parity_axes(axes[0], relative_rmd17, relative_orca, ACCENT_COLORS[1], 'rMD17 energy (meV)',
                'ORCA energy (meV)', size=18, alpha=0.9)
    axes[0].text(0.05, 0.95, f"MAE {summary['relative_energy_mae_meV']:.3f} meV\n"
                             f"{summary['n_configurations']} configurations\n"
                             f"offset {summary['mean_energy_offset_meV']:+.2f} meV removed",
                 transform=axes[0].transAxes, va='top', fontsize=7)
    panel_letter(axes[0], 'a')

    parity_axes(axes[1], forces['rmd17_force_eV_per_A'], forces['orca_force_eV_per_A'], ACCENT_COLORS[1],
                'rMD17 force (eV/Å)', 'ORCA force (eV/Å)', size=6, alpha=0.7)
    axes[1].text(0.05, 0.95, f"MAE {summary['force_mae_meV_per_A']:.3f} meV/Å",
                 transform=axes[1].transAxes, va='top', fontsize=7)
    panel_letter(axes[1], 'b')

    # the differences on the scale of the MACE errors: Table 1 of the paper gives 2.1 meV/Å for n = 1000
    difference = (forces['orca_force_eV_per_A'] - forces['rmd17_force_eV_per_A']) * 1000
    axes[2].hist(difference, bins=np.linspace(-1.2, 1.2, 41), color=ACCENT_COLORS[1], alpha=0.8)
    axes[2].axvspan(-PAPER_ETHANOL[1000]['force_mae_meV_per_A'], PAPER_ETHANOL[1000]['force_mae_meV_per_A'],
                    color=PAPER_COLOR, alpha=0.08, zorder=0)
    axes[2].set_xlim(-2.4, 2.4)
    axes[2].set_xlabel('ORCA − rMD17 force (meV/Å)')
    axes[2].set_ylabel('force components')
    axes[2].text(0.97, 0.95, 'shaded: ± 2.1 meV/Å,\nMACE force MAE\nof the paper', transform=axes[2].transAxes,
                 ha='right', va='top', fontsize=7, color=PAPER_COLOR)
    panel_letter(axes[2], 'c')
    save(fig, 'Figure_2_orca_check')


# ----------------------------------------------------------------------
# Figure 3: accuracy of MACE
# ----------------------------------------------------------------------

def figure_3():
    """Learning curves of energy and force errors, with the paper values, and the force errors by element."""
    errors = read_table(RESULTS_DIR / 'test_errors.csv')
    force_table = read_table(RESULTS_DIR / 'test_force_predictions_N1000.csv')

    fig, axes = new_figure(2.5, ncols=3, gridspec_kw={'wspace': 0.5, 'left': 0.08, 'right': 0.98,
                                                      'bottom': 0.19, 'top': 0.9})
    paper_sizes = sorted(PAPER_ETHANOL)
    quantities = [('energy_mae_meV', 'energy MAE (meV)', [0.3, 1, 3, 10, 30], (0.25, 40)),
                  ('force_mae_meV_per_A', 'force MAE (meV/Å)', [1, 3, 10, 30, 100], (1.0, 120))]
    for ax, (column, label, ticks, limits), letter in zip(axes[:2], quantities, ['a', 'b']):
        ax.plot(errors['n_train'], errors[column], marker='o', color=MACE_COLOR, label='this work (laptop)')
        paper_values = [PAPER_ETHANOL[size][column] for size in paper_sizes]
        ax.plot(paper_sizes, paper_values, marker='s', ls='none', markersize=6, markerfacecolor='white',
                markeredgecolor=PAPER_COLOR, markeredgewidth=1.4, label='Batatia et al. (2022)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('training configurations')
        ax.set_ylabel(label)
        ax.set_xticks([50, 100, 250, 500, 1000])
        ax.set_xticklabels(['50', '100', '250', '500', '1000'])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:g}' for tick in ticks])
        ax.set_ylim(*limits)
        ax.minorticks_off()
        panel_letter(ax, letter, x=-0.24)
    axes[0].legend(loc='lower left')

    # (c) distribution of the error of each force component, n = 1000
    bins = np.linspace(0, 40, 41)
    for symbol in ['H', 'C', 'O']:
        is_element = force_table['element'] == symbol
        difference = 1000 * np.abs(force_table['mace_force_eV_per_A'][is_element]
                                   - force_table['rmd17_force_eV_per_A'][is_element])
        color = ELEMENT_COLORS[symbol] if symbol != 'H' else ACCENT_COLORS[0]
        axes[2].hist(difference, bins=bins, density=True, histtype='step', lw=1.3, color=color,
                     label=f'{symbol}  {difference.mean():.1f}')
    axes[2].axvline(PAPER_ETHANOL[1000]['force_mae_meV_per_A'], color=PAPER_COLOR, lw=0.9, ls='--')
    axes[2].text(PAPER_ETHANOL[1000]['force_mae_meV_per_A'] + 0.8, 0.97, 'paper\nMAE', color=PAPER_COLOR,
                 transform=axes[2].get_xaxis_transform(), ha='left', va='top', fontsize=7)
    axes[2].set_xlim(0, 40)
    axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.1)
    axes[2].set_xlabel('|ΔF| of a component (meV/Å)')
    axes[2].set_ylabel('probability density (Å/meV)')
    axes[2].legend(loc='upper right', title='MAE (meV/Å)', title_fontsize=7, fontsize=7)
    axes[2].text(0.97, 0.45, 'n = 1000', transform=axes[2].transAxes, ha='right', va='top', fontsize=7)
    panel_letter(axes[2], 'c', x=-0.24)
    save(fig, 'Figure_3_accuracy')


# ----------------------------------------------------------------------
# Figure 4: molecular dynamics with MACE
# ----------------------------------------------------------------------

def figure_4():
    """Temperature and energy conservation, torsion distributions, and the ORCA check of MD frames."""
    nvt = read_table(RESULTS_DIR / 'md_nvt_500K.csv')
    nve = read_table(RESULTS_DIR / 'md_nve.csv')
    torsions_mace = read_table(RESULTS_DIR / 'md_torsions.csv')
    torsions_md17 = np.load(DATA_DIR / 'rmd17_ethanol_torsions.npz')
    md_forces = read_table(RESULTS_DIR / 'md_orca_check_forces.csv')
    summary = json.loads((RESULTS_DIR / 'md_summary.json').read_text())

    fig = plt.figure(figsize=(FIG_WIDTH, 4.6))
    grid = GridSpec(2, 3, figure=fig, left=0.08, right=0.98, bottom=0.1, top=0.95, wspace=0.5, hspace=0.55)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(nvt['time_ps'], nvt['temperature_K'], color=MACE_COLOR, lw=0.4)
    ax.axhline(500, color=REFERENCE_COLOR, ls='--', lw=1.0)
    ax.set_xlabel('time (ps)')
    ax.set_ylabel('temperature (K)')
    ax.text(0.97, 0.95, f"NVT, mean {summary['nvt_mean_temperature_K']:.0f} K", transform=ax.transAxes,
            ha='right', va='top', fontsize=7)
    panel_letter(ax, 'a', x=-0.27)

    ax = fig.add_subplot(grid[0, 1:])
    time_nve = nve['time_ps'] - nve['time_ps'][0]
    ax.plot(time_nve, (nve['potential_energy_eV'] - nve['total_energy_eV'][0]) * 1000, color=ACCENT_COLORS[0],
            lw=0.5, label='potential')
    ax.plot(time_nve, (nve['kinetic_energy_eV']) * 1000, color=ACCENT_COLORS[1], lw=0.5, label='kinetic')
    ax.plot(time_nve, (nve['total_energy_eV'] - nve['total_energy_eV'][0]) * 1000, color=REFERENCE_COLOR, lw=1.3,
            label='total')
    ax.set_xlabel('time in NVE (ps)')
    ax.set_ylabel('energy (meV)')
    # room above and below the curves for the legend and the drift
    largest = max(np.abs(ax.get_ylim()))
    ax.set_ylim(-1.45 * largest, 1.45 * largest)
    ax.legend(loc='upper center', ncol=3)
    ax.text(0.5, 0.03, f"drift of the total energy {summary['nve_drift_meV_per_ps']:+.2f} meV/ps, "
                        f"std {summary['nve_total_energy_std_meV']:.2f} meV",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
    panel_letter(ax, 'b', x=-0.11)

    ax = fig.add_subplot(grid[1, 0])
    bins = np.linspace(-180, 180, 73)
    ax.hist(torsions_md17['hydroxyl_deg'], bins=bins, density=True, color=REFERENCE_COLOR, alpha=SHADE_ALPHA + 0.1,
            label='MD17 (DFT)')
    ax.hist(torsions_mace['hydroxyl_deg'], bins=bins, density=True, histtype='step', color=MACE_COLOR, lw=1.4,
            label='MACE')
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -60, 60, 180])
    ax.set_xlabel('hydroxyl torsion (°)')
    ax.set_ylabel('probability density')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
    ax.legend(loc='upper left', fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.2)
    panel_letter(ax, 'c', x=-0.27)

    ax = fig.add_subplot(grid[1, 1])
    bins = np.linspace(0, 120, 41)
    ax.hist(torsions_md17['methyl_deg'], bins=bins, density=True, color=REFERENCE_COLOR, alpha=SHADE_ALPHA + 0.1,
            label='MD17 (DFT)')
    ax.hist(torsions_mace['methyl_deg'], bins=bins, density=True, histtype='step', color=MACE_COLOR, lw=1.4,
            label='MACE')
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 60, 120])
    ax.set_xlabel('methyl torsion (°)')
    ax.set_ylabel('probability density')
    panel_letter(ax, 'd', x=-0.27)

    ax = fig.add_subplot(grid[1, 2])
    parity_axes(ax, md_forces['orca_force_eV_per_A'], md_forces['mace_force_eV_per_A'], MACE_COLOR,
                'ORCA force (eV/Å)', 'MACE force (eV/Å)', size=5, alpha=0.6)
    ax.text(0.05, 0.95, f"{summary['md_frames_checked_with_orca']} MACE frames\n"
                        f"MAE {summary['md_force_mae_meV_per_A']:.1f} meV/Å",
            transform=ax.transAxes, va='top', fontsize=7)
    panel_letter(ax, 'e', x=-0.27)
    save(fig, 'Figure_4_md')


# ----------------------------------------------------------------------
# Banner of the README, in a light and a dark version
# ----------------------------------------------------------------------

def banner(theme):
    """
    Wide picture for the top of the README: the molecule, the force error against the number of training
    configurations (this work and the paper), and the hydroxyl torsion in MACE and ab initio MD.
    theme is 'light' or 'dark' (GitHub dark background #0d1117).
    """
    if theme == 'dark':
        background, foreground, line_color = '#0d1117', '#e6edf3', '#58a6ff'
    else:
        background, foreground, line_color = 'white', 'black', MACE_COLOR
    colors = {'figure.facecolor': background, 'axes.facecolor': background, 'savefig.facecolor': background,
              'text.color': foreground, 'axes.labelcolor': foreground, 'axes.edgecolor': foreground,
              'xtick.color': foreground, 'ytick.color': foreground}
    errors = read_table(RESULTS_DIR / 'test_errors.csv')
    torsions_mace = read_table(RESULTS_DIR / 'md_torsions.csv')
    torsions_md17 = np.load(DATA_DIR / 'rmd17_ethanol_torsions.npz')
    first = read(DATA_DIR / 'ethanol_test.xyz', 0)

    with plt.rc_context(colors):
        fig = plt.figure(figsize=(10.0, 2.6))
        grid = GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1.0, 1.0], left=0.02, right=0.98, bottom=0.2,
                        top=0.93, wspace=0.35)
        ax = fig.add_subplot(grid[0])
        if theme == 'dark':
            # lighter atoms, so that carbon stays visible on the dark background
            draw_molecule(ax, first, colors={'H': '#d0d7de', 'C': '#6e7681', 'O': '#ff7b72'}, labels=False)
        else:
            draw_molecule(ax, first, labels=False)

        ax = fig.add_subplot(grid[1])
        ax.plot(errors['n_train'], errors['force_mae_meV_per_A'], marker='o', color=line_color,
                label='MACE, laptop CPU')
        paper_sizes = sorted(PAPER_ETHANOL)
        paper_values = [PAPER_ETHANOL[size]['force_mae_meV_per_A'] for size in paper_sizes]
        ax.plot(paper_sizes, paper_values, marker='s', ls='none', markersize=6, markerfacecolor=background,
                markeredgecolor=PAPER_COLOR if theme == 'light' else '#ff7b72', markeredgewidth=1.4,
                label='MACE paper, A100 GPU')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([50, 100, 250, 500, 1000])
        ax.set_xticklabels(['50', '100', '250', '500', '1000'])
        ax.set_yticks([3, 10, 30])
        ax.set_yticklabels(['3', '10', '30'])
        ax.set_ylim(1.2, 90)
        ax.minorticks_off()
        ax.set_xlabel('training configurations (rMD17 ethanol)')
        ax.set_ylabel('force MAE (meV/Å)')
        ax.legend(loc='upper right')

        ax = fig.add_subplot(grid[2])
        bins = np.linspace(-180, 180, 73)
        ax.hist(torsions_md17['hydroxyl_deg'], bins=bins, density=True, color=foreground, alpha=0.18,
                label='ab initio MD (MD17)')
        ax.hist(torsions_mace['hydroxyl_deg'], bins=bins, density=True, histtype='step', color=line_color, lw=1.5,
                label='MACE MD, 40 ps')
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -60, 60, 180])
        ax.set_yticks([])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.set_xlabel('hydroxyl torsion at 500 K (°)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
        fig.savefig(DOCS_IMAGES / f'banner_{theme}.png', dpi=200)
        plt.close(fig)
    print(f'wrote docs/images/banner_{theme}.png')


def banners():
    """Both versions of the README banner."""
    banner('light')
    banner('dark')


# ----------------------------------------------------------------------
# Animation of the MD run
# ----------------------------------------------------------------------

def find_hydroxyl_jump(hydroxyl, frames_in_movie, first_frame):
    """
    First frame of a piece of the run in which the hydroxyl torsion (degrees) goes from trans (near 180) to
    gauche (near +-60), so that the animation shows a jump between the minima. Starts the search at first_frame.
    """
    is_trans = np.abs(hydroxyl) > 150
    is_gauche = (np.abs(hydroxyl) > 30) & (np.abs(hydroxyl) < 90)
    for i in range(first_frame, len(hydroxyl) - frames_in_movie):
        # trans now and gauche within the first half of the piece; start a quarter of the piece earlier
        if is_trans[i] and is_gauche[i:i + frames_in_movie // 2].any():
            return max(i - frames_in_movie // 4, first_frame)
    return first_frame


def md_movie():
    """
    Animated GIF of 2 ps of the NVT run of step 5: the molecule (left), drawn in the frame of its C-C-O atoms so
    that it does not tumble, with the hydroxyl hydrogen in orange, and the hydroxyl torsion against time (right).
    """
    frames = read(WORK_DIR / 'md' / 'nvt_500K.traj', ':')
    positions = np.array([frame.positions for frame in frames])
    oxygen, carbon_1, carbon_2, hydroxyl_hydrogen, methyl_hydrogens = torsion_atoms(frames[0].numbers,
                                                                                      positions[0])
    hydroxyl = dihedral_angles(positions, hydroxyl_hydrogen, oxygen, carbon_1, carbon_2)
    # on 0 ... 360 degrees trans sits in the middle (180) and the gauche minima at 60 and 300, so the curve does
    # not wrap around while the molecule is trans
    hydroxyl_360 = hydroxyl % 360.0
    times_ps = np.arange(len(frames)) * SAVED_FRAME_FS / 1000.0

    # a piece of the production run (after the 10 ps of equilibration) that holds a trans-gauche jump
    frames_in_movie = int(round(MOVIE_PS * 1000.0 / SAVED_FRAME_FS))
    first_production = int(round(10.0 * 1000.0 / SAVED_FRAME_FS))
    start = find_hydroxyl_jump(hydroxyl, frames_in_movie, first_production)
    chosen = list(range(start, start + frames_in_movie, MOVIE_STEP))
    piece_times = times_ps[start:start + frames_in_movie] - times_ps[start]
    piece_hydroxyl = hydroxyl_360[start:start + frames_in_movie].copy()
    # where the angle passes 0/360 the curve would jump across the whole panel; leave a gap there instead
    wraps = np.abs(np.diff(piece_hydroxyl)) > 180.0
    piece_hydroxyl[1:][wraps] = np.nan

    fig = plt.figure(figsize=(6.4, 2.6), dpi=120)
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.5], left=0.02, right=0.89, bottom=0.2, top=0.88,
                    wspace=0.3)
    molecule_ax = fig.add_subplot(grid[0])
    torsion_ax = fig.add_subplot(grid[1])
    pictures = []
    for i in chosen:
        molecule_ax.cla()
        draw_molecule(molecule_ax, frames[i], labels=False, highlight=(hydroxyl_hydrogen, ACCENT_COLORS[1]))
        # fixed limits (Å), so that the picture does not jump from frame to frame
        molecule_ax.set_xlim(-2.6, 2.6)
        molecule_ax.set_ylim(-2.3, 2.3)
        molecule_ax.set_title(f'MACE, 500 K,  t = {piece_times[i - start]:.2f} ps', fontsize=8)

        torsion_ax.cla()
        torsion_ax.plot(piece_times, piece_hydroxyl, color='0.8', lw=0.8)
        done = i - start + 1
        torsion_ax.plot(piece_times[:done], piece_hydroxyl[:done], color=MACE_COLOR, lw=1.2)
        torsion_ax.plot(piece_times[done - 1], piece_hydroxyl[done - 1], 'o', color=ACCENT_COLORS[1],
                        markersize=5)
        for angle, name in [(60, 'gauche'), (180, 'trans'), (300, 'gauche')]:
            torsion_ax.axhline(angle, color='0.6', lw=0.6, ls=':')
            torsion_ax.text(MOVIE_PS * 1.02, angle, name, fontsize=7, ha='left', va='center', color='0.4')
        torsion_ax.set_xlim(0, MOVIE_PS)
        torsion_ax.set_ylim(0, 360)
        torsion_ax.set_yticks([0, 60, 180, 300, 360])
        torsion_ax.set_xlabel('time (ps)')
        torsion_ax.set_ylabel('hydroxyl torsion (°)')

        fig.canvas.draw()
        picture = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
        pictures.append(picture.convert('P', palette=Image.Palette.ADAPTIVE, colors=64))
    plt.close(fig)

    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    pictures[0].save(DOCS_IMAGES / 'md_ethanol.gif', save_all=True, append_images=pictures[1:],
                     duration=MOVIE_FRAME_MS, loop=0)
    print(f'wrote docs/images/md_ethanol.gif ({len(pictures)} pictures, from {times_ps[start]:.2f} ps)')


def main():
    figures = {'1': figure_1, '2': figure_2, '3': figure_3, '4': figure_4, 'banner': banners, 'movie': md_movie}
    chosen = sys.argv[1:] or list(figures)
    for number in chosen:
        try:
            figures[number]()
        except FileNotFoundError as error:
            print(f'Figure {number} skipped: {error.filename} is missing')


if __name__ == '__main__':
    main()
