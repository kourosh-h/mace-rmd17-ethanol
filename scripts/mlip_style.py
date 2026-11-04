"""
Shared style of all figures and the paths of the project.

Every plotting script imports this module, so that all figures have the same page width, fonts, colors and
line styles. A value changed here changes every figure.

Figures are FIG_WIDTH = 7.2 inches wide (two-column page), with 8 pt Arial text, and are saved as PDF (text stays
text) and as PNG at 300 dpi for the README. Energies are in meV, forces in meV/Å, lengths in Å and angles in
degrees.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')  # write files only, no window
import matplotlib.pyplot as plt

# the project folder: scripts/, with this file, is one level below it
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
FIGURE_DIR = ROOT / 'figures'
WORK_DIR = ROOT / 'work'
ORCA_DIR = ROOT / 'orca' / 'calculations'   # every ORCA calculation: input, output, gradient

# ----------------------------------------------------------------------
# Page and fonts
# ----------------------------------------------------------------------
FIG_WIDTH = 7.2          # page width (inches)
FS = 8.0                 # axis labels and other text (pt)
FS_PANEL = 9.0           # panel letters "(a)" (pt)
SHADE_ALPHA = 0.18       # opacity of shaded bands

# Type 42 embeds TrueType fonts, so that the text of a PDF stays text
STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'mathtext.default': 'it',
    'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS, 'xtick.labelsize': FS,
    'ytick.labelsize': FS, 'legend.fontsize': FS, 'legend.title_fontsize': FS, 'figure.titlesize': FS,
    'axes.linewidth': 1.3, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 4.2, 'ytick.major.size': 4.2,
    'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.pad': 3.0, 'ytick.major.pad': 3.0, 'axes.labelpad': 2.5, 'axes.titlepad': 3.0,
    'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.handletextpad': 0.4,
    'legend.labelspacing': 0.28, 'legend.borderaxespad': 0.3,
    'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'lines.markeredgewidth': 0.7,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.dpi': 300}
mpl.rcParams.update(STYLE)

# ----------------------------------------------------------------------
# Colors, the same in every figure
# ----------------------------------------------------------------------
REFERENCE_COLOR = '#282B28'   # DFT reference (rMD17 or ORCA)
MACE_COLOR = '#0D4F8B'        # MACE trained in this work
PAPER_COLOR = '#B7371B'       # values published by Batatia et al. (2022)
ACCENT_COLORS = ['#4DBBD5', '#E68613', '#756BB1', '#B7371B', '#0D4F8B']
ELEMENT_COLORS = {'H': '#9E9E9E', 'C': '#282B28', 'O': '#B7371B'}

# Table 1 of Batatia et al., NeurIPS 2022 (arXiv:2206.07697), ethanol row of rMD17
PAPER_ETHANOL = {1000: {'energy_mae_meV': 0.4, 'force_mae_meV_per_A': 2.1},
                 50: {'energy_mae_meV': 6.7, 'force_mae_meV_per_A': 32.6}}


def new_figure(height, ncols=1, nrows=1, **kwargs):
    """A figure of the page width and the given height (inches) with a grid of axes."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, height), **kwargs)
    return fig, axes


def panel_letter(ax, letter, x=-0.16, y=1.04):
    """Write the panel letter "(a)" above the upper-left corner of the axes."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold',
            ha='left', va='bottom')


def save(fig, name):
    """Save the figure as figures/<name>.pdf and figures/<name>.png (300 dpi)."""
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / f'{name}.pdf')
    fig.savefig(FIGURE_DIR / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'wrote figures/{name}.pdf and .png')
