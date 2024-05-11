"""pymatviz: A Python package for visualizing materials data.

Importing this module has side-effects that apply sensible (often, not always) global
defaults settings for plotly and matplotlib like increasing font size, prettier
axis labels (plotly only) and higher figure resolution (matplotlib only).

To use it, simply import this module before generating any plots:

import pymatviz
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pymatgen.symmetry.groups import SYMM_DATA

from pymatviz.correlation import marchenko_pastur, marchenko_pastur_pdf
from pymatviz.cumulative import cumulative_error, cumulative_residual
from pymatviz.enums import Key, angstrom_per_atom, cubic_angstrom, eV
from pymatviz.histograms import elements_hist, spacegroup_hist, true_pred_hist
from pymatviz.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.phonons import (
    plot_phonon_bands,
    plot_phonon_bands_and_dos,
    plot_phonon_dos,
)
from pymatviz.ptable import (
    ChildPlotters,
    PTableProjector,
    count_elements,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_lines,
    ptable_scatters,
)
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.structure_viz import plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import PKG_DIR, ROOT, styled_html_tag


PKG_NAME = "pymatviz"
try:
    __version__ = version(PKG_NAME)
except PackageNotFoundError:
    pass  # package not installed


# define a sensible order for crystal systems across plots
crystal_sys_order = (
    "cubic hexagonal trigonal tetragonal orthorhombic monoclinic triclinic".split()
)
# map of space group numbers to symbols
spg_num_to_symbol = {
    v["int_number"]: k for k, v in SYMM_DATA["space_group_encoding"].items()
}
spg_num_to_symbol = dict(sorted(spg_num_to_symbol.items()))  # sort


px.defaults.labels |= {
    "n_atoms": "Atom Count",
    "n_elems": "Element Count",
    "gap expt": "Experimental band gap (eV)",
    "n": "Refractive index n",
    "n_wyckoff": "Number of Wyckoff positions",
} | Key.val_label_dict()

# to hide math loading MathJax message in bottom left corner of plotly PDFs
# https://github.com/plotly/Kaleido/issues/122#issuecomment-994906924
# use pio.kaleido.scope.mathjax = None


plt.rc("font", size=16)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("figure", dpi=200, titlesize=18)
plt.rcParams["figure.constrained_layout.use"] = True

axis_template = dict(
    mirror=True,
    showline=True,
    ticks="outside",
    zeroline=True,
    linewidth=1,
    showgrid=True,
)
white_axis_template = axis_template | dict(linecolor="black", gridcolor="lightgray")
# inherit from plotly_white template
pmv_white_template = go.layout.Template(pio.templates["plotly_white"])

pio.templates[f"{PKG_NAME}_white"] = pmv_white_template.update(
    layout=dict(
        xaxis=white_axis_template, yaxis=white_axis_template, font=dict(color="black")
    )
)
dark_axis_template = axis_template | dict(linecolor="white", gridcolor="darkgray")
# inherit from plotly_dark template
pmv_dark_template = go.layout.Template(pio.templates["plotly_dark"])

pio.templates[f"{PKG_NAME}_dark"] = pmv_dark_template.update(
    layout=dict(
        xaxis=dark_axis_template, yaxis=dark_axis_template, font=dict(color="white")
    )
)
