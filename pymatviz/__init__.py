"""pymatviz: A Python package for visualizing materials data.

Importing this module has side-effects that apply sensible (often, not always) global
defaults settings for plotly and matplotlib like increasing font size, prettier
axis labels (plotly only) and higher figure resolution (matplotlib only).

To use it, simply import this module before generating any plots:

import pymatviz
"""

from __future__ import annotations

import builtins
from importlib.metadata import PackageNotFoundError, version

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from pymatviz import (
    bar,
    colors,
    correlation,
    cumulative,
    enums,
    io,
    phonons,
    powerups,
    process_data,
    ptable,
    rdf,
    relevance,
    sankey,
    scatter,
    structure_viz,
    sunburst,
    templates,
    uncertainty,
    utils,
    xrd,
)
from pymatviz.correlation import marchenko_pastur, marchenko_pastur_pdf
from pymatviz.cumulative import cumulative_error, cumulative_residual
from pymatviz.enums import Key, angstrom_per_atom, cubic_angstrom, eV
from pymatviz.histogram import elements_hist, histogram, spacegroup_bar
from pymatviz.io import df_to_html, df_to_pdf, df_to_svg, save_fig
from pymatviz.phonons import phonon_bands, phonon_bands_and_dos, phonon_dos
from pymatviz.process_data import count_elements
from pymatviz.ptable import (
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_lines,
    ptable_scatters,
)
from pymatviz.rdf import element_pair_rdfs
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.scatter import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_plotly,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.structure_viz import (
    structure_2d,
    structure_2d_plotly,
    structure_3d_plotly,
)
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.templates import (
    pmv_dark_template,
    pmv_white_template,
    set_plotly_template,
)
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import (
    BACKENDS,
    PKG_DIR,
    ROOT,
    df_ptable,
    html_tag,
    si_fmt,
    si_fmt_int,
)
from pymatviz.xrd import xrd_pattern


PKG_NAME = "pymatviz"
try:
    __version__ = version(PKG_NAME)
except PackageNotFoundError:
    pass  # package not installed


IS_IPYTHON = hasattr(builtins, "__IPYTHON__")

# define a sensible order for crystal systems across plots
crystal_sys_order = (
    "cubic hexagonal trigonal tetragonal orthorhombic monoclinic triclinic".split()
)

px.defaults.labels |= {
    "gap expt": "Experimental band gap (eV)",
} | Key.val_label_dict()

# to hide math loading MathJax message in bottom left corner of plotly PDFs
# https://github.com/plotly/Kaleido/issues/122#issuecomment-994906924
# use pio.kaleido.scope.mathjax = None
