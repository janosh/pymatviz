"""pymatviz: A Python package for visualizing materials data.

Importing this module has side-effects that apply (usually) sensible global
defaults for plotly like increasing font size, prettier
axis labels with sub/superscripts.
"""

from __future__ import annotations

import builtins
import os
from importlib.metadata import PackageNotFoundError, version

import plotly.express as px

from pymatviz import (
    bar,
    brillouin,
    chem_env,
    classify,
    cluster,
    colors,
    coordination,
    data,
    enums,
    io,
    notebook,
    phonons,
    powerups,
    process_data,
    ptable,
    rdf,
    sankey,
    scatter,
    structure,
    sunburst,
    templates,
    treemap,
    typing,
    uncertainty,
    utils,
    widgets,
    xrd,
)
from pymatviz.bar import spacegroup_bar
from pymatviz.brillouin import brillouin_zone_3d
from pymatviz.classify import precision_recall_curve_plotly, roc_curve_plotly
from pymatviz.classify.confusion_matrix import confusion_matrix
from pymatviz.cluster.composition import cluster_compositions
from pymatviz.coordination import coordination_hist, coordination_vs_cutoff_line
from pymatviz.enums import Key, angstrom_per_atom, cubic_angstrom, eV
from pymatviz.histogram import elements_hist, histogram
from pymatviz.io import df_to_html, df_to_pdf, df_to_svg, save_fig
from pymatviz.notebook import notebook_mode, set_renderer
from pymatviz.phonons import phonon_bands, phonon_bands_and_dos, phonon_dos
from pymatviz.process_data import count_elements, count_formulas
from pymatviz.ptable import (
    ptable_heatmap_plotly,
    ptable_heatmap_splits_plotly,
    ptable_hists_plotly,
    ptable_scatter_plotly,
)
from pymatviz.rainclouds import rainclouds
from pymatviz.rdf.plotly import element_pair_rdfs, full_rdf
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.scatter import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_plotly,
    density_scatter_with_hist,
)
from pymatviz.structure import structure_2d, structure_3d
from pymatviz.structure.plotly import structure_2d_plotly, structure_3d_plotly
from pymatviz.sunburst import chem_env_sunburst, chem_sys_sunburst, spacegroup_sunburst
from pymatviz.templates import (
    pmv_dark_template,
    pmv_white_template,
    set_plotly_template,
)
from pymatviz.treemap import chem_env_treemap, chem_sys_treemap, py_pkg_treemap
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import PKG_DIR, ROOT, df_ptable, html_tag, si_fmt, si_fmt_int
from pymatviz.widgets import CompositionWidget, StructureWidget, TrajectoryWidget
from pymatviz.xrd import xrd_pattern


PKG_NAME = "pymatviz"
try:
    __version__ = version(PKG_NAME)
except PackageNotFoundError:
    __version__ = "n/a"  # package not correctly installed


IS_IPYTHON = hasattr(builtins, "__IPYTHON__")

# define a sensible order for crystal systems across plots
crystal_sys_order = (
    "cubic",
    "hexagonal",
    "trigonal",
    "tetragonal",
    "orthorhombic",
    "monoclinic",
    "triclinic",
)

px.defaults.labels |= {
    "gap expt": "Experimental band gap (eV)",
} | {key: key.label for key in Key}

# to hide math loading MathJax message in bottom left corner of plotly PDFs
# https://github.com/plotly/Kaleido/issues/122#issuecomment-994906924
# use pio.kaleido.scope.mathjax = None


if os.environ.get("CI"):  # Configure Plotly to be silent in CI
    import plotly.graph_objects as go

    _plotly_fig_orig_show = go.Figure.show  # Store original show method

    # Replace fig.show() method with a noop version for CI environments to avoid
    # spamming logs with huge HTML strings
    go.Figure.show = lambda *_args, **_kwargs: None  # type: ignore[assignment]

notebook_mode(on=IS_IPYTHON)
