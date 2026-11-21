"""pymatviz: A Python package for visualizing materials data.

Plot functions return MatterViz anywidgets that render in notebooks and export
to HTML or static images (``widget.to_html()``, ``widget.to_img()``).
"""

from __future__ import annotations

import builtins
from importlib.metadata import PackageNotFoundError, version

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
    treemap,
    typing,
    uncertainty,
    utils,
    widgets,
    xrd,
)
from pymatviz.bar import spacegroup_bar
from pymatviz.brillouin import brillouin_zone_3d
from pymatviz.classify import precision_recall_curve, roc_curve
from pymatviz.classify.confusion_matrix import confusion_matrix
from pymatviz.cluster.composition import cluster_compositions
from pymatviz.coordination import coordination_hist, coordination_vs_cutoff_line
from pymatviz.enums import Key, angstrom_per_atom, cubic_angstrom, eV
from pymatviz.histogram import elements_hist, histogram
from pymatviz.notebook import notebook_mode
from pymatviz.phonons import phonon_bands, phonon_bands_and_dos, phonon_dos
from pymatviz.process_data import (
    count_elements,
    count_formulas,
    normalize_phonon_bands,
    normalize_spacegroups,
    sankey_flow_data,
)
from pymatviz.ptable import (
    ptable_heatmap,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_scatter,
)
from pymatviz.rainclouds import rainclouds
from pymatviz.rdf.figures import element_pair_rdfs, full_rdf
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.scatter import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
)
from pymatviz.structure import structure_3d
from pymatviz.sunburst import chem_env_sunburst, chem_sys_sunburst, spacegroup_sunburst
from pymatviz.treemap import chem_env_treemap, chem_sys_treemap, py_pkg_treemap
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import PKG_DIR, ROOT, df_ptable, html_tag, si_fmt, si_fmt_int
from pymatviz.widgets import (
    BandsAndDosWidget,
    BandStructureWidget,
    BarPlotWidget,
    BinnedScatterPlotWidget,
    BoxPlotWidget,
    BrillouinZoneWidget,
    ChemPotDiagramWidget,
    CompositionWidget,
    ConvexHullWidget,
    CoordinationBarPlotWidget,
    DosWidget,
    FermiSurfaceWidget,
    HeatmapMatrixWidget,
    HistogramWidget,
    PeriodicTableWidget,
    PhaseDiagramWidget,
    RdfPlotWidget,
    SankeyWidget,
    ScatterPlot3DWidget,
    ScatterPlotWidget,
    SpacegroupBarPlotWidget,
    StructureWidget,
    SunburstWidget,
    TrajectoryWidget,
    TreemapWidget,
    WidgetLink,
    XrdWidget,
    configure_assets,
    link_selection,
)
from pymatviz.xrd import xrd_pattern


PKG_NAME = "pymatviz"
try:
    __version__ = version(PKG_NAME)
except PackageNotFoundError:
    __version__ = "n/a"  # package not correctly installed


IS_IPYTHON = hasattr(builtins, "__IPYTHON__")

if IS_IPYTHON:
    from IPython import get_ipython

    if get_ipython() is not None:  # a kernel, not just IPython's builtins flag
        notebook_mode(on=True)
