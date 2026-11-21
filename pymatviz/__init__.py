"""pymatviz: A Python package for visualizing materials data.

Plot functions return MatterViz anywidgets that render in notebooks and export
to HTML or static images (``widget.to_html()``, ``widget.to_img()``).
"""

import builtins
from importlib.metadata import PackageNotFoundError, version

from pymatviz import (
    bar,
    chem_env,
    classify,
    cluster,
    colors,
    composition,
    coordination,
    data,
    enums,
    notebook,
    phonons,
    powerups,
    ptable,
    rdf,
    scatter,
    structures,
    sunburst,
    treemap,
    typing,
    uncertainty,
    utils,
    widgets,
    xrd,
)
from pymatviz.bar import spacegroup_bar
from pymatviz.classify import precision_recall_curve, roc_curve
from pymatviz.classify.confusion_matrix import confusion_matrix
from pymatviz.cluster.composition import cluster_compositions
from pymatviz.composition import count_elements
from pymatviz.coordination import coordination_hist, coordination_vs_cutoff_line
from pymatviz.enums import Key
from pymatviz.histogram import elements_hist, histogram
from pymatviz.notebook import notebook_mode
from pymatviz.phonons import phonon_dos
from pymatviz.ptable import (
    ptable_heatmap,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_scatter,
)
from pymatviz.rainclouds import rainclouds
from pymatviz.rdf.figures import element_pair_rdfs, full_rdf
from pymatviz.sankey import sankey
from pymatviz.scatter import density_hexbin, density_scatter
from pymatviz.sunburst import chem_env_sunburst, chem_sys_sunburst, spacegroup_sunburst
from pymatviz.treemap import chem_env_treemap, chem_sys_treemap, py_pkg_treemap
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
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
    WidgetGrid,
    WidgetLink,
    XrdWidget,
    configure_assets,
    link_selection,
    widget_grid,
)
from pymatviz.xrd import xrd_pattern


try:
    __version__ = version("pymatviz")
except PackageNotFoundError:
    __version__ = "n/a"  # package not correctly installed

if hasattr(builtins, "__IPYTHON__"):
    from IPython import get_ipython

    if get_ipython() is not None:  # a kernel, not just IPython's builtins flag
        notebook_mode(on=True)
    del get_ipython

# keep the top-level namespace to the public API
del builtins, PackageNotFoundError, version
