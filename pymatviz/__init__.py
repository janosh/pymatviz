"""pymatviz: A Python package for visualizing materials data.

Plot functions return MatterViz anywidgets that render in notebooks and export
to HTML or static images (``widget.to_html()``, ``widget.to_img()``). Each MatterViz
component has a function of the same name in snake_case (``ScatterPlot`` ->
``scatter_plot``). The widget classes live in ``pymatviz.widgets``.
"""

import builtins
import sys
from importlib.metadata import PackageNotFoundError, version

from pymatviz import (
    bar,
    classify,
    cluster,
    compositions,
    coordination,
    data,
    enums,
    hierarchy,
    notebook,
    phonons,
    powerups,
    ptable,
    py_pkg,
    rdf,
    scatter,
    structures,
    typing,
    uncertainty,
    utils,
    widgets,
    xrd,
)
from pymatviz.bar import spacegroup_bar_plot
from pymatviz.classify import confusion_matrix, precision_recall_curve, roc_curve
from pymatviz.cluster import cluster_compositions
from pymatviz.compositions import count_elements
from pymatviz.coordination import coordination_bar_plot, coordination_vs_cutoff_line
from pymatviz.hierarchy import (
    chem_env_hierarchy,
    chem_sys_hierarchy,
    spacegroup_sunburst,
)
from pymatviz.histogram import elements_hist, histogram
from pymatviz.notebook import notebook_mode
from pymatviz.phonons import phonon_dos
from pymatviz.ptable import (
    ptable_heatmap,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_scatter,
)
from pymatviz.py_pkg import py_pkg_treemap
from pymatviz.rainclouds import rainclouds
from pymatviz.rdf import rdf_plot
from pymatviz.sankey import sankey
from pymatviz.scatter import density_hexbin, density_scatter
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.widgets import (
    configure_assets,
    export_widgets,
    from_json,
    link_selection,
    widget_grid,
)
from pymatviz.xrd import xrd_plot


# components drawn from data as given: their widget classes, under component names
band_structure = widgets.BandStructureWidget
bands_and_dos = widgets.BandsAndDosWidget
bar_plot = widgets.BarPlotWidget
binned_scatter_plot = widgets.BinnedScatterPlotWidget
box_plot = widgets.BoxPlotWidget
brillouin_zone = widgets.BrillouinZoneWidget
chem_pot_diagram = widgets.ChemPotDiagramWidget
composition = widgets.CompositionWidget
convex_hull = widgets.ConvexHullWidget
dos = widgets.DosWidget
fermi_surface = widgets.FermiSurfaceWidget
heatmap_matrix = widgets.HeatmapMatrixWidget
periodic_table = widgets.PeriodicTableWidget
phase_diagram = widgets.PhaseDiagramWidget
scatter_plot = widgets.ScatterPlotWidget
scatter_plot_3d = widgets.ScatterPlot3DWidget
structure = widgets.StructureWidget
sunburst = widgets.SunburstWidget
trajectory = widgets.TrajectoryWidget
treemap = widgets.TreemapWidget


try:
    __version__ = version("pymatviz")
except PackageNotFoundError:
    __version__ = "n/a"  # package not correctly installed


if "marimo" in sys.modules or hasattr(builtins, "__IPYTHON__"):
    from IPython import get_ipython

    # marimo, or an IPython kernel (not just IPython's builtins flag)
    if "marimo" in sys.modules or get_ipython() is not None:
        notebook_mode(on=True)
    del get_ipython

# keep the top-level namespace to the public API
del builtins, sys, PackageNotFoundError, version
