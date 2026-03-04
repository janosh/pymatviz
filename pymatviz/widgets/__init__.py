"""MatterViz Jupyter/Marimo notebook extension for visualizing crystal structures,
MD trajectories, band structures, DOS, convex hulls, Fermi surfaces, periodic tables,
RDF plots, heatmap matrices, space group distributions, chemical potential diagrams,
and more.
"""

from __future__ import annotations

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.fermi_surface import FermiSurfaceWidget
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget
from pymatviz.widgets.histogram import HistogramWidget
from pymatviz.widgets.matterviz import configure_assets
from pymatviz.widgets.mime import register_matterviz_widgets
from pymatviz.widgets.periodic_table import PeriodicTableWidget
from pymatviz.widgets.phase_diagram import PhaseDiagramWidget
from pymatviz.widgets.rdf_plot import RdfPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from pymatviz.widgets.xrd import XrdWidget


register_matterviz_widgets()  # Auto-register in all supported environments
