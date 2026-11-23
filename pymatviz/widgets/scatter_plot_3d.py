"""3D scatter plot widget for Jupyter notebooks."""

from __future__ import annotations

import traitlets as tl

from pymatviz.widgets._schema_traits import ScatterPlot3dTraits
from pymatviz.widgets._traits import (
    CAMERA_PROJECTIONS,
    PlotControlsTraits,
    color_scale_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget, SeriesPlotMixin


class ScatterPlot3DWidget(
    SeriesPlotMixin, PlotControlsTraits, ScatterPlot3dTraits, MatterVizWidget
):
    """MatterViz widget for 3D scatter/surface plots with camera controls.

    ``series`` entries hold ``x``, ``y``, ``z`` arrays and optional ``label``,
    ``color``, ``size`` fields.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.scatter_plot_3d(
        ...     series=[{"x": [1, 2], "y": [3, 4], "z": [5, 6], "label": "pts"}],
        ... )
    """

    color_scale = color_scale_trait()
    camera_projection = tl.CaselessStrEnum(
        CAMERA_PROJECTIONS, default_value="perspective"
    ).tag(sync=True)

    _series_keys = ("x", "y", "z")
    _positional_traits = ("series",)
