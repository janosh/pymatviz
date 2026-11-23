"""Density-binned scatter plot widget for large point clouds."""

from __future__ import annotations

import traitlets as tl

from pymatviz.widgets._schema_traits import BinnedScatterPlotTraits
from pymatviz.widgets._traits import color_bar_trait, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget, SeriesPlotMixin


class BinnedScatterPlotWidget(
    SeriesPlotMixin, BinnedScatterPlotTraits, MatterVizWidget
):
    """MatterViz ``BinnedScatterPlot``: bins points into a screen-space density map
    and switches to individual points when zoomed in far enough.

    ``series`` entries hold required ``x`` and ``y`` arrays and optional
    ``label``, ``color``, ``point_ids``, ``metadata``, ``size_values``. Options
    include ``density`` (``{"bin_px": 4}``), ``marginals`` (True, "histogram",
    "kde", "cdf", "rug" or a per-side ``{top, right, ...}`` map of those or
    MarginalConfig dicts), ``overlays`` and ``annotation``.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.binned_scatter_plot(
        ...     series=[{"x": [1, 2, 3], "y": [1, 2, 4]}],
        ...     marginals={"top": "histogram", "right": "histogram"},
        ... )
    """

    selected_point_id = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Int()])
    # "density" (bins) or "points"; the frontend switches it as the user zooms
    render_mode = tl.CaselessStrEnum(
        ["density", "points"], default_value="density"
    ).tag(sync=True)
    color_bar = color_bar_trait()

    _positional_traits = ("series",)
