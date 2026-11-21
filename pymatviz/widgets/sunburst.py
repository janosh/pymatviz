"""Zoomable sunburst and icicle charts for Jupyter notebooks."""

from __future__ import annotations

import traitlets as tl

from pymatviz.widgets._traits import HierarchyPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class SunburstWidget(HierarchyPlotTraits, MatterVizWidget):
    """Show a hierarchy as concentric rings or stacked icicle rows.

    Set ``data`` to a node or list of root nodes. Nodes support ``id``,
    ``label``, ``value``, ``children``, ``color``, ``pattern`` and ``metadata``.
    Clicking a branch updates ``zoom_root_id``; assign an ID to zoom from Python.

    Examples:
        >>> import pymatviz as pmv
        >>> pmv.sunburst(data=[{"label": "Fe-O", "value": 4}], shape="icicle")
    """

    _chart_type = "sunburst"
    shape = optional_trait(tl.CaselessStrEnum, values=["sunburst", "icicle"])
    inner_radius = optional_trait(tl.Float)  # center hole as fraction of chart radius
    pad_angle = optional_trait(tl.Float)  # degrees between sibling arcs
    label_rotation = optional_trait(
        tl.CaselessStrEnum, values=["auto", "radial", "tangential", "horizontal"]
    )
