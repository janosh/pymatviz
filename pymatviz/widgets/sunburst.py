"""Zoomable sunburst and icicle charts for Jupyter notebooks."""

from __future__ import annotations

from pymatviz.widgets._schema_traits import SunburstTraits
from pymatviz.widgets._traits import HierarchyPlotTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class SunburstWidget(HierarchyPlotTraits, SunburstTraits, MatterVizWidget):
    """Show a hierarchy as concentric rings or stacked icicle rows.

    Set ``data`` to a node or list of root nodes. Nodes support ``id``,
    ``label``, ``value``, ``children``, ``color``, ``pattern`` and ``metadata``.
    Clicking a branch updates ``zoom_root_id``; assign an ID to zoom from Python.

    Examples:
        >>> import pymatviz as pmv
        >>> pmv.sunburst(data=[{"label": "Fe-O", "value": 4}], shape="icicle")
    """
