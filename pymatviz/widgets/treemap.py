"""Zoomable treemap visualization widget for Jupyter notebooks."""

from __future__ import annotations

from pymatviz.widgets._schema_traits import TreemapTraits
from pymatviz.widgets._traits import HierarchyPlotTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class TreemapWidget(HierarchyPlotTraits, TreemapTraits, MatterVizWidget):
    """MatterViz widget for zoomable treemap charts of hierarchical data.

    Data is a nested node dict (or list of root nodes) like
    ``{"label": ..., "value": ..., "children": [...]}`` with
    optional ``id``, ``color``, ``pattern``, and ``metadata`` keys. Clicking a
    cell zooms into it (breadcrumb pathbar zooms back out); ``zoom_root_id``
    syncs both ways: zooming in the UI notifies Python and setting it from
    Python re-roots the view (None = data root).

    Examples:
        >>> import pymatviz as pmv
        >>> nodes = {"label": "systems", "children": [{"label": "Li-Fe-O", "value": 4}]}
        >>> widget = pmv.treemap(data=nodes)
    """
