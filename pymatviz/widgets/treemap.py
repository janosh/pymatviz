"""Zoomable treemap visualization widget for Jupyter notebooks."""

from __future__ import annotations

import traitlets as tl

from pymatviz.widgets._traits import HierarchyPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class TreemapWidget(HierarchyPlotTraits, MatterVizWidget):
    """MatterViz widget for zoomable treemap charts of hierarchical data.

    Data is a nested node dict (or list of root nodes) like
    ``{"label": ..., "value": ..., "children": [...]}`` with
    optional ``id``, ``color``, ``pattern``, and ``metadata`` keys. Clicking a
    cell zooms into it (breadcrumb pathbar zooms back out); ``zoom_root_id``
    syncs both ways: zooming in the UI notifies Python and setting it from
    Python re-roots the view (None = data root).

    Examples:
        >>> from pymatviz import TreemapWidget
        >>> nodes = {"label": "systems", "children": [{"label": "Li-Fe-O", "value": 4}]}
        >>> widget = TreemapWidget(data=nodes)
    """

    _chart_type = "treemap"
    padding_inner = optional_trait(tl.Float)  # px gap between sibling cells
    padding_top = optional_trait(tl.Float)  # px header strip on branch cells (0 = none)
    padding_outer = optional_trait(tl.Float)  # px inset of children within their parent
    label_fit = optional_trait(
        tl.CaselessStrEnum, values=["hide", "shrink", "clip"]
    )  # overflow mode
    label_min_font_size = optional_trait(tl.Float)  # px floor in shrink mode
    label_max_font_size = optional_trait(tl.Float)  # px ceiling for leaf labels
    parent_label_font_size = optional_trait(tl.Float)  # px size for branch headers
