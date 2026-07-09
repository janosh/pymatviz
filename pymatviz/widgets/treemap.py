"""Zoomable treemap visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets._traits import optional_trait as _optional
from pymatviz.widgets.matterviz import MatterVizWidget


_LABEL_TEXTS = "label value percent label+value label+percent label+parent-percent"
_LABEL_FITS = ["hide", "shrink", "clip"]
_VALUE_MODES = ["leaf-sum", "total", "remainder"]  # plotly branchvalues semantics


class TreemapWidget(MatterVizWidget):
    """MatterViz widget for zoomable treemap charts of hierarchical data.

    Data is a nested node dict (or list of root nodes) with plotly-like
    semantics: ``{"label": ..., "value": ..., "children": [...]}`` plus
    optional ``id``, ``color``, ``hatch``, and ``metadata`` keys. Clicking a
    cell zooms into it (breadcrumb pathbar zooms back out); ``zoom_root_id``
    syncs both ways: zooming in the UI notifies Python and setting it from
    Python re-roots the view (None = data root).

    Examples:
        >>> from pymatviz import TreemapWidget
        >>> nodes = {"label": "systems", "children": [{"label": "Li-Fe-O", "value": 4}]}
        >>> widget = TreemapWidget(data=nodes)
    """

    data = tl.Any(allow_none=True).tag(sync=True)
    value_mode = _optional(tl.CaselessStrEnum, values=_VALUE_MODES)
    sort = _optional(tl.CaselessStrEnum, values=["descending", "ascending", "none"])
    # fraction by which each depth level lightens its inherited color
    level_lighten = _optional(tl.Float)
    # group siblings below this fraction of the total into one "Other" cell (0 = off)
    min_fraction = _optional(tl.Float)
    other_label = _optional(tl.Unicode)  # label for the aggregated "Other" cell
    max_depth = _optional(tl.Int)  # levels shown below the zoom root (0 = all)
    padding_inner = _optional(tl.Float)  # px gap between sibling cells
    padding_top = _optional(tl.Float)  # px header strip on branch cells (0 = none)
    padding_outer = _optional(tl.Float)  # px inset of children within their parent
    show_labels = _optional(tl.Bool)
    # label content (plotly textinfo equivalent)
    label_text = _optional(tl.CaselessStrEnum, values=_LABEL_TEXTS.split())
    label_fit = _optional(tl.CaselessStrEnum, values=_LABEL_FITS)  # overflow mode
    label_min_font_size = _optional(tl.Float)  # px floor in shrink mode
    label_max_font_size = _optional(tl.Float)  # px ceiling for leaf labels
    parent_label_font_size = _optional(tl.Float)  # px size for branch headers
    zoom_on_click = _optional(tl.Bool)
    show_breadcrumbs = _optional(tl.Bool)  # clickable ancestor pathbar when zoomed
    legend = _optional(tl.Dict)
    show_legend = _optional(tl.Bool)  # depth-1 category legend
    value_format = _optional(tl.Unicode)  # d3-format string for labels/tooltips
    # chart-edge padding {"t": N, "b": N, "l": N, "r": N} in pixels
    padding = _optional(tl.Dict)
    # SVG/PNG download buttons in the controls pane + their base filename
    export_buttons = _optional(tl.Bool)
    export_filename = _optional(tl.Unicode)
    fullscreen_toggle = _optional(tl.Bool)
    # two-way synced zoom state: id of the cell the view is rooted on
    zoom_root_id = tl.Any(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self, data: dict[str, Any] | list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> None:
        """Initialize a treemap widget.

        Args:
            data: Hierarchy as a nested node dict or list of root nodes. Each
                node supports ``label``, ``value`` (required on leaves unless
                ``value_mode="total"``), ``color``, ``children``, ``id``,
                ``hatch``, and ``metadata`` keys.
            **kwargs: Config traits (see class-level trait comments) and
                additional base widget keyword arguments.
        """
        for key in ("legend", "padding"):
            if key in kwargs:
                kwargs[key] = normalize_plot_json(kwargs[key], f"Treemap.{key}")
        super().__init__(
            widget_type="treemap",
            data=normalize_plot_json(data, "Treemap.data"),
            **kwargs,
        )
