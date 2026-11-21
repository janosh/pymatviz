"""Density-binned scatter plot widget for large point clouds."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs
from pymatviz.widgets._traits import color_bar_trait, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BinnedScatterPlotWidget(MatterVizWidget):
    """MatterViz ``BinnedScatterPlot``: bins points into a screen-space density map
    and switches to individual points when zoomed in far enough.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.binned_scatter_plot(
        ...     series=[{"x": [1, 2, 3], "y": [1, 2, 4]}],
        ...     marginals={"top": "histogram", "right": "histogram"},
        ... )
    """

    series = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)  # [min, max], None ends fit the data
    y_range = optional_trait(tl.List)
    size_scale = optional_trait(tl.Dict)
    # {bin_px, color_scale, auto_point_mode, bin_click}
    density = optional_trait(tl.Dict)
    # {ref_lines: [...]}: line geometry and style only (no labels or events)
    overlays = optional_trait(tl.Dict)
    padding = optional_trait(tl.Dict)
    range_padding = optional_trait(tl.Float)
    title = optional_trait(tl.Unicode)
    point_labels = optional_trait(tl.Dict)
    selected_point_id = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Int()])
    # True/"histogram"/"kde"/"cdf"/"rug" or a per-side {top, right, ...} map of those
    # or MarginalConfig dicts
    marginals = optional_trait(
        tl.Union,
        trait_types=[
            tl.Bool(),
            tl.CaselessStrEnum(["histogram", "kde", "cdf", "rug"]),
            tl.Dict(),
        ],
    )
    annotation = optional_trait(tl.Unicode)  # text badge auto-placed in free space
    fullscreen_toggle = optional_trait(tl.Bool)
    # "density" (bins) or "points"; the frontend switches it as the user zooms
    render_mode = tl.CaselessStrEnum(
        ["density", "points"], default_value="density"
    ).tag(sync=True)
    color_bar = color_bar_trait()

    _JSON_KWARGS = (
        "x_axis", "y_axis", "x_range", "y_range", "size_scale", "density", "overlays",
        "padding", "point_labels", "marginals", "color_bar",
    )  # fmt: skip

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a binned scatter plot widget.

        Args:
            series: Point series with required ``x`` and ``y`` arrays and optional
                ``label``, ``color``, ``point_ids``, ``metadata``, ``size_values``.
            **kwargs: Widget traits, e.g. ``density`` (``{"bin_px": 4}``),
                ``marginals``, ``overlays``, ``annotation``, ``x_axis``/``y_axis``.
        """
        normalize_json_kwargs(kwargs, self._JSON_KWARGS, "BinnedScatterPlot")
        super().__init__(
            widget_type="binned_scatter_plot",
            series=series,
            **kwargs,
        )

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, binary when large."""
        return self._binary_series(proposal["value"], "BinnedScatterPlot")
