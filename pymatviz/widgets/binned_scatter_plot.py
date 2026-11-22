"""Density-binned scatter plot widget for large point clouds."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import BinnedScatterPlotTraits
from pymatviz.widgets._traits import color_bar_trait, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BinnedScatterPlotWidget(BinnedScatterPlotTraits, MatterVizWidget):
    """MatterViz ``BinnedScatterPlot``: bins points into a screen-space density map
    and switches to individual points when zoomed in far enough.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.binned_scatter_plot(
        ...     series=[{"x": [1, 2, 3], "y": [1, 2, 4]}],
        ...     marginals={"top": "histogram", "right": "histogram"},
        ... )
    """

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
    # "density" (bins) or "points"; the frontend switches it as the user zooms
    render_mode = tl.CaselessStrEnum(
        ["density", "points"], default_value="density"
    ).tag(sync=True)
    color_bar = color_bar_trait()

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
        super().__init__(
            series=series,
            **kwargs,
        )

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, binary when large."""
        return self._binary_series(proposal["value"], "BinnedScatterPlot")
