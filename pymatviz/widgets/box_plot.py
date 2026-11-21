"""Box, violin and raincloud plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs
from pymatviz.widgets._schema_traits import BoxPlotTraits
from pymatviz.widgets._traits import CartesianPlotTraits, df_series, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BoxPlotWidget(CartesianPlotTraits, BoxPlotTraits, MatterVizWidget):
    """MatterViz widget for box, violin and raincloud plots of raw distributions.

    Each series holds the raw samples in ``y``; quartiles, whiskers and the KDE are
    computed in the frontend. ``kind="violin+box"`` with ``side="positive"``,
    ``points="strip"`` and ``point_side="negative"`` draws a raincloud (half violin
    above, jittered samples below the box).

    Examples:
        >>> import pymatviz as pmv
        >>> pmv.box_plot([{"y": [1, 2, 3, 8], "label": "a"}], kind="violin+box")
    """

    # KDE bandwidth in value-axis units, or "silverman"/"scott"
    bandwidth = optional_trait(
        tl.Union, trait_types=[tl.Float(), tl.Enum(["silverman", "scott"])]
    )

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series (raw samples in ``y``, binary when large) before syncing."""
        return self._binary_series(proposal["value"], "BoxPlot", samples_key="y")

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any = None,
        *,
        y: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a box plot widget.

        Args:
            series: Series dicts ``{"y": [...], "label", "color", "kind", "side",
                "bandwidth", "box_width", ...}`` with the raw samples in ``y``, or a
                DataFrame whose columns ``y`` and ``color`` name.
            y: DataFrame column of the samples.
            color: DataFrame column to split into one series per value.
            **kwargs: Widget traits, e.g. ``kind``, ``side``, ``points``,
                ``orientation``, ``whisker_mode``, axis/range configs (see
                ``CartesianPlotTraits``).
        """
        series = df_series(series, color, kwargs, y=y)
        json_keys = (
            *self._JSON_KWARGS, "box", "whisker", "median_style", "sample_style",
            "violin_style", "whisker_percentiles", "kde_clip",
        )  # fmt: skip
        normalize_json_kwargs(kwargs, json_keys, "BoxPlot")
        super().__init__(
            series=series,
            **kwargs,
        )
