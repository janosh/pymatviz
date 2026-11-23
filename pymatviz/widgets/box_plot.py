"""Box, violin and raincloud plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import BoxPlotTraits
from pymatviz.widgets._traits import PlotControlsTraits, df_series, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget, SeriesPlotMixin


class BoxPlotWidget(
    SeriesPlotMixin, PlotControlsTraits, BoxPlotTraits, MatterVizWidget
):
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

    _series_keys = ("y",)  # raw samples

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
                ``orientation``, ``whisker_mode``, axis/range configs (``x_axis``
                dicts, ``[min, max]`` ``x_range`` lists).
        """
        series = df_series(series, color, kwargs, y=y)
        super().__init__(series=series, **kwargs)
