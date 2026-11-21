"""Box, violin and raincloud plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs
from pymatviz.widgets._traits import CartesianPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BoxPlotWidget(CartesianPlotTraits, MatterVizWidget):
    """MatterViz widget for box, violin and raincloud plots of raw distributions.

    Each series holds the raw samples in ``y``; quartiles, whiskers and the KDE are
    computed in the frontend. ``kind="violin+box"`` with ``side="positive"``,
    ``points="strip"`` and ``point_side="negative"`` draws a raincloud (half violin
    above, jittered samples below the box).

    Examples:
        >>> import pymatviz as pmv
        >>> pmv.box_plot([{"y": [1, 2, 3, 8], "label": "a"}], kind="violin+box")
    """

    orientation = optional_trait(tl.CaselessStrEnum, values=["vertical", "horizontal"])
    title = optional_trait(tl.Unicode)
    box = optional_trait(tl.Dict)  # {color, opacity, stroke_width, ...}
    whisker = optional_trait(tl.Dict)  # {width, color, cap_fraction}
    median_style = optional_trait(tl.Dict)  # {width, color}
    sample_style = optional_trait(tl.Dict)  # {radius, opacity, stroke_width}
    whisker_mode = optional_trait(
        tl.CaselessStrEnum, values=["tukey", "minmax", "percentile", "std"]
    )
    whisker_range = optional_trait(tl.Float)  # IQR (tukey) or std multiple
    whisker_percentiles = optional_trait(tl.List)  # [low, high] for percentile mode
    show_outliers = optional_trait(tl.Bool)
    show_mean = optional_trait(tl.Bool)
    points = optional_trait(tl.CaselessStrEnum, values=["none", "strip", "swarm"])
    show_value_labels = optional_trait(tl.Bool)
    value_label_stat = optional_trait(tl.CaselessStrEnum, values=["median", "mean"])
    value_label_format = optional_trait(tl.Unicode)  # d3-format string
    kind = optional_trait(tl.CaselessStrEnum, values=["box", "violin", "violin+box"])
    side = optional_trait(tl.CaselessStrEnum, values=["both", "positive", "negative"])
    # KDE bandwidth in value-axis units, or "silverman"/"scott"
    bandwidth = optional_trait(
        tl.Union, trait_types=[tl.Float(), tl.Enum(["silverman", "scott"])]
    )
    violin_width = optional_trait(tl.Float)  # fraction of the category slot
    violin_style = optional_trait(tl.Dict)  # {opacity, stroke_width}
    # which half of the slot strip/swarm points occupy; None follows a half violin
    point_side = optional_trait(
        tl.CaselessStrEnum, values=["both", "positive", "negative"]
    )
    kde_clip = optional_trait(tl.List)  # hard KDE bounds [low, high], None = open

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series (raw samples in ``y``, binary when large) before syncing."""
        return self._binary_series(proposal["value"], "BoxPlot", samples_key="y")

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a box plot widget.

        Args:
            series: Series dicts ``{"y": [...], "label", "color", "kind", "side",
                "bandwidth", "box_width", ...}`` with the raw samples in ``y``.
            **kwargs: Widget traits, e.g. ``kind``, ``side``, ``points``,
                ``orientation``, ``whisker_mode``, axis/range configs (see
                ``CartesianPlotTraits``).
        """
        json_keys = (
            *self._JSON_KWARGS, "box", "whisker", "median_style", "sample_style",
            "violin_style", "whisker_percentiles", "kde_clip",
        )  # fmt: skip
        normalize_json_kwargs(kwargs, json_keys, "BoxPlot")
        super().__init__(
            widget_type="box_plot",
            series=series,
            **kwargs,
        )
