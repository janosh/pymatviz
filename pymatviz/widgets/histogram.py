"""Histogram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import HistogramTraits
from pymatviz.widgets._traits import (
    JsonBool,
    JsonInt,
    PlotControlsTraits,
    checked_position,
    clamped_position,
    df_series,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget, SeriesPlotMixin


class HistogramWidget(
    SeriesPlotMixin, PlotControlsTraits, HistogramTraits, MatterVizWidget
):
    """MatterViz widget wrapper for histogram visualizations. Each series holds the
    raw samples to bin in ``values``; binning happens in the frontend.

    Assign a new ``series`` to update the data. Samples are normalized to numeric
    lists; invalid updates raise ``traitlets.TraitError`` and preserve prior data.
    """

    bins = JsonInt(default_value=100, min=1).tag(sync=True)
    mode = tl.CaselessStrEnum(values=["single", "overlay"], default_value="single").tag(
        sync=True
    )
    # Two-way synced index of the active series in single-mode controls, in range of
    # the series (replacing them with fewer clamps it to the last one)
    selected_series_idx = JsonInt(default_value=0).tag(sync=True)
    show_legend = JsonBool(default_value=True).tag(sync=True)

    _series_keys = ("values",)  # raw samples
    _binary_series = False

    @validator("selected_series_idx")
    def _check_series_idx(self, proposal: dict[str, Any]) -> int:
        """Reject indices beyond the series (clamping frontend ones)."""
        n_series = len(self.series or [])
        return checked_position(
            self, "selected_series_idx", proposal["value"], n_series, "series"
        )

    @tl.observe("series")
    def _clamp_series_idx(self, change: dict[str, Any]) -> None:
        """Move the active series to the last of fewer new series."""
        self.selected_series_idx = clamped_position(
            self.selected_series_idx, len(change["new"] or [])
        )

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any = None,
        *,
        x: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a histogram widget.

        Args:
            series: Series dicts ``{"values": [...], "label", "color", ...}`` with
                the raw samples in ``values``, or a DataFrame whose columns ``x``
                and ``color`` name.
            x: DataFrame column of the values to bin.
            color: DataFrame column to split into one series per value.
            **kwargs: Widget traits, e.g. ``bins``, ``mode`` (``single`` or
                ``overlay``), ``selected_series_idx``, axis/range configs (``x_axis``
                dicts, ``[min, max]`` ``x_range`` lists), ``bar``.
        """
        series = df_series(series, color, kwargs, values=x)
        super().__init__(series=series, **kwargs)
