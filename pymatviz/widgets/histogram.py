"""Histogram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_series
from pymatviz.widgets._schema_traits import HistogramTraits
from pymatviz.widgets._traits import PlotControlsTraits, df_series
from pymatviz.widgets.matterviz import MatterVizWidget


class HistogramWidget(PlotControlsTraits, HistogramTraits, MatterVizWidget):
    """MatterViz widget wrapper for histogram visualizations. Each series holds the
    raw samples to bin in ``values``; binning happens in the frontend.

    Assign a new ``series`` to update the data. Samples are normalized to numeric
    lists; invalid updates raise ``traitlets.TraitError`` and preserve prior data.
    """

    bins = tl.Int(default_value=100).tag(sync=True)
    mode = tl.CaselessStrEnum(values=["single", "overlay"], default_value="single").tag(
        sync=True
    )
    # Two-way synced index of the active series in single-mode controls
    selected_series_idx = tl.Int(default_value=0).tag(sync=True)
    show_legend = tl.Bool(default_value=True).tag(sync=True)

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, including batched updates."""
        try:
            return normalize_plot_series(
                proposal["value"], component_name="Histogram", samples_key="values"
            )
        except (TypeError, ValueError) as exc:
            # TraitError triggers rollback inside hold_trait_notifications().
            raise tl.TraitError(str(exc)) from exc

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
