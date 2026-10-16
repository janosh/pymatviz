"""Histogram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_series
from pymatviz.widgets._traits import CartesianPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class HistogramWidget(CartesianPlotTraits, MatterVizWidget):
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
    bar = optional_trait(tl.Dict)  # bar-style config

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize replacements before syncing, including batched updates."""
        try:
            return normalize_plot_series(
                proposal["value"], component_name="Histogram", samples_key="values"
            )
        except (TypeError, ValueError) as exc:
            # TraitError triggers rollback inside hold_trait_notifications().
            raise tl.TraitError(str(exc)) from exc

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a histogram widget.

        Args:
            series: Series dicts ``{"values": [...], "label", "color", ...}`` with
                the raw samples in ``values``.
            **kwargs: Widget traits, e.g. ``bins``, ``mode`` (``single`` or
                ``overlay``), ``selected_series_idx``, axis/range configs (see
                ``CartesianPlotTraits``), ``bar``.
        """
        normalize_json_kwargs(kwargs, (*self._JSON_KWARGS, "bar"), "Histogram")
        super().__init__(
            widget_type="histogram",
            # Preserve the constructor's existing TypeError/ValueError behavior.
            series=normalize_plot_series(
                series, component_name="Histogram", samples_key="values"
            ),
            **kwargs,
        )
