"""Bar plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_series
from pymatviz.widgets._schema_traits import BarPlotTraits
from pymatviz.widgets._traits import CartesianPlotTraits, color_scale_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def bar_axis_defaults(
    kwargs: dict[str, Any], *, category: dict[str, Any], value: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Category and value axis defaults keyed by the axis each lands on: the category
    axis is x for vertical bars and y for ``orientation="horizontal"`` in ``kwargs``.
    Merge the user's kwargs over them with ``with_defaults``.
    """
    horizontal = str(kwargs.get("orientation")).lower() == "horizontal"
    cat_key, val_key = ("y_axis", "x_axis") if horizontal else ("x_axis", "y_axis")
    return {cat_key: category, val_key: value}


class BarPlotWidget(CartesianPlotTraits, BarPlotTraits, MatterVizWidget):
    """MatterViz widget wrapper for grouped/stacked/overlay bar plots."""

    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    mode = tl.CaselessStrEnum(
        values=["overlay", "stacked", "grouped"], default_value="overlay"
    ).tag(sync=True)
    color_scale = color_scale_trait()

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment (arrays to lists, reject
        non-finite values) before syncing.
        """
        try:
            return normalize_plot_series(proposal["value"], component_name="BarPlot")
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a bar plot widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays.
            **kwargs: Widget traits, e.g. ``orientation``, ``mode`` (``overlay``,
                ``stacked`` or ``grouped``), axis/range configs (see
                ``CartesianPlotTraits``), ``bar``, ``line``, ``color_scale``.
        """
        json_keys = (*self._JSON_KWARGS, "bar", "line", "color_scale", "size_scale")
        normalize_json_kwargs(kwargs, json_keys, "BarPlot")
        super().__init__(series=series, **kwargs)
