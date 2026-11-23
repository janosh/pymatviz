"""Grid layout of several titled widgets."""

from __future__ import annotations

import html
import numbers
import os
from typing import TYPE_CHECKING, Any

import ipywidgets as ipw


if TYPE_CHECKING:
    from collections.abc import Mapping


class WidgetGrid(ipw.GridBox):
    """GridBox of titled widget cells that, with the ``PYMATVIZ_STATIC`` env var set,
    displays as an HTML grid of the cells' PNGs (like single widgets do) instead of
    live widgets.
    """

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any] | None:
        """Live grid, or a static HTML grid of PNGs under ``PYMATVIZ_STATIC``."""
        if not os.environ.get("PYMATVIZ_STATIC"):
            return super()._repr_mimebundle_(**kwargs)
        from pymatviz.widgets.matterviz import MatterVizWidget, static_png_bundle

        if non_matterviz := {
            title.value: type(widget).__name__
            for title, widget in (cell.children for cell in self.children)
            if not isinstance(widget, MatterVizWidget)
        }:
            raise TypeError(
                "PYMATVIZ_STATIC renders only MatterViz widgets as PNGs, got "
                f"{non_matterviz} (cell title: widget type)"
            )
        # cells share the notebook width, so smaller renders keep the output light
        img_kwargs = {"width": 360, "dpi": 96}

        figures = ""
        for title, widget in (cell.children for cell in self.children):
            png = static_png_bundle(widget, **img_kwargs)["image/png"]
            figures += (
                f"<figure style='margin:0'><figcaption>{title.value}</figcaption>"
                f"<img src='data:image/png;base64,{png}' style='width:100%'></figure>"
            )
        cols = self.layout.grid_template_columns
        style = f"display:grid;gap:8px;grid-template-columns:{cols}"
        return {
            "text/html": f"<div style='{style}'>{figures}</div>",
            "text/plain": "<WidgetGrid>",
        }


def widget_grid(widgets: Mapping[Any, ipw.DOMWidget], n_cols: int = 3) -> WidgetGrid:
    """Lay out labeled widgets in a grid with n_cols columns, e.g. several structures:
    ``widget_grid({key: StructureWidget(structure=struct) for key, struct in ...})``.
    Each cell is a VBox of an HTML title and the widget, so
    ``grid.children[idx].children[1]`` is the idx-th widget.

    Args:
        widgets (Mapping[Any, ipw.DOMWidget]): Cell titles mapped to their widgets.
        n_cols (int): Number of grid columns, capped at the number of widgets.
            Defaults to 3.

    Returns:
        WidgetGrid: An ipywidgets GridBox of titled cells.

    Raises:
        TypeError: If n_cols is not an int (bools and floats like 2.5 aren't).
        ValueError: If n_cols is not positive or widgets is empty.
    """
    # n_cols goes into CSS repeat(), which needs a positive integer
    if isinstance(n_cols, bool) or not isinstance(n_cols, numbers.Integral):
        raise TypeError(f"{n_cols=} must be an int")
    if n_cols < 1:
        raise ValueError(f"{n_cols=} must be positive")
    if not widgets:  # repeat(0, ...) is invalid CSS
        raise ValueError("widget_grid needs at least one widget")
    cells = [
        ipw.VBox([ipw.HTML(f"<b>{html.escape(str(label))}</b>"), widget])
        for label, widget in widgets.items()
    ]
    n_cols = min(int(n_cols), len(cells))
    return WidgetGrid(
        cells,
        layout=ipw.Layout(grid_template_columns=f"repeat({n_cols}, minmax(0, 1fr))"),
    )
