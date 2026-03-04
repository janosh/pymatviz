"""Space group bar plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class SpacegroupBarPlotWidget(MatterVizWidget):
    """MatterViz widget for space group frequency bar plots.

    Accepts a list of space group numbers or symbols and renders a bar chart
    showing their frequency distribution.

    Examples:
        >>> from pymatviz import SpacegroupBarPlotWidget
        >>> widget = SpacegroupBarPlotWidget(data=[225, 225, 166, 62, 62, 62])
    """

    data = tl.List(allow_none=True).tag(sync=True)
    show_counts = tl.Bool(default_value=True).tag(sync=True)
    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self,
        data: list[int | str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a space group bar plot widget.

        Args:
            data: List of space group numbers or Hermann-Mauguin symbols.
            **kwargs: Additional widget properties.
        """
        super().__init__(
            widget_type="spacegroup_bar",
            data=normalize_plot_json(data or [], "SpacegroupBarPlot.data"),
            **kwargs,
        )
