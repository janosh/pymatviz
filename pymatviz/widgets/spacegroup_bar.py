"""Space group bar plot widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class SpacegroupBarPlotWidget(MatterVizWidget):
    """MatterViz widget for space group frequency bar plots.

    Accepts space group numbers/symbols or ``{spacegroup: count}`` mappings and
    renders a bar chart showing their frequency distribution.

    Examples:
        >>> from pymatviz import SpacegroupBarPlotWidget
        >>> widget = SpacegroupBarPlotWidget(data=[225, 225, 166, 62, 62, 62])
    """

    data = tl.List(allow_none=True).tag(sync=True)
    show_counts = tl.Bool(default_value=True).tag(sync=True)
    show_legend = tl.Bool(default_value=False).tag(sync=True)
    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a space group bar plot widget.

        Args:
            data: Space group numbers/Hermann-Mauguin symbols or count mapping.
            **kwargs: Additional widget properties.
        """
        if data is None:
            data = []
        elif isinstance(data, Mapping):
            try:
                counts = {spg: index(count) for spg, count in data.items()}
            except TypeError as exc:
                raise TypeError("Space group counts must be integers.") from exc
            if any(count < 0 for count in counts.values()):
                raise ValueError("Space group counts must be non-negative integers.")
            data = [spg for spg, count in counts.items() for _ in range(count)]
        super().__init__(
            widget_type="spacegroup_bar",
            data=normalize_plot_json(data, "SpacegroupBarPlot.data"),
            **kwargs,
        )
