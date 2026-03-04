"""3D scatter plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlot3DWidget(MatterVizWidget):
    """MatterViz widget for 3D scatter/surface plots with camera controls.

    Examples:
        >>> from pymatviz import ScatterPlot3DWidget
        >>> widget = ScatterPlot3DWidget(
        ...     series=[{"x": [1, 2], "y": [3, 4], "z": [5, 6], "label": "pts"}],
        ... )
    """

    series = tl.List(allow_none=True).tag(sync=True)
    surfaces = tl.List(allow_none=True).tag(sync=True)
    ref_lines = tl.List(allow_none=True).tag(sync=True)
    ref_planes = tl.List(allow_none=True).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)
    z_axis = tl.Dict(allow_none=True).tag(sync=True)
    display = tl.Dict(allow_none=True).tag(sync=True)
    styles = tl.Dict(allow_none=True).tag(sync=True)
    color_scale = tl.Dict(allow_none=True).tag(sync=True)
    size_scale = tl.Dict(allow_none=True).tag(sync=True)
    legend = tl.Dict(allow_none=True).tag(sync=True)
    controls = tl.Dict(allow_none=True).tag(sync=True)
    camera_projection = tl.Unicode(default_value="perspective").tag(sync=True)

    def __init__(
        self,
        series: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a 3D scatter plot widget.

        Args:
            series: 3D plot series with ``x``, ``y``, ``z`` arrays and optional
                ``label``, ``color``, ``size`` fields.
            **kwargs: Additional widget properties.
        """
        if series is not None:
            for idx, entry in enumerate(series):
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"ScatterPlot3D series entry at index {idx} must be a "
                        f"dict, got {type(entry).__name__}."
                    )
                missing = {"x", "y", "z"} - entry.keys()
                if missing:
                    raise ValueError(
                        f"ScatterPlot3D series entry at index {idx} is missing "
                        f"required key(s): {sorted(missing)}. "
                        f"Got keys: {sorted(entry)}."
                    )

        super().__init__(
            widget_type="scatter_plot_3d",
            series=normalize_plot_json(series, "ScatterPlot3D.series"),
            **kwargs,
        )
