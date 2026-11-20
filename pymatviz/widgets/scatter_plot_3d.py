"""3D scatter plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_json
from pymatviz.widgets._traits import (
    PlotControlsTraits,
    color_scale_trait,
    optional_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlot3DWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for 3D scatter/surface plots with camera controls.

    Examples:
        >>> from pymatviz import ScatterPlot3DWidget
        >>> widget = ScatterPlot3DWidget(
        ...     series=[{"x": [1, 2], "y": [3, 4], "z": [5, 6], "label": "pts"}],
        ... )
    """

    series = optional_trait(tl.List)
    surfaces = optional_trait(tl.List)
    ref_lines = optional_trait(tl.List)
    ref_planes = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)
    z_axis = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    styles = optional_trait(tl.Dict)
    color_scale = color_scale_trait()
    size_scale = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    show_legend = optional_trait(tl.Bool)
    camera_projection = tl.Unicode(default_value="perspective").tag(sync=True)
    # matterviz InstanceTween {duration, easing, ...} animating point changes
    point_tween = optional_trait(tl.Dict)

    _JSON_KWARGS = (
        "surfaces", "ref_lines", "ref_planes", "x_axis", "y_axis", "z_axis", "display",
        "styles", "color_scale", "size_scale", "legend", "point_tween",
    )  # fmt: skip

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
        for idx, entry in enumerate(series or []):
            prefix = f"ScatterPlot3D series entry at index {idx}"
            if not isinstance(entry, dict):
                raise TypeError(f"{prefix} must be a dict, got {type(entry).__name__}")
            if not all(isinstance(key, str) for key in entry):
                raise TypeError(f"{prefix} has non-string keys: {list(entry)}")
            if missing := {"x", "y", "z"} - entry.keys():
                raise ValueError(
                    f"{prefix} is missing required key(s): {sorted(missing)}. "
                    f"Got keys: {sorted(entry)}"
                )
            lengths = {axis: len(entry[axis]) for axis in "xyz"}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"{prefix} has mismatched x/y/z lengths: {lengths}")

        normalize_json_kwargs(kwargs, self._JSON_KWARGS, "ScatterPlot3D")
        super().__init__(
            widget_type="scatter_plot_3d",
            series=normalize_plot_json(series, "ScatterPlot3D.series"),
            **kwargs,
        )
