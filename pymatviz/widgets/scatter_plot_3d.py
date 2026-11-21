"""3D scatter plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._binary import is_binary_envelope, plot_series_envelope
from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_json
from pymatviz.widgets._traits import (
    CAMERA_PROJECTIONS,
    PlotControlsTraits,
    color_scale_trait,
    optional_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlot3DWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for 3D scatter/surface plots with camera controls.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.scatter_plot_3d(
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
    camera_projection = tl.CaselessStrEnum(
        CAMERA_PROJECTIONS, default_value="perspective"
    ).tag(sync=True)
    # matterviz InstanceTween {duration, easing, ...} animating point changes
    point_tween = optional_trait(tl.Dict)

    _JSON_KWARGS = (
        "surfaces", "ref_lines", "ref_planes", "x_axis", "y_axis", "z_axis", "display",
        "styles", "color_scale", "size_scale", "legend", "point_tween",
    )  # fmt: skip

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Check series on init and reassignment: dicts with string keys and equal
        length ``x``/``y``/``z``. Large numeric series sync as binary plot-series
        envelopes (``z`` included), others as finite JSON.

        Raises:
            tl.TraitError: For invalid series (prior data is kept).
        """
        if (series := proposal["value"]) is None:
            return None
        try:
            return [
                self._normalize_entry(entry, idx) for idx, entry in enumerate(series)
            ]
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc

    @staticmethod
    def _normalize_entry(entry: Any, idx: int) -> dict[str, Any]:
        """Validate one 3D series and encode it as a binary envelope or JSON dict.

        Raises:
            TypeError: For non-dict entries or non-string keys.
            ValueError: For missing or mismatched x/y/z or non-finite values.
        """
        if is_binary_envelope(entry, "plot-series"):
            return entry
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
        name = f"ScatterPlot3D.series[{idx}]"
        envelope = plot_series_envelope(entry, name, ("x", "y", "z"))
        return envelope or normalize_plot_json(entry, name)

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
        normalize_json_kwargs(kwargs, self._JSON_KWARGS, "ScatterPlot3D")
        super().__init__(widget_type="scatter_plot_3d", series=series, **kwargs)
