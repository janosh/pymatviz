"""3D scatter plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import ScatterPlot3dTraits
from pymatviz.widgets._traits import (
    CAMERA_PROJECTIONS,
    PlotControlsTraits,
    color_scale_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlot3DWidget(PlotControlsTraits, ScatterPlot3dTraits, MatterVizWidget):
    """MatterViz widget for 3D scatter/surface plots with camera controls.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.scatter_plot_3d(
        ...     series=[{"x": [1, 2], "y": [3, 4], "z": [5, 6], "label": "pts"}],
        ... )
    """

    color_scale = color_scale_trait()
    camera_projection = tl.CaselessStrEnum(
        CAMERA_PROJECTIONS, default_value="perspective"
    ).tag(sync=True)

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series (equal-length ``x``/``y``/``z``, binary when large) on
        init and reassignment.
        """
        return self._normalize_series(
            proposal["value"], "ScatterPlot3D", ("x", "y", "z")
        )

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
        super().__init__(series=series, **kwargs)
