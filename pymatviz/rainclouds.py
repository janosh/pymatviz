"""Raincloud plots as MatterViz box plot widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.box_plot import BoxPlotWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal


def rainclouds(
    data: Mapping[
        str, Sequence[float] | np.ndarray | pd.Series | tuple[pd.DataFrame, str]
    ],
    *,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    alpha: float = 0.7,
    width_viol: float = 0.6,
    width_box: float = 0.1,
    point_size: float = 2,
    bw: float = 0.2,
    show_violin: bool = True,
    show_box: bool = True,
    show_points: bool = True,
    **kwargs: Any,
) -> BoxPlotWidget:
    """Raincloud plot (half violin, box and jittered samples) per dataset.

    This plot type was proposed in https://wellcomeopenresearch.org/articles/4-63/v2.
    Each dataset is drawn as a one-sided violin (the cloud), a box plot (the
    umbrella) and jittered samples on the opposite side (the rain). Quartiles,
    whiskers and the KDE are computed in the frontend from the raw samples. Each
    violin is scaled to its own maximum density.

    Args:
        data (dict[str, Sequence[float] | tuple[pd.DataFrame, str]]): Map from labels
            to either sequences of floats or (DataFrame, column name) tuples.
        orientation ("horizontal" | "vertical", optional): "horizontal" draws
            distributions along the x-axis stacked vertically, "vertical" along the
            y-axis. Invalid values raise ``traitlets.TraitError``. Defaults to
            "horizontal".
        alpha (float, optional): Opacity of the violins. Defaults to 0.7.
        width_viol (float, optional): Violin width as fraction of each category slot.
            Defaults to 0.6.
        width_box (float, optional): Box width as fraction of each category slot.
            Defaults to 0.1.
        point_size (float, optional): Radius of the rain points in px. Defaults to 2.
        bw (float, optional): KDE bandwidth as a multiple of each dataset's sample
            standard deviation (scipy's ``gaussian_kde(bw_method=bw)`` convention).
            Defaults to 0.2.
        show_violin (bool, optional): Whether to show the violin. Defaults to True.
        show_box (bool, optional): Whether to show the box plot. Defaults to True.
        show_points (bool, optional): Whether to show the samples. Defaults to True.
        **kwargs: BoxPlotWidget traits, e.g. ``points`` (``"swarm"``),
            ``whisker_mode``, ``show_mean``, ``x_axis``, ``y_axis``, ``style``.
            ``violin_style`` and ``sample_style`` are merged over the defaults.

    Returns:
        BoxPlotWidget: one series per dataset.

    Raises:
        ValueError: If data is empty, neither violin nor box is shown, or a dataset
            is empty or holds non-finite values.
    """
    if not data:
        raise ValueError("rainclouds needs at least one dataset")
    if not (show_violin or show_box):
        raise ValueError("rainclouds needs show_violin or show_box (or both)")

    series: list[dict[str, Any]] = []
    for label, data_itm in data.items():
        match data_itm:
            case (pd.DataFrame() as df_i, col):
                values = np.asarray(df_i[col], dtype=float)
            case _:
                values = np.asarray(data_itm, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(
                f"rainclouds data for {label!r} must be a non-empty 1D array, got "
                f"shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"rainclouds data for {label!r} holds non-finite values")
        entry: dict[str, Any] = {
            "y": values,
            "label": str(label),
            "box_width": width_box,
        }
        # scipy's scalar bw_method scales the sample std; MatterViz takes value units
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        if show_violin and std > 0:
            entry["bandwidth"] = bw * std
        series.append(entry)

    kind = (
        "violin+box" if show_violin and show_box else "violin" if show_violin else "box"
    )
    defaults: dict[str, Any] = {
        "orientation": orientation,
        "kind": kind,
        "side": "positive",
        "point_side": "negative",  # rain falls opposite the cloud
        "points": "strip" if show_points else "none",
        "violin_width": width_viol,
        "violin_style": {"opacity": alpha},
        "sample_style": {"radius": point_size, "opacity": 0.5},
        # samples are drawn as rain, not as separate outlier markers
        "show_outliers": not show_points,
    }
    return BoxPlotWidget(series, **with_defaults(defaults, kwargs))
