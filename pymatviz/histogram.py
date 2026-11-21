"""Histograms as MatterViz bar plot widgets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from pymatviz.composition import count_elements
from pymatviz.enums import ElemCountMode
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.bar_plot import BarPlotWidget, bar_axis_defaults


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    import pandas as pd

    from pymatviz.typing import ElemValues


def elements_hist(
    values: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    log_y: bool = False,
    max_elements: int | None = None,
    show_values: Literal["percent", "count"] | None = "percent",
    bar_width: float = 0.7,
    opacity: float = 0.8,
    **kwargs: Any,
) -> BarPlotWidget:
    """Bar chart of element occurrence counts, sorted from most to least common.

    Args:
        values (ElemValues): compositional strings, e.g. ["Fe2O3", "Bi2Te3"], or a
            map/Series from element symbols to precomputed counts.
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See `count_elements` for
            details. Only used when values is a list of composition strings/objects.
        log_y (bool, optional): Whether y-axis is log or linear. Default = False.
        max_elements (int | None): Display only the top n elements by prevalence.
        show_values ("percent"|"count"|None): "percent" (default) labels bars with
            their share of the total count (incl. elements dropped by max_elements),
            "count" with the count, None hides labels.
        bar_width (float): Width of bars as fraction of available space. Default = 0.7.
        opacity (float): Bar opacity between 0 and 1. Default = 0.8.
        **kwargs: BarPlotWidget traits, e.g. ``x_axis``, ``y_axis``, ``bar`` (merged
            over the defaults set here), ``orientation``, ``style``.

    Returns:
        BarPlotWidget: one bar per element.
    """
    elem_counts = count_elements(values, count_mode)
    non_zero = elem_counts[elem_counts > 0].sort_values(ascending=False)
    total_count = non_zero.sum()  # percent labels are shares of all elements
    if max_elements is not None:
        non_zero = non_zero.head(max_elements)

    counts = non_zero.to_numpy(dtype=float)
    series: dict[str, Any] = {
        "x": [str(elem) for elem in non_zero.index],
        "y": counts,
        "label": "Element Count",
        "bar_width": bar_width,
    }
    if show_values == "percent":
        series["labels"] = [f"{val / total_count:.0%}" for val in counts]
    elif show_values == "count":
        series["labels"] = [
            f"{val:,.0f}" if val.is_integer() else f"{val:.3g}" for val in counts
        ]
    elif show_values is not None:
        raise ValueError(f"Invalid {show_values=}, must be 'percent', 'count' or None")

    count_axis = {"label": "Element Count", "scale_type": "log" if log_y else "linear"}
    axes = bar_axis_defaults(kwargs, category={"label": "Element"}, value=count_axis)
    return BarPlotWidget(
        [series],
        **with_defaults(
            axes
            | {
                "bar": {"opacity": opacity, "stroke_width": 1, "stroke_color": "black"},
                "show_legend": False,
            },
            kwargs,
        ),
    )


def binned_histogram(
    values: Sequence[float]
    | np.ndarray
    | pd.Series
    | Mapping[str, Sequence[float] | np.ndarray | pd.Series],
    *,
    bins: int | Sequence[float] | str = 200,
    x_range: tuple[float | None, float | None] | None = None,
    density: bool = False,
    bin_width: float = 1.0,
    log_y: bool = False,
    **kwargs: Any,
) -> BarPlotWidget:
    """Histogram of one or more distributions, binned in Python with NumPy.

    All series share one set of bin edges. Only bin counts reach the frontend (and
    HTML exports), not every input value.

    Args:
        values: Values to bin. Accepts a sequence, NumPy array, pandas Series, or dict
            of named series. Dict keys are used as legend labels. NaN/inf values
            raise a ValueError rather than being silently dropped.
        bins (int | sequence | str, optional): Number of bins, bin edges or a NumPy
            binning rule like "auto". Defaults to 200.
        x_range (tuple, optional): The range of values to include in the histogram.
            A None bound is taken from the data. Defaults to None (whole data range).
        density (bool, optional): Whether to normalize each histogram to unit area.
            Defaults to False.
        bin_width (float, optional): Bar width as a fraction of the distance between
            bin edges. Defaults to 1.0 (bars touch).
        log_y (bool, optional): Whether to log scale the count axis. Defaults to False.
        **kwargs: BarPlotWidget traits, e.g. ``x_axis``, ``y_axis``, ``bar`` (merged
            over the defaults set here), ``orientation``, ``mode``, ``style``.

    Returns:
        BarPlotWidget: one overlaid bar series per distribution.
    """
    x_axis_title = "Value" if (name := getattr(values, "name", None)) is None else name
    data = values if isinstance(values, Mapping) else {x_axis_title: values}
    data_arrays = {label: np.asarray(vals, dtype=float) for label, vals in data.items()}
    if not data_arrays:
        raise ValueError("histogram values must not be empty.")
    for label, vals in data_arrays.items():
        if vals.ndim != 1 or len(vals) == 0:
            raise ValueError(
                f"histogram values for {label!r} must be a non-empty 1D array."
            )
        if not np.isfinite(vals).all():
            n_bad = int((~np.isfinite(vals)).sum())
            raise ValueError(
                f"histogram values for {label!r} contain {n_bad} non-finite "
                "(NaN/inf) entries. Drop or replace them before plotting."
            )

    all_values = np.concatenate(list(data_arrays.values()))
    x_min, x_max = (None, None) if x_range is None else x_range
    x_range = (
        np.min(all_values) if x_min is None else x_min,
        np.max(all_values) if x_max is None else x_max,
    )

    bin_edges = np.histogram_bin_edges(all_values, bins=bins, range=x_range)
    widths = np.diff(bin_edges)
    centers = bin_edges[:-1] + widths / 2

    counts = {
        label: np.histogram(vals, bins=bin_edges)[0]
        for label, vals in data_arrays.items()
    }
    if density and (empty := [label for label, cnt in counts.items() if not cnt.any()]):
        raise ValueError(
            f"density=True is undefined for series {empty} with no values in {x_range=}"
        )
    series = [
        {
            "x": centers,
            "y": cnt / (cnt.sum() * widths) if density else cnt,
            "label": str(label),
            "bar_width": bin_width * widths,
        }
        for label, cnt in counts.items()
    ]

    count_axis = {
        "label": "Density" if density else "Count",
        "scale_type": "log" if log_y else "linear",
    }
    axes = bar_axis_defaults(
        kwargs, category={"label": str(x_axis_title)}, value=count_axis
    )
    return BarPlotWidget(
        series,
        **with_defaults(
            axes
            | {
                "bar": {"border_radius": 0},
                "mode": "overlay",
                "show_legend": len(series) > 1,
            },
            kwargs,
        ),
    )
