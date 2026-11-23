"""Histograms of distributions and element counts as MatterViz widgets."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymatviz.compositions import count_elements
from pymatviz.enums import ElemCountMode
from pymatviz.utils import str_labels
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.bar_plot import BarPlotWidget, bar_axis_defaults
from pymatviz.widgets.histogram import HistogramWidget


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    from numpy.typing import ArrayLike

    from pymatviz.typing import ElemCountModeLike, ElemValues


def elements_hist(
    values: ElemValues,
    *,
    count_mode: ElemCountModeLike = ElemCountMode.composition,
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
        max_elements (int | None): Display only the top n >= 1 elements by
            prevalence.
        show_values ("percent"|"count"|None): "percent" (default) labels bars with
            their share of the total count (incl. elements dropped by max_elements),
            "count" with the count, None hides labels.
        bar_width (float): Width of bars as fraction of available space. Default = 0.7.
        opacity (float): Bar opacity between 0 and 1. Default = 0.8.
        **kwargs: BarPlotWidget traits, e.g. ``x_axis``, ``y_axis``, ``bar`` (merged
            over the defaults set here), ``orientation``, ``style``.

    Returns:
        BarPlotWidget: one bar per element.

    Raises:
        ValueError: For max_elements < 1, negative element counts or an invalid
            show_values.
    """
    if max_elements is not None and max_elements < 1:  # head(-n) drops the last n
        raise ValueError(f"{max_elements=} must be a positive integer or None")
    elem_counts = count_elements(values, count_mode)
    if len(negative := elem_counts[elem_counts < 0]):
        raise ValueError(f"element counts must be >= 0, got {negative.to_dict()}")
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
    bar_style = {"opacity": opacity, "stroke_width": 1, "stroke_color": "black"}
    defaults = axes | {"bar": bar_style, "show_legend": False}
    return BarPlotWidget([series], **with_defaults(defaults, kwargs))


# Above this many values in all distributions, histograms bin in Python and send only
# bin counts: in the browser, every raw value is sent along
MAX_BROWSER_HISTOGRAM_VALUES = 1_000_000


def histogram(
    values: ArrayLike | pd.Series | pd.DataFrame | Mapping[Any, ArrayLike],
    *,
    x: str | None = None,
    color: str | None = None,
    sort_groups: Callable[[list[Any]], list[Any]] | None = None,
    bins: int | Sequence[float] | str = 200,
    x_range: tuple[float | None, float | None] | None = None,
    density: bool = False,
    bin_width: float = 1.0,
    log_y: bool = False,
    **kwargs: Any,
) -> HistogramWidget | BarPlotWidget:
    """Histogram of one or more distributions, all on one set of bin edges.

    MatterViz bins the raw values in the browser, where the controls pane changes the
    bin count live. ``density=True``, bin edges or a NumPy binning rule as ``bins``,
    ``bin_width``, ``orientation`` or more than ``MAX_BROWSER_HISTOGRAM_VALUES``
    values bin in Python with NumPy instead and send only the counts, as a bar plot.

    Args:
        values: Values to bin: a sequence, NumPy array or pandas Series, a dict of
            named ones (keys label the legend), or a DataFrame with ``x`` naming
            the column to bin. NaN/inf values raise a ValueError rather than being
            silently dropped (DataFrame rows with NaN are dropped).
        x (str, optional): DataFrame column to bin.
        color (str, optional): DataFrame column to split into one distribution per
            value, in order of first appearance.
        sort_groups (Callable[[list], list], optional): Reorders the distributions
            (legend and draw order): takes the list of group keys (``color`` column
            values or dict keys, in given order) and returns them reordered, e.g.
            ``sorted`` or ``lambda keys: keys[::-1]``. Defaults to None (keep order).
        bins (int | sequence | str, optional): Number of bins, or (in Python) bin
            edges or a NumPy binning rule like "auto". Defaults to 200.
        x_range (tuple, optional): The range of values to include in the histogram.
            A None bound is taken from the data. Defaults to None (whole data range).
        density (bool, optional): Whether to normalize each histogram to unit area.
            Defaults to False.
        bin_width (float, optional): Bar width as a fraction of the distance between
            bin edges. Defaults to 1.0 (bars touch).
        log_y (bool, optional): Whether to log scale the count axis. Defaults to False.
        **kwargs: Widget traits, e.g. ``x_axis``, ``y_axis``, ``bar`` (merged over the
            defaults set here), ``mode``, ``style``, or ``orientation`` (Python).

    Returns:
        HistogramWidget | BarPlotWidget: The histogram, binned in the browser or
            Python (one overlaid bar series per distribution).

    Raises:
        ValueError: For empty or non-finite values, labels sharing a string form
            (1 and "1"), a DataFrame without ``x``, an x_range resolving to
            lo > hi, density in an x_range holding no values of a distribution, or
            sort_groups not returning a reordering of the group keys.
        TypeError: If x or color name columns but values is no DataFrame.
    """
    if isinstance(values, pd.DataFrame):
        if x is None:
            raise ValueError("name the DataFrame column to bin with x=")
        df_used = values[
            list(dict.fromkeys([x, color] if color is not None else [x]))
        ].dropna()
        groups = (
            df_used.groupby(color, sort=False, observed=True)
            if color is not None
            else [(x, df_used)]
        )
        # grouped keys 1 and "1" would share a legend label: str_labels raises
        x_axis_title, data = x, {key: grp[x] for key, grp in groups}
    elif x is not None or color is not None:
        raise TypeError(f"x={x!r} and color={color!r} need a DataFrame")
    else:
        name = getattr(values, "name", None)
        x_axis_title = "Value" if name is None else name
        data = values if isinstance(values, Mapping) else {x_axis_title: values}
    if sort_groups is not None:
        keys = list(data)
        sorted_keys = list(sort_groups(keys))
        # dict keys are unique, so equal length and key set means a permutation
        if len(sorted_keys) != len(keys) or set(sorted_keys) != set(keys):
            raise ValueError(
                f"sort_groups must return a reordering of {keys}, got {sorted_keys}"
            )
        data = {key: data[key] for key in sorted_keys}
    data_arrays = {
        label: np.asarray(vals, dtype=float) for label, vals in str_labels(data).items()
    }
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
    user_range = x_range is not None
    x_min, x_max = (None, None) if x_range is None else x_range
    x_range = (
        np.min(all_values) if x_min is None else x_min,
        np.max(all_values) if x_max is None else x_max,
    )
    if x_range[0] > x_range[1]:  # would bin no values in the browser
        raise ValueError(
            f"x_range={(x_min, x_max)} resolves to {x_range} with None ends taken "
            "from the data, need lo <= hi"
        )

    count_axis = {
        "label": "Density" if density else "Count",
        "scale_type": "log" if log_y else "linear",
    }
    in_browser = (
        not density
        and isinstance(bins, Integral)
        and not isinstance(bins, bool)
        and bin_width == 1
        and "orientation" not in kwargs
        and len(all_values) <= MAX_BROWSER_HISTOGRAM_VALUES
    )
    if in_browser:
        lo, hi = x_range
        defaults = {
            "x_axis": {"label": str(x_axis_title)},
            "y_axis": count_axis,
            "mode": "overlay" if len(data_arrays) > 1 else "single",
            "show_legend": len(data_arrays) > 1,
        }
        if user_range:
            defaults["x_range"] = [lo, hi]
        # like np.histogram, values outside x_range count toward no bin
        series = [
            {"values": vals[(vals >= lo) & (vals <= hi)], "label": label}
            for label, vals in data_arrays.items()
        ]
        return HistogramWidget(
            series, bins=int(bins), **with_defaults(defaults, kwargs)
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
            "label": label,
            "bar_width": bin_width * widths,
        }
        for label, cnt in counts.items()
    ]

    axes = bar_axis_defaults(
        kwargs, category={"label": str(x_axis_title)}, value=count_axis
    )
    defaults = axes | {
        "bar": {"border_radius": 0},
        "mode": "overlay",
        "show_legend": len(series) > 1,
    }
    return BarPlotWidget(series, **with_defaults(defaults, kwargs))
