"""Powerups for MatterViz scatter widgets: identity and best-fit lines, metric
annotations and ECDF lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import sklearn.metrics
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from pymatviz.widgets._normalize import normalize_plot_series
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

# series indices to read x/y data from, None for all series
type SeriesSelector = int | np.integer | Sequence[int] | None

PARITY_WIDGETS = (ScatterPlotWidget, BinnedScatterPlotWidget)
PRETTY_METRIC_LABELS = {"R2": "R²", "R2_adj": "R²_adj"}
BUILTIN_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "MAE": lambda xs, ys: np.abs(xs - ys).mean(),
    "RMSE": lambda xs, ys: ((xs - ys) ** 2).mean() ** 0.5,
    "MSE": lambda xs, ys: ((xs - ys) ** 2).mean(),
    "MAPE": mean_absolute_percentage_error,
    "R2": r2_score,
    # adjusted R2 is undefined for n <= 2 samples (zero/negative denominator)
    "R2_adj": lambda xs, ys: (
        1 - (1 - r2_score(xs, ys)) * (len(xs) - 1) / (len(xs) - 2)
        if len(xs) > 2
        else float("nan")
    ),
}


def _check_widget(widget: Any, types: tuple[type, ...] = PARITY_WIDGETS) -> None:
    """Raise TypeError unless widget is an instance of one of types."""
    if not isinstance(widget, types):
        names = " | ".join(cls.__name__ for cls in types)
        raise TypeError(f"widget must be {names}, got {type(widget).__name__}")


def _selected_series(
    widget: ScatterPlotWidget | BinnedScatterPlotWidget, series: SeriesSelector
) -> list[dict[str, Any]]:
    """Series dicts picked by index (None: all series except y2 ones, which hold
    ECDF lines from add_ecdf_line rather than parity data).
    """
    all_series = widget.series or []
    if series is None:
        indices = [
            idx for idx, srs in enumerate(all_series) if srs.get("y_axis") != "y2"
        ]
    else:
        indices = [series] if isinstance(series, (int, np.integer)) else list(series)
        if bad := [
            idx
            for idx in indices
            if isinstance(idx, bool) or not 0 <= idx < len(all_series)
        ]:
            raise IndexError(
                f"series indices {bad} out of range for {len(all_series)} series"
            )
    if not indices:
        raise ValueError(f"{type(widget).__name__} has no series to read data from")
    return [all_series[int(idx)] for idx in indices]


def _resolve_xy(
    widget: ScatterPlotWidget | BinnedScatterPlotWidget,
    xs: ArrayLike,
    ys: ArrayLike,
    series: SeriesSelector,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit xs/ys as float arrays, else the concatenated selected series data."""
    xs_arr, ys_arr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xs_arr.size or ys_arr.size:
        if xs_arr.ndim != 1 or xs_arr.shape != ys_arr.shape:
            raise ValueError(
                f"xs and ys must be 1D with equal length, got {xs_arr.shape=}, "
                f"{ys_arr.shape=}"
            )
        return xs_arr, ys_arr
    picked = _selected_series(widget, series)
    return (
        np.concatenate([np.asarray(srs["x"], dtype=float) for srs in picked]),
        np.concatenate([np.asarray(srs["y"], dtype=float) for srs in picked]),
    )


def _append_annotation(
    widget: ScatterPlotWidget | BinnedScatterPlotWidget, text: str
) -> None:
    """Add a line block to the widget's free-space annotation badge."""
    widget.annotation = "\n".join(filter(None, (widget.annotation, text)))


def _add_ref_line(
    widget: ScatterPlotWidget | BinnedScatterPlotWidget, line: dict[str, Any]
) -> None:
    """Append a reference line to a ScatterPlot's ref_lines or a BinnedScatterPlot's
    overlays. BinnedScatterPlot overlay lines render no labels, so a line annotation
    moves into the widget's annotation badge there.
    """
    if isinstance(widget, BinnedScatterPlotWidget):
        line = dict(line)
        if anno := line.pop("annotation", None):
            _append_annotation(widget, anno["text"])
        overlays = dict(widget.overlays or {})
        overlays["ref_lines"] = [*overlays.get("ref_lines", []), line]
        widget.overlays = overlays
    else:
        widget.ref_lines = [*(widget.ref_lines or []), line]


def annotate_metrics[Widget: ScatterPlotWidget | BinnedScatterPlotWidget](
    widget: Widget,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    *,
    series: SeriesSelector = None,
    metrics: Mapping[str, float | str] | Sequence[str] = ("MAE", "R2"),
    prefix: str = "",
    suffix: str = "",
    fmt: str = ".3",
) -> Widget:
    """Show regression metrics of ys vs xs in the widget's annotation badge.

    Args:
        widget (ScatterPlotWidget | BinnedScatterPlotWidget): Widget to annotate.
        xs (array): x values. Defaults to the x data of the selected series.
        ys (array): y values, same shape as xs. Defaults to the selected series' y.
        series (int | Sequence[int] | None): Series to read data from when xs/ys are
            not given. Defaults to all series (concatenated).
        metrics (Mapping[str, float | str] | Sequence[str]): Metrics to show. A
            sequence of metric names (built-in MAE, RMSE, MSE, MAPE, R2, R2_adj or
            any sklearn.metrics function name) computed from xs and ys and labeled
            by name. A mapping of labels to either precomputed numbers (shown as is)
            or metric names (computed), e.g. {"Mean abs. error": "MAE", "Custom":
            1.23}. Defaults to ("MAE", "R2").
        prefix (str): Text before the metrics, e.g. a title line ending in a newline.
        suffix (str): Text after the metrics.
        fmt (str): Format spec for metric values. Defaults to ".3".

    Returns:
        ScatterPlotWidget | BinnedScatterPlotWidget: The annotated widget.
    """
    _check_widget(widget)
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, (Mapping, list, tuple, set)):
        raise TypeError(
            f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
        )
    # label -> precomputed number or name of the metric to compute
    label_to_metric: dict[str, float | str] = (
        {str(label): val for label, val in metrics.items()}
        if isinstance(metrics, Mapping)
        else {str(name): str(name) for name in metrics}
    )
    if bad_vals := {
        label: val
        for label, val in label_to_metric.items()
        if not isinstance(val, (Real, str))
    }:
        raise TypeError(
            f"metrics values must be numbers or metric names, got {bad_vals}"
        )
    funcs = dict(BUILTIN_METRICS)
    for name in {val for val in label_to_metric.values() if isinstance(val, str)}:
        if name not in funcs:
            if (func := getattr(sklearn.metrics, name, None)) is None:
                raise ValueError(f"Unrecognized metric {name!r}")
            funcs[name] = func

    xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
    finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    xs_arr, ys_arr = xs_arr[finite], ys_arr[finite]
    lines = [
        f"{PRETTY_METRIC_LABELS.get(label, label)} = "
        f"{float(funcs[metric](xs_arr, ys_arr) if isinstance(metric, str) else metric):{fmt}}"  # noqa: E501
        for label, metric in label_to_metric.items()
    ]
    _append_annotation(widget, prefix + "\n".join(lines) + suffix)
    return widget


def add_identity_line[Widget: ScatterPlotWidget | BinnedScatterPlotWidget](
    widget: Widget, *, style: dict[str, Any] | None = None
) -> Widget:
    """Add a dashed y = x reference line spanning the whole plot.

    Args:
        widget (ScatterPlotWidget | BinnedScatterPlotWidget): Widget to add it to.
        style (dict, optional): matterviz RefLineStyle overrides (color, width, dash,
            opacity). Defaults to a 1px gray dashed line.

    Returns:
        ScatterPlotWidget | BinnedScatterPlotWidget: The widget.
    """
    _check_widget(widget)
    line_style = {"color": "gray", "width": 1, "dash": "6 4"} | (style or {})
    _add_ref_line(
        widget,
        {
            "type": "diagonal",
            "slope": 1,
            "intercept": 0,
            "style": line_style,
            "label": "y = x",
            "z_index": "below-points",
        },
    )
    return widget


def add_best_fit_line[Widget: ScatterPlotWidget | BinnedScatterPlotWidget](
    widget: Widget,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    *,
    series: SeriesSelector = None,
    style: dict[str, Any] | None = None,
    annotate_params: bool = True,
) -> Widget:
    """Add a least-squares fit line spanning the x range of the fitted data.

    Args:
        widget (ScatterPlotWidget | BinnedScatterPlotWidget): Widget to add it to.
        xs (array): x values to fit. Defaults to the selected series' x data.
        ys (array): y values to fit. Defaults to the selected series' y data.
        series (int | Sequence[int] | None): Series to fit (combined) when xs/ys are
            not given. Defaults to all series. Call once per series index for
            separate fits.
        style (dict, optional): matterviz RefLineStyle overrides (color, width, dash,
            opacity). The color defaults to the explicit "color" of a single fitted
            series (series=int), else the theme text color.
        annotate_params (bool): Label the line with its equation
            "LS fit: y = ax + b". Defaults to True.

    Raises:
        ValueError: If fewer than 2 finite points remain to fit.

    Returns:
        ScatterPlotWidget | BinnedScatterPlotWidget: The widget.
    """
    _check_widget(widget)
    xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
    finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    if (n_finite := int(finite.sum())) < 2:
        raise ValueError(
            f"Best fit line needs >= 2 finite points, got {n_finite} of {len(xs_arr)}"
        )
    xs_arr, ys_arr = xs_arr[finite], ys_arr[finite]
    if np.ptp(xs_arr) == 0:
        raise ValueError(f"Best fit line needs varying x values, all are {xs_arr[0]}")
    slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
    x_min, x_max = float(xs_arr.min()), float(xs_arr.max())

    color = "currentColor"
    no_explicit_data = not np.size(xs) and not np.size(ys)
    if no_explicit_data and isinstance(series, (int, np.integer)):
        color = (widget.series or [])[series].get("color", color)
    line_style = {"color": color, "width": 2, "dash": "8 4"} | (style or {})
    line: dict[str, Any] = {
        "type": "segment",
        "p1": [x_min, float(slope * x_min + intercept)],
        "p2": [x_max, float(slope * x_max + intercept)],
        "style": line_style,
        "label": "LS fit",
        "metadata": {"slope": float(slope), "intercept": float(intercept)},
    }
    if annotate_params:
        sign = "+" if intercept >= 0 else "-"
        text = f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}"
        line["annotation"] = {
            "text": text,
            "position": "end",
            "side": "below",
            "color": line_style["color"],
        }
    _add_ref_line(widget, line)
    return widget


def enhance_parity_plot[Widget: ScatterPlotWidget | BinnedScatterPlotWidget](
    widget: Widget,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    *,
    series: SeriesSelector = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
) -> Widget:
    """Add parity plot powerups: identity line (y = x), least-squares best-fit line
    and regression metrics (MAE, R², ...).

    Args:
        widget (ScatterPlotWidget | BinnedScatterPlotWidget): Widget to enhance.
        xs (array): x values for the fit and metrics. Defaults to the selected series.
        ys (array): y values for the fit and metrics. Defaults to the selected series.
        series (int | Sequence[int] | None): Series to read data from when xs/ys are
            not given. Defaults to all series (combined).
        identity_line (bool | dict): Whether to add y = x. A dict is passed as
            add_identity_line() kwargs. Defaults to True.
        best_fit_line (bool | dict | None): Whether to add a best-fit line. A dict is
            passed as add_best_fit_line() kwargs. None (default) adds it if R² > 0.3.
        stats (bool | dict): Whether to show metrics. A dict is passed as
            annotate_metrics() kwargs. Defaults to True.

    Returns:
        ScatterPlotWidget | BinnedScatterPlotWidget: The enhanced widget.
    """
    _check_widget(widget)
    for name, opt in (
        ("identity_line", identity_line),
        ("best_fit_line", best_fit_line),
        ("stats", stats),
    ):
        if not isinstance(opt, (bool, dict, type(None))):
            raise TypeError(f"{name} must be bool or dict, got {type(opt).__name__}")
    if identity_line:
        add_identity_line(
            widget, **identity_line if isinstance(identity_line, dict) else {}
        )
    if not best_fit_line and not stats and best_fit_line is not None:
        return widget

    xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
    if best_fit_line is None:
        finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
        best_fit_line = (
            finite.sum() > 1 and r2_score(xs_arr[finite], ys_arr[finite]) > 0.3
        )
    if best_fit_line:
        fit_kwargs = best_fit_line if isinstance(best_fit_line, dict) else {}
        add_best_fit_line(widget, xs, ys, series=series, **fit_kwargs)
    if stats:
        stats_kwargs = stats if isinstance(stats, dict) else {}
        annotate_metrics(widget, xs_arr, ys_arr, **stats_kwargs)
    return widget


def _ecdf_xy(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF vertices: sorted finite values and fractions <= each value."""
    sorted_vals = np.sort(np.asarray(values, dtype=float).ravel())
    sorted_vals = sorted_vals[np.isfinite(sorted_vals)]
    if not sorted_vals.size:
        raise ValueError("ECDF needs at least one finite value")
    return sorted_vals, np.arange(1, sorted_vals.size + 1) / sorted_vals.size


def add_ecdf_line(
    widget: ScatterPlotWidget,
    values: ArrayLike = (),
    *,
    series: SeriesSelector = None,
    label: str = "Cumulative",
    line_style: dict[str, Any] | None = None,
) -> ScatterPlotWidget:
    """Add empirical cumulative distribution function (ECDF) step lines on a right
    y2 axis spanning [0, 1].

    Args:
        widget (ScatterPlotWidget): Widget to add the ECDF series to.
        values (array): Values to compute one ECDF from. Defaults to the x data of
            each selected series (one ECDF per series, matching its color).
        series (int | Sequence[int] | None): Series whose x data to use when values
            is not given. Defaults to all series.
        label (str): Legend label and y2 axis title. Per-series ECDFs of several
            series are labeled "{label} ({series label})". Defaults to "Cumulative".
        line_style (dict, optional): matterviz LineStyle overrides (color, width,
            dash) applied to every added ECDF series.

    Returns:
        ScatterPlotWidget: The widget.
    """
    _check_widget(widget, (ScatterPlotWidget,))
    if np.size(values):
        sources: list[tuple[ArrayLike, str, str | None]] = [(values, label, None)]
    else:
        picked = _selected_series(widget, series)
        sources = [
            (
                srs["x"],
                f"{label} ({srs['label']})"
                if len(picked) > 1 and srs.get("label")
                else label,
                srs.get("color"),
            )
            for srs in picked
        ]
    new_series = []
    for source, series_label, color in sources:
        xs, fractions = _ecdf_xy(source)
        # explicit post-step (hv) vertices since matterviz' step curve steps midway
        style = {"width": 2} | ({"color": color} if color else {}) | (line_style or {})
        new_series.append(
            {
                "x": np.repeat(xs, 2)[1:],
                "y": np.repeat(fractions, 2)[:-1],
                "y_axis": "y2",
                "markers": "line",
                "label": series_label,
                "line_style": style,
            }
        )
    new_series = normalize_plot_series(new_series, component_name="ScatterPlot") or []
    widget.series = [*(widget.series or []), *new_series]
    widget.y2_axis = {"label": label} | (widget.y2_axis or {})
    widget.y2_range = [0, 1]
    return widget
