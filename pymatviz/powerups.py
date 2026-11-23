"""Powerups for MatterViz scatter widgets: identity and best-fit lines, metric
annotations and ECDF lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from pymatviz.utils import align_series
from pymatviz.widgets._binary import series_column
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

# series indices to read x/y data from, None for all series
type SeriesSelector = int | np.integer | Sequence[int] | None

PARITY_WIDGETS = (ScatterPlotWidget, BinnedScatterPlotWidget)


PRETTY_METRIC_LABELS = {"R2": "R²", "R2_adj": "R²_adj"}


def _r2(xs: np.ndarray, ys: np.ndarray) -> float:
    """Numpy ``r2_score``: -inf or nan (without RuntimeWarning) for constant targets."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1 - ((xs - ys) ** 2).sum() / ((xs - xs.mean()) ** 2).sum())


# numpy equivalents of sklearn's r2_score and mean_absolute_percentage_error (keeps
# sklearn out of `import pymatviz`)
BUILTIN_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "MAE": lambda xs, ys: np.abs(xs - ys).mean(),
    "RMSE": lambda xs, ys: ((xs - ys) ** 2).mean() ** 0.5,
    "MSE": lambda xs, ys: ((xs - ys) ** 2).mean(),
    "MAPE": lambda xs, ys: (
        np.abs(xs - ys) / np.maximum(np.abs(xs), np.finfo(float).eps)
    ).mean(),
    "R2": _r2,
    # adjusted R2 is undefined for n <= 2 samples (zero/negative denominator)
    "R2_adj": lambda xs, ys: (
        1 - (1 - _r2(xs, ys)) * (len(xs) - 1) / (len(xs) - 2)
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
    """Explicit xs/ys as float arrays (pandas Series aligned by index), else the
    concatenated selected series data.
    """
    xs, ys = align_series(xs, ys)
    xs_arr, ys_arr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xs_arr.size or ys_arr.size:
        if xs_arr.ndim != 1 or xs_arr.shape != ys_arr.shape:
            raise ValueError(
                f"xs and ys must be 1D with equal length, got {xs_arr.shape=}, "
                f"{ys_arr.shape=}"
            )
        return xs_arr, ys_arr
    picked = _selected_series(widget, series)
    xs_arr, ys_arr = (
        np.concatenate([series_column(srs, axis) for srs in picked]).astype(float)
        for axis in "xy"
    )
    return xs_arr, ys_arr


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
    fmt: str = ".3g",
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
            by name. A mapping of labels to either precomputed numbers or metric
            names (computed), e.g. {"Mean abs. error": "MAE", "Custom":
            1.23}. Defaults to ("MAE", "R2").
        prefix (str): Text before the metrics, e.g. a title line ending in a newline.
        suffix (str): Text after the metrics.
        fmt (str): Format spec for metric values, precomputed ones included.
            Defaults to ".3g" (".3" would show e.g. 123 as 1.23e+02).

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
            import sklearn.metrics

            if (func := getattr(sklearn.metrics, name, None)) is None:
                raise ValueError(f"Unrecognized metric {name!r}")
            funcs[name] = func

    xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
    finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    xs_arr, ys_arr = xs_arr[finite], ys_arr[finite]
    computes_metrics = any(isinstance(val, str) for val in label_to_metric.values())
    if computes_metrics and not finite.any():
        raise ValueError(f"Metrics need finite (x, y) pairs, got none of {finite.size}")
    lines = []
    for label, metric in label_to_metric.items():
        value = funcs[metric](xs_arr, ys_arr) if isinstance(metric, str) else metric
        lines.append(f"{PRETTY_METRIC_LABELS.get(label, label)} = {float(value):{fmt}}")
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


def _log_axes(
    widget: ScatterPlotWidget | BinnedScatterPlotWidget,
) -> tuple[bool, bool]:
    """Whether the widget's x and y axes are log-scaled.

    Raises:
        ValueError: For scale types other than linear and log (e.g. arcsinh, time),
            where a least-squares line has no simple straight-segment form.
    """
    log_flags = []
    for axis in ("x", "y"):
        scale_type = (getattr(widget, f"{axis}_axis") or {}).get("scale_type")
        name = scale_type.get("type") if isinstance(scale_type, Mapping) else scale_type
        if name not in (None, "linear", "log"):
            raise ValueError(
                f"Best fit line supports linear and log axes, got {axis}_axis "
                f"scale_type={scale_type!r}"
            )
        log_flags.append(name == "log")
    return log_flags[0], log_flags[1]


def _fit_mask(
    xs_arr: np.ndarray, ys_arr: np.ndarray, *, log_x: bool, log_y: bool
) -> np.ndarray:
    """Points a best-fit line can use: finite, and positive on log axes."""
    keep = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    if log_x:
        keep &= xs_arr > 0
    if log_y:
        keep &= ys_arr > 0
    return keep


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

    The fit is done in the space the widget's axes display (log10 of values on log
    axes, read from x_axis/y_axis scale_type at call time), so the fit is a straight
    line on screen: y = ax + b on linear axes, a power law log10(y) = a log10(x) + b
    on log-log axes, and exponential/logarithmic fits on semi-log axes. Points with
    non-positive values on a log axis are dropped since they can't be displayed.

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
            "LS fit: y = ax + b" (with log10(x) and/or log10(y) on log axes).
            Defaults to True.

    Raises:
        ValueError: If fewer than 2 finite points (positive on log axes) remain
            to fit, or an axis has a scale_type other than linear or log.

    Returns:
        ScatterPlotWidget | BinnedScatterPlotWidget: The widget.
    """
    _check_widget(widget)
    log_x, log_y = _log_axes(widget)
    xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
    keep = _fit_mask(xs_arr, ys_arr, log_x=log_x, log_y=log_y)
    if (n_keep := int(keep.sum())) < 2:
        positive = " (positive on log axes)" if log_x or log_y else ""
        raise ValueError(
            f"Best fit line needs >= 2 finite points{positive}, got {n_keep} of "
            f"{len(xs_arr)}"
        )
    xs_arr, ys_arr = xs_arr[keep], ys_arr[keep]
    if np.ptp(xs_arr) == 0:
        raise ValueError(f"Best fit line needs varying x values, all are {xs_arr[0]}")
    # fit in display space, where the fit is straight and so is a segment
    fit_xs = np.log10(xs_arr) if log_x else xs_arr
    fit_ys = np.log10(ys_arr) if log_y else ys_arr
    slope, intercept = np.polyfit(fit_xs, fit_ys, 1)
    x_min, x_max = float(xs_arr.min()), float(xs_arr.max())

    def fit_y(x_val: float) -> float:
        """Fitted y at data-space x, mapped back from display space."""
        y_val = slope * (np.log10(x_val) if log_x else x_val) + intercept
        return float(10**y_val if log_y else y_val)

    color = "currentColor"
    no_explicit_data = not np.size(xs) and not np.size(ys)
    if no_explicit_data and isinstance(series, (int, np.integer)):
        color = (widget.series or [])[series].get("color", color)
    line_style = {"color": color, "width": 2, "dash": "8 4"} | (style or {})
    line: dict[str, Any] = {
        "type": "segment",
        "p1": [x_min, fit_y(x_min)],
        "p2": [x_max, fit_y(x_max)],
        "style": line_style,
        "label": "LS fit",
        "metadata": {
            "slope": float(slope),
            "intercept": float(intercept),
            "log_x": log_x,
            "log_y": log_y,
        },
    }
    if annotate_params:
        sign = "+" if intercept >= 0 else "-"
        x_term = "·log10(x)" if log_x else "x"
        y_term = "log10(y)" if log_y else "y"
        text = f"LS fit: {y_term} = {slope:.2g}{x_term} {sign} {abs(intercept):.2g}"
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
            passed as add_best_fit_line() kwargs. None (default) adds it if R² > 0.3
            and the axes leave >= 2 fittable points with varying x.
        stats (bool | dict): Whether to show metrics. A dict is passed as
            annotate_metrics() kwargs. Defaults to True. Dicts for best_fit_line and
            stats may override xs, ys and series.

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
    # True enables a powerup with default kwargs, any dict (even {}) with its kwargs
    identity_kwargs, fit_kwargs, stats_kwargs = (
        {} if opt is True else opt if isinstance(opt, dict) else None
        for opt in (identity_line, best_fit_line, stats)
    )
    if identity_kwargs is not None:
        add_identity_line(widget, **identity_kwargs)
    if fit_kwargs is None and stats_kwargs is None and best_fit_line is not None:
        return widget

    data_kwargs: dict[str, Any] = {"xs": xs, "ys": ys, "series": series}
    if best_fit_line is None:
        xs_arr, ys_arr = _resolve_xy(widget, xs, ys, series)
        finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
        auto_fit = bool(
            finite.sum() > 1
            and BUILTIN_METRICS["R2"](xs_arr[finite], ys_arr[finite]) > 0.3
        )
        if auto_fit:  # skip, rather than fail, fits the axes leave ill-posed
            log_x, log_y = _log_axes(widget)
            fit_xs = xs_arr[_fit_mask(xs_arr, ys_arr, log_x=log_x, log_y=log_y)]
            auto_fit = fit_xs.size > 1 and bool(np.ptp(fit_xs) > 0)
        fit_kwargs = {} if auto_fit else None
    if fit_kwargs is not None:
        add_best_fit_line(widget, **data_kwargs | fit_kwargs)
    if stats_kwargs is not None:
        annotate_metrics(widget, **data_kwargs | stats_kwargs)
    return widget


def add_ecdf_line(
    widget: ScatterPlotWidget,
    values: ArrayLike = (),
    *,
    series: SeriesSelector = None,
    label: str = "Cumulative",
    line_style: dict[str, Any] | None = None,
) -> ScatterPlotWidget:
    """Add empirical cumulative distribution function (ECDF) step lines on a right
    y2 axis spanning [0, 1] (unless the widget already sets y2_range).

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

    Raises:
        ValueError: If the y2 axis already holds non-ECDF data (y values outside
            [0, 1], e.g. a temperature), whose range the ECDF would clip.
    """
    _check_widget(widget, (ScatterPlotWidget,))
    # earlier ECDFs (any label) keep y2 within [0, 1], other y2 data is foreign
    for srs in widget.series or []:
        if srs.get("y_axis") != "y2":
            continue
        y_vals = np.asarray(series_column(srs, "y"), dtype=float)
        if ((y_vals < 0) | (y_vals > 1)).any():
            y2_label = (widget.y2_axis or {}).get("label")
            raise ValueError(
                f"widget's y2 axis already plots {srs.get('label') or y2_label!r} "
                f"with values outside [0, 1], so it can't also hold the ECDF {label!r}"
            )
    if np.size(values):
        sources: list[tuple[ArrayLike, str, str | None]] = [(values, label, None)]
    else:
        picked = _selected_series(widget, series)
        sources = [
            (
                series_column(srs, "x"),
                f"{label} ({srs['label']})"
                if len(picked) > 1 and srs.get("label")
                else label,
                srs.get("color"),
            )
            for srs in picked
        ]
    new_series = []
    for source, series_label, color in sources:
        xs = np.sort(np.asarray(source, dtype=float).ravel())
        xs = xs[np.isfinite(xs)]
        if not xs.size:
            raise ValueError("ECDF needs at least one finite value")
        fractions = np.arange(1, xs.size + 1) / xs.size  # share of values <= each x
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
    # the widget's series validator normalizes them (binary for large ECDFs)
    widget.series = [*(widget.series or []), *new_series]
    widget.y2_axis = {"label": label} | (widget.y2_axis or {})
    if widget.y2_range is None:
        widget.y2_range = [0, 1]
    return widget
