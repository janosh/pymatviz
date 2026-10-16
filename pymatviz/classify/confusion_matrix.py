"""Confusion matrix plotting functions."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import plotly.graph_objects as go

from pymatviz.utils.plotting import annotated_heatmap


if TYPE_CHECKING:
    from numpy.typing import NDArray


def confusion_matrix(
    conf_mat: Sequence[Sequence[int]] | np.ndarray | None = None,
    *,
    y_true: Sequence[str | int] | np.ndarray | None = None,
    y_pred: Sequence[str | int] | np.ndarray | None = None,
    x_labels: tuple[str, ...] | None = None,
    y_labels: tuple[str, ...] | None = None,
    annotations: Sequence[Sequence[str]]
    | Callable[[int, int, float, float], str]
    | None = None,
    normalize: bool = True,
    colorscale: str = "blues",
    float_fmt: str = "",
    heatmap_kwargs: dict[str, Any] | None = None,
    metrics: dict[str, str | None] | Sequence[str] | set[str] = ("Acc",),
    metrics_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Plot confusion matrix using plotly. Accepts either a pre-computed confusion
    matrix or raw class labels.

    Args:
        conf_mat (np.ndarray | None): Pre-computed confusion matrix. If None, will be
            computed from y_true and y_pred
        y_true (np.ndarray | None): True class labels
        y_pred (np.ndarray | None): Predicted class labels
        x_labels (tuple[str, ...] | None): Labels for x-axis (predicted). If None and
            using raw labels, will be inferred from unique values in y_true and y_pred
        y_labels (tuple[str, ...] | None): Labels for y-axis (true). If None, same as
            x_labels
        annotations (Sequence[Sequence[str]] | Callable | None): Custom cell
            annotations (indexed like the confusion matrix: rows = true classes,
            columns = predicted) or a callable that takes
            (count, total, row_pct, col_pct) and returns a string. If None, will
            use pretty-formatted values in conf_mat. The callable receives:
            - count: Raw count for this cell
            - total: Total count across all cells
            - row_pct: Row percentage (count/row_sum)
            - col_pct: Column percentage (count/col_sum)
        normalize (bool): Whether to normalize values to percentages that sum to 100%
        colorscale (str): Plotly colorscale name
        float_fmt (str): Format string for floating point numbers in annotations
        heatmap_kwargs (dict | None): Additional keywords for
            pymatviz.utils.plotting.annotated_heatmap()
        metrics (dict[str, str | None] | Sequence[str]): Which metrics to display and
            their format strings. If not dict, uses float_fmt for all metrics. Defaults
            to ("Acc",). Available metrics: "Acc", "Prec", "Rec", "F1", "MCC"
            (Matthews correlation coefficient). Only "Acc" applies to multi-class
            problems.
        metrics_kwargs (dict | None): Additional keywords for metrics annotation.

    Use fig.layout.(x|y)axis.title = str to set axis titles.

    Returns:
        go.Figure: Plotly figure object
    """
    import sklearn.metrics as skm

    if float_fmt == "":
        float_fmt = ".1%" if normalize else ".0f"

    if conf_mat is None and (y_true is None or y_pred is None):
        raise ValueError("Must provide either conf_mat or both y_true and y_pred")

    # normalize=None so we manually normalize to ensure sum over all cells = 100%
    if y_true is not None and y_pred is not None:
        classes = np.array(sorted({*y_true, *y_pred}))
        conf_mat_arr = skm.confusion_matrix(
            y_true=y_true, y_pred=y_pred, normalize=None, labels=classes
        )
        if x_labels is None:
            x_labels = tuple(map(str, classes))
    else:
        conf_mat_arr = np.array(conf_mat)

    if x_labels is None:
        raise ValueError("Must provide x_labels when passing pre-computed conf_mat")

    y_labels = y_labels or x_labels
    n_classes = len(x_labels)

    # Create hover text (and callable annotations) with absolute counts and percentages
    sample_counts = conf_mat_arr.copy()  # retain unnormalized counts for hover text
    total = sample_counts.sum()
    if normalize:
        conf_mat_arr = conf_mat_arr / conf_mat_arr.sum()
    annotation_func = (
        cast("Callable[[int, int, float, float], str]", annotations)
        if callable(annotations)
        else None
    )
    hover_text: list[list[str]] = []
    callable_annotations: list[list[str]] = []
    for ii, row in enumerate(conf_mat_arr):
        hover_row, anno_row = [], []
        for jj, val in enumerate(row):
            count = int(sample_counts[ii, jj])
            pct = val if normalize else val / total
            row_sum, col_sum = row.sum(), conf_mat_arr[:, jj].sum()
            row_pct = val / row_sum if row_sum > 0 else 0
            col_pct = val / col_sum if col_sum > 0 else 0
            hover_row.append(
                f"True: {y_labels[ii]}<br>"
                f"Pred: {x_labels[jj]}<br>"
                f"Count: {count:,}<br>"
                f"Percent: {pct:.1%}<br>"
                f"Row %: {row_pct:.1%}<br>"
                f"Col %: {col_pct:.1%}"
            )
            if annotation_func is not None:
                anno_row.append(annotation_func(count, total, row_pct, col_pct))
        hover_text.append(hover_row)
        callable_annotations.append(anno_row)

    # Split long y-labels with line breaks if needed (>15 chars)
    formatted_labels: dict[str, list[str]] = {"x": [], "y": []}
    for key, labels in (("y", y_labels[::-1]), ("x", x_labels)):
        for label in labels:  # Already reversed for conventional orientation
            if len(label) > 15:
                # Split at space closest to middle
                mid = len(label) // 2
                spaces = [idx for idx, char in enumerate(label) if char == " "]
                if spaces:
                    split_point = min(spaces, key=lambda x: abs(x - mid))
                    label = f"{label[:split_point]}<br>{label[split_point + 1 :]}"  # noqa: PLW2901
            formatted_labels[key] += [label]

    fmt_tile_vals = np.array(
        [[f"{val:{float_fmt}}" for val in row] for row in conf_mat_arr]
    )

    # Process annotations into a numpy array for the heatmap
    processed_annotations: NDArray[np.str_]
    if annotations is None:
        processed_annotations = fmt_tile_vals
    elif callable(annotations):  # callable was applied to each cell above
        processed_annotations = np.array(callable_annotations)
    else:  # When custom annotations provided, append percentage values
        anno_arr = np.asarray(annotations)
        if anno_arr.shape != fmt_tile_vals.shape:
            raise ValueError(
                f"Custom annotations shape {anno_arr.shape} does not match confusion "
                f"matrix shape {fmt_tile_vals.shape}. Annotations are indexed like "
                f"the confusion matrix: rows = true classes, columns = predicted."
            )
        processed_annotations = np.char.add(anno_arr, "<br>")
        processed_annotations = np.char.add(processed_annotations, fmt_tile_vals)

    # Reverse true-class rows for Plotly's bottom-left origin.
    heatmap_defaults: dict[str, Any] = dict(
        z=np.flipud(conf_mat_arr),
        x=formatted_labels["x"],
        y=formatted_labels["y"],
        annotation_text=np.flipud(processed_annotations),
        colorscale=colorscale,
        xgap=7,
        ygap=7,
        hoverongaps=False,
        hoverinfo="text",
        text=np.flipud(hover_text),
    )
    fig = annotated_heatmap(**heatmap_defaults | (heatmap_kwargs or {}))

    if isinstance(metrics, list | tuple | set):
        metrics_dict: dict[Any, str | None] = dict.fromkeys(metrics)
    elif isinstance(metrics, dict):
        metrics_dict = metrics
    else:
        raise TypeError(f"Unknown {metrics=}")
    for metric in metrics_dict:
        if not isinstance(metric, str):
            raise TypeError(f"Metric keys must be strings, got {type(metric).__name__}")

    available_metrics = {"Acc": conf_mat_arr.diagonal().sum() / conf_mat_arr.sum()}
    if n_classes == 2:  # binary classification metrics
        tn, fp, fn, tp = conf_mat_arr.ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        available_metrics |= {
            "Prec": precision,
            "Rec": recall,
            "F1": 2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0,
            "MCC": (tp * tn - fp * fn) / np.sqrt(mcc_denom) if mcc_denom > 0 else 0,
        }

    metrics_text = []
    for metric, fmt in cast("dict[str, str | None]", metrics_dict).items():
        if metric not in available_metrics:
            raise ValueError(
                f"Unknown {metric=}. Available: {', '.join(available_metrics)}"
            )
        metric_fmt = float_fmt if fmt is None else fmt
        metrics_text += [f"{metric}={available_metrics[metric]:{metric_fmt}}"]

    if metrics_text:  # only add annotation if there are metrics to show
        metrics_defaults = dict(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            text=", ".join(metrics_text),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font_size=22,
        ) | (metrics_kwargs or {})
        fig.add_annotation(**metrics_defaults)
        y_pos = metrics_defaults["y"]
        if isinstance(y_pos, (int, float)) and y_pos >= 1:
            fig.layout.margin.t = 60
        elif isinstance(y_pos, (int, float)) and y_pos <= 0:
            fig.layout.margin.b = 60

    # Update axes formatting
    axes_kwargs = dict(
        showline=False,
        showgrid=False,
        constrain="domain",  # ensure nearly square overall figure
    )
    fig.layout.font.size = 18
    fig.layout.paper_bgcolor = fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.xaxis.update(dict(title="Predicted") | axes_kwargs)
    fig.layout.yaxis.update(
        dict(
            title="Actual",
            scaleanchor="x",  # ensure square tiles by forcing same scale as x-axis
            tickangle=-90,  # Rotate labels 90 degrees
        )
        | axes_kwargs,
    )

    return fig
