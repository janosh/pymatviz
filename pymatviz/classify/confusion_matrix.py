"""Confusion matrix plotting functions."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go


def confusion_matrix(
    conf_mat: Sequence[Sequence[int]] | np.ndarray | None = None,
    *,
    y_true: Sequence[str | int] | None = None,
    y_pred: Sequence[str | int] | None = None,
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
            annotations or a callable that takes (count, total, pct, row_pct, col_pct)
            and returns a string. If None, will use pretty-formatted values in conf_mat.
            The callable receives:
            - count: Raw count for this cell
            - total: Total count across all cells
            - row_pct: Row percentage (count/row_sum)
            - col_pct: Column percentage (count/col_sum)
        normalize (bool): Whether to normalize values to percentages that sum to 100%
        colorscale (str): Plotly colorscale name
        float_fmt (str): Format string for floating point numbers in annotations
        heatmap_kwargs (dict | None): Additional keywords for
            plotly.figure_factory.create_annotated_heatmap()
        metrics (dict[str, str | None] | Sequence[str]): Which metrics to display and
            their format strings. If not dict, uses float_fmt for all metrics. Defaults
            to ("Acc", "MCC"). Available metrics: "Acc", "Prec", "Rec", "F1", "MCC"
            (Matthews correlation coefficient). Only "Acc" applies to multi-class
            problems.
        metrics_kwargs (dict | None): Additional keywords for metrics annotation.

    Use fig.layout.(x|y)axis.title = str to set axis titles.

    Returns:
        go.Figure: Plotly figure object
    """
    import plotly.figure_factory as ff
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

    # Create hover text with absolute counts and percentages
    sample_counts = conf_mat_arr.copy()  # retain unnormalized counts for hover text
    if normalize:
        conf_mat_arr = conf_mat_arr / conf_mat_arr.sum()
    hover_text: list[list[str]] = []
    for ii, row in enumerate(conf_mat_arr):
        hover_row = []
        for jj, val in enumerate(row):
            count = int(sample_counts[ii, jj])
            pct = val if normalize else val / sample_counts.sum()
            hover_row += [
                f"True: {y_labels[ii]}<br>"
                f"Pred: {x_labels[jj]}<br>"
                f"Count: {count:,}<br>"
                f"Percent: {pct:.1%}<br>"
                f"Row %: {val / row.sum():.1%}<br>"
                f"Col %: {val / conf_mat_arr[:, jj].sum():.1%}"
            ]
        hover_text += [hover_row]

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
    ).T
    if annotations is None:
        annotations = fmt_tile_vals
    elif callable(annotations):  # If annotations is a callable, apply it to each cell
        total = sample_counts.sum()
        anno_matrix = []
        for ii in range(len(conf_mat_arr)):
            row = []
            for jj in range(len(conf_mat_arr[ii])):
                count = int(sample_counts[ii, jj])
                pct = conf_mat_arr[ii, jj]
                row_pct = (
                    conf_mat_arr[ii, jj] / conf_mat_arr[ii].sum()
                    if conf_mat_arr[ii].sum() > 0
                    else 0
                )
                col_pct = (
                    conf_mat_arr[ii, jj] / conf_mat_arr[:, jj].sum()
                    if conf_mat_arr[:, jj].sum() > 0
                    else 0
                )
                row += [annotations(count, total, row_pct, col_pct)]
            anno_matrix += [row]
        annotations = np.array(anno_matrix).T
    else:  # When custom annotations provided, append percentage values
        annotations = np.char.add(annotations, "<br>")
        annotations = np.char.add(annotations, fmt_tile_vals)

    heatmap_defaults = dict(
        z=np.rot90(conf_mat_arr),
        x=formatted_labels["x"],
        y=formatted_labels["y"],
        annotation_text=np.rot90(np.array(annotations).T),
        colorscale=colorscale,
        xgap=7,
        ygap=7,
        hoverongaps=False,
        hoverinfo="text",
        text=np.rot90(hover_text),
    )
    fig = ff.create_annotated_heatmap(**heatmap_defaults | (heatmap_kwargs or {}))

    # Calculate accuracy and other metrics
    acc = conf_mat_arr.diagonal().sum() / conf_mat_arr.sum()

    # Handle metrics parameter
    if isinstance(metrics, list | tuple | set):
        metrics_dict: dict[str, str | None] = dict.fromkeys(metrics)
    elif isinstance(metrics, dict):
        metrics_dict = metrics
    else:
        raise TypeError(f"Unknown {metrics=}")

    # Add metrics annotation
    metrics_text = []
    available_metrics = {"Acc": lambda: acc}
    if n_classes == 2:  # binary classification metrics
        tn, fp, fn, tp = conf_mat_arr.ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        available_metrics |= {
            "Prec": lambda: precision,
            "Rec": lambda: recall,
            "F1": lambda: 2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0,
            "MCC": lambda: (tp * tn - fp * fn)
            / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if n_classes == 2
            else None,
        }

    for metric, fmt in metrics_dict.items():
        if metric not in available_metrics:
            raise ValueError(
                f"Unknown {metric=}. Available: {', '.join(available_metrics)}"
            )

        # Skip binary classification metrics for multi-class problems
        if n_classes > 2 and metric != "Acc":
            print(f"Warning: skipping binary {metric=} for multi-class problem")  # noqa: T201
            continue

        metric_val = available_metrics[metric]()
        if metric_val is None:  # skip metrics that returned None
            continue

        # use float_fmt if no specific format given
        f_fmt = float_fmt if fmt is None else fmt
        metrics_text += [f"{metric}={metric_val:{f_fmt}}"]

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
        if metrics_defaults.get("y", 0) >= 1:  # type: ignore[operator]
            fig.layout.margin.t = 60
        if metrics_defaults.get("y", 0) <= 0:  # type: ignore[operator]
            fig.layout.margin.b = 60

    # Update axes formatting
    axes_kwargs = dict(
        showline=False,
        showgrid=False,
        constrain="domain",  # ensure nearly square overall figure
    )
    fig.layout.font.size = 18
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.xaxis.update(dict(title="Actual") | axes_kwargs)
    fig.layout.yaxis.update(
        dict(
            title="Predicted",
            scaleanchor="x",  # ensure square tiles by forcing same scale as x-axis
            tickangle=-90,  # Rotate labels 90 degrees
        )
        | axes_kwargs,
    )

    return fig
