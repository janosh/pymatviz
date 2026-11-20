"""Confusion matrix heatmap."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget


def _classification_metrics(conf_mat: np.ndarray) -> dict[str, float]:
    """Accuracy, plus precision, recall, F1 and MCC for binary confusion matrices
    (rows = true, columns = predicted, class 1 positive).
    """
    metrics = {"Acc": conf_mat.diagonal().sum() / conf_mat.sum()}
    if conf_mat.shape == (2, 2):
        tn, fp, fn, tp = conf_mat.ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        metrics |= {
            "Prec": precision,
            "Rec": recall,
            "F1": 2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0,
            "MCC": (tp * tn - fp * fn) / np.sqrt(mcc_denom) if mcc_denom > 0 else 0,
        }
    return {key: float(val) for key, val in metrics.items()}


def confusion_matrix(
    conf_mat: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    y_true: Sequence[str | int] | np.ndarray | None = None,
    y_pred: Sequence[str | int] | np.ndarray | None = None,
    x_labels: Sequence[str] | np.ndarray | None = None,
    y_labels: Sequence[str] | np.ndarray | None = None,
    normalize: bool = True,
    float_fmt: str = "",
    metrics: Mapping[str, str | None] | Sequence[str] = ("Acc",),
    **kwargs: Any,
) -> HeatmapMatrixWidget:
    """Plot a confusion matrix from a pre-computed matrix or raw class labels.

    Rows are true classes (y-axis, "Actual"), columns predicted classes (x-axis,
    "Predicted"). Cells show their (optionally normalized) values; requested metrics
    are appended to the x-axis title.

    Args:
        conf_mat (array | None): Pre-computed confusion matrix (rows = true classes).
            If None, computed from y_true and y_pred.
        y_true (array | None): True class labels.
        y_pred (array | None): Predicted class labels.
        x_labels (Sequence[str] | None): Predicted class labels. Inferred from the
            sorted unique values of y_true and y_pred when passing raw labels.
        y_labels (Sequence[str] | None): True class labels. Defaults to x_labels.
        normalize (bool): Normalize values to fractions of all samples (shown as
            percentages). Defaults to True.
        float_fmt (str): format_num spec for cell values. Defaults to ".1%" when
            normalizing, else ".0f".
        metrics (Mapping[str, str | None] | Sequence[str]): Metrics to show, as names
            or a {name: format spec or None for ".1%"} mapping. Available: "Acc"
            and, for 2 classes, "Prec", "Rec", "F1" and "MCC" (Matthews correlation
            coefficient), taking the second label as the positive class.
            Defaults to ("Acc",).
        **kwargs: Passed to HeatmapMatrixWidget, e.g. color_scale (default
            "interpolateBlues"), color_bar (default None: hidden), tile_size or
            x_axis/y_axis.

    Returns:
        HeatmapMatrixWidget: The confusion matrix heatmap.
    """
    import sklearn.metrics as skm

    float_fmt = float_fmt or (".1%" if normalize else ".0f")
    if (conf_mat is None) == (y_true is None or y_pred is None) or (
        (y_true is None) != (y_pred is None)
    ):
        raise ValueError("Must provide either conf_mat or both y_true and y_pred")
    if y_true is not None and y_pred is not None:
        classes = sorted({*y_true, *y_pred})
        if x_labels is not None:  # order classes like x_labels (matched as strings)
            class_of = {str(cls): cls for cls in classes}
            if sorted(class_of) != sorted(map(str, x_labels)):
                raise ValueError(
                    f"{x_labels=} must name each class in y_true/y_pred: {classes}"
                )
            classes = [class_of[str(label)] for label in x_labels]
        if y_labels is not None and list(map(str, y_labels)) != list(map(str, classes)):
            raise ValueError("y_labels must equal x_labels when passing raw labels")
        conf_mat_arr = skm.confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=classes
        )
        x_labels = tuple(map(str, classes))
    else:
        conf_mat_arr = np.asarray(conf_mat, dtype=float)
    if x_labels is None:
        raise ValueError("Must provide x_labels when passing pre-computed conf_mat")
    if y_labels is None:
        y_labels = x_labels
    if conf_mat_arr.shape != (len(y_labels), len(x_labels)):
        raise ValueError(
            f"conf_mat shape {conf_mat_arr.shape} doesn't match "
            f"{len(y_labels)} y_labels x {len(x_labels)} x_labels"
        )
    if conf_mat_arr.sum() <= 0:
        raise ValueError(f"conf_mat must contain samples, got {conf_mat_arr.tolist()}")
    if normalize:
        conf_mat_arr = conf_mat_arr / conf_mat_arr.sum()

    metric_fmts: Mapping[str, str | None] = (
        metrics if isinstance(metrics, Mapping) else dict.fromkeys(metrics)
    )
    available = _classification_metrics(conf_mat_arr)
    if unknown := set(metric_fmts) - set(available):
        raise ValueError(
            f"Unknown metrics {sorted(unknown)}. Available: {', '.join(available)}"
        )
    metrics_text = ", ".join(
        f"{name}={available[name]:{fmt or '.1%'}}" for name, fmt in metric_fmts.items()
    )
    x_title = f"Predicted ({metrics_text})" if metrics_text else "Predicted"
    return HeatmapMatrixWidget(
        x_items=list(x_labels),
        y_items=list(y_labels),
        values=conf_mat_arr.tolist(),
        **with_defaults(
            {
                "show_values": float_fmt,
                "color_scale": "interpolateBlues",
                "x_axis": {"label": x_title},
                "y_axis": {"label": "Actual"},
                "gap": "4px",
                "color_bar": None,  # cells print their values
            },
            kwargs,
        ),
    )
