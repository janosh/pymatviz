"""Plot functions for classification tasks: confusion matrices, ROC and
precision-recall curves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pymatviz.utils import df_to_arrays
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


def _classification_metrics(conf_mat: np.ndarray) -> dict[str, float]:
    """Accuracy, plus precision, recall, F1 and MCC for binary confusion matrices
    (rows = true, columns = predicted, class 1 positive).
    """
    conf_mat = np.asarray(conf_mat, dtype=float)  # int64 MCC products overflow
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
    one_y_missing = (y_true is None) != (y_pred is None)
    if (conf_mat is None) == (y_true is None or y_pred is None) or one_y_missing:
        raise ValueError("Must provide either conf_mat or both y_true and y_pred")
    if y_true is not None and y_pred is not None:
        try:
            classes = sorted({*y_true, *y_pred})
        except TypeError as exc:
            label_types = sorted({type(val).__name__ for val in (*y_true, *y_pred)})
            raise ValueError(
                f"y_true and y_pred labels must be mutually sortable (one type), got "
                f"types {label_types}"
            ) from exc
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
    # metrics need true and predicted classes in the same order on both axes: rows
    # naming the same classes as columns are reordered to match, differently named
    # axes (e.g. "Pred stable" vs "True stable") are aligned by position
    x_strs, y_strs = list(map(str, x_labels)), list(map(str, y_labels))
    metrics_mat = conf_mat_arr
    if metric_fmts and x_strs != y_strs and not set(x_strs).isdisjoint(y_strs):
        if sorted(x_strs) != sorted(y_strs) or len(set(x_strs)) != len(x_strs):
            raise ValueError(
                f"metrics need x_labels and y_labels to name the same classes (or "
                f"disjoint ones aligned by position), got {x_strs=} and {y_strs=}"
            )
        metrics_mat = conf_mat_arr[[y_strs.index(label) for label in x_strs]]
    available = _classification_metrics(metrics_mat)
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


type Predictions = ArrayLike | str | Mapping[str, ArrayLike | dict[str, Any]]


def _standardize_input(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
) -> tuple[ArrayLike, dict[str, dict[str, Any]]]:
    """Standardize input into tuple of (targets, {name: {probs_positive,
    **series_kwargs}}).

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **series_kwargs}}
        df (pd.DataFrame | None): Optional DataFrame containing targets and
            probs_positive columns

    Returns:
        tuple[ArrayLike, dict[str, dict[str, Any]]]: targets, curves_dict
    """
    if df is not None:
        if not isinstance(targets, str):
            raise ValueError(
                f"when passing a DataFrame, targets must be a column name, got "
                f"{type(targets).__name__}"
            )
        # column labels (str or int), or a list of them for several models; arrays
        # could misalign with the rows dropped for NaN
        is_label = lambda val: isinstance(val, str | int) and not isinstance(val, bool)
        is_list = isinstance(probs_positive, list)
        is_label_list = is_list and all(map(is_label, probs_positive))
        if not (is_label(probs_positive) or is_label_list):
            raise TypeError(
                "when passing a DataFrame, probs_positive must be a column name or "
                f"list of them, got {type(probs_positive).__name__}. Pass df=None to "
                "use arrays or a dict of predictions."
            )
        targets, probs_positive = df_to_arrays(df, targets, probs_positive)

    curves_dict: dict[str, dict[str, Any]]
    if isinstance(probs_positive, Mapping):
        # array values become {"probs_positive": array} dicts
        curves_dict = {
            name: probs if isinstance(probs, dict) else {"probs_positive": probs}
            for name, probs in probs_positive.items()
        }
        for name, trace_dict in curves_dict.items():
            if "probs_positive" not in trace_dict:
                raise ValueError(
                    f"probs_positive key is required for all classifiers when passing "
                    f"a dict of dicts, but missing for {name}"
                )
    else:
        curves_dict = {"": {"probs_positive": probs_positive}}

    return targets, curves_dict


# SVG stroke-dasharrays cycled over classifiers to tell overlapping curves apart
LINE_DASHES = ("", "6 4", "2 3", "8 3 2 3", "12 4", "2 6")


def _classifier_series(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: pd.DataFrame | None,
    *,
    build_series: Callable[[str, np.ndarray, np.ndarray], tuple[dict[str, Any], float]],
    required_classes: tuple[float, ...],
) -> list[dict[str, Any]]:
    """One line series per classifier, sorted by a score (descending).

    Args:
        targets (ArrayLike | str): Ground truth labels (see roc_curve).
        probs_positive (Predictions): Per-classifier probabilities (see roc_curve).
        df (pd.DataFrame | None): Optional DataFrame holding both columns.
        build_series (Callable): Given (name, targets_no_nan, probs_no_nan), returns
            the series dict (x, y, label, metadata) and the score to sort by.
        required_classes (tuple[float, ...]): Classes (0 negative, 1 positive) each
            classifier's non-NaN rows must contain for its curve to be defined.

    Raises:
        ValueError: If targets hold more than 2 classes, a classifier's probabilities
            don't match targets in shape or its targets lack any of required_classes.
    """
    targets, curves_dict = _standardize_input(targets, probs_positive, df)
    targets = np.asarray(targets, dtype=float)
    if (n_classes := len(np.unique(targets[~np.isnan(targets)]))) > 2:
        raise ValueError(f"targets must be binary, got {n_classes} classes")
    # 1 is the positive class, the other (e.g. 0 or -1) negative, NaN stays NaN
    targets = np.where(np.isnan(targets), np.nan, targets == 1)
    scored_series = []
    for idx, (name, curve_cfg) in enumerate(curves_dict.items()):
        series_kwargs = dict(curve_cfg)  # don't mutate caller-supplied dicts
        curve_probs = np.asarray(series_kwargs.pop("probs_positive"), dtype=float)
        if curve_probs.shape != targets.shape:
            raise ValueError(
                f"Classifier {name or 'probs_positive'!r} has probabilities of shape "
                f"{curve_probs.shape}, expected one per target {targets.shape}"
            )
        no_nan = ~np.isnan(targets) & ~np.isnan(curve_probs)
        present = set(np.unique(targets[no_nan]).tolist())
        if missing := [f"{cls:g}" for cls in required_classes if cls not in present]:
            raise ValueError(
                f"Classifier {name or 'probs_positive'!r} has no targets of class "
                f"{' and '.join(missing)} (after dropping NaN rows, got classes "
                f"{sorted(present)}), so its curve is undefined"
            )
        series, score = build_series(name, targets[no_nan], curve_probs[no_nan])
        # undefined rates (e.g. FPR without negative samples) are NaN: not plottable
        finite = np.isfinite(series["x"]) & np.isfinite(series["y"])
        series["x"], series["y"] = series["x"][finite], series["y"][finite]
        series["metadata"] = series["metadata"][finite]
        line_style = {"width": 2, "dash": LINE_DASHES[idx % len(LINE_DASHES)]}
        series |= {"markers": "line", "line_style": line_style} | series_kwargs
        scored_series.append((score, series))
    scored_series.sort(key=lambda pair: pair[0], reverse=True)
    return [series for _, series in scored_series]


def _point_metadata(**columns: ArrayLike) -> pd.DataFrame:
    """Per-point metadata as columns (one row per curve point). Widgets turn
    non-finite values (e.g. sklearn's inf first ROC threshold) into None, and send
    large curves' columns as binary arrays instead of per-point JSON dicts.
    """
    return pd.DataFrame(
        {key: np.asarray(vals, dtype=float) for key, vals in columns.items()}
    )


def roc_curve(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    *,
    df: pd.DataFrame | None = None,
    no_skill: dict[str, Any] | Literal[False] | None = None,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Plot receiver operating characteristic (ROC) curves, one per classifier,
    sorted by AUC (descending).

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **series_kwargs}}
              with matterviz series overrides like color or line_style
        df (pd.DataFrame | None): Optional DataFrame containing targets and
            probs_positive columns
        no_skill (dict[str, Any] | False): matterviz RefLine overrides (e.g. style,
            annotation, merged one level deep into the defaults) for the no-skill
            diagonal, or False to hide it.
        **kwargs: Passed to ScatterPlotWidget, e.g. x_axis, y_axis, styles.

    Raises:
        ValueError: If a classifier's targets lack either class.

    Returns:
        ScatterPlotWidget: ROC curves with point metadata holding the thresholds.
    """

    def build_series(
        name: str, tgt: np.ndarray, probs: np.ndarray
    ) -> tuple[dict[str, Any], float]:
        import sklearn.metrics as skm

        fpr, tpr, thresholds = skm.roc_curve(tgt, probs)
        roc_auc = float(skm.roc_auc_score(tgt, probs))
        label = f"{name} · AUC={roc_auc:.2f}" if name else f"AUC={roc_auc:.2f}"
        series = {
            "x": fpr,
            "y": tpr,
            "label": label,
            "metadata": _point_metadata(threshold=thresholds),
        }
        return series, roc_auc

    # ROC rates need both classes (FPR is 0/0 without negatives, TPR without positives)
    series = _classifier_series(
        targets,
        probs_positive,
        df,
        build_series=build_series,
        required_classes=(0, 1),
    )
    ref_lines = []
    if no_skill is not False:
        ref_lines.append(
            with_defaults(
                {
                    "type": "segment",
                    "p1": [0, 0],
                    "p2": [1, 1],
                    "style": {"color": "gray", "dash": "6 4"},
                    "label": "No skill",
                    "annotation": {
                        "text": "No skill",
                        "position": "center",
                        "color": "gray",
                    },
                },
                no_skill or {},
            )
        )
    return ScatterPlotWidget(
        series=series,
        **with_defaults(
            {
                "ref_lines": ref_lines,
                "x_axis": {"label": "False Positive Rate"},
                "y_axis": {"label": "True Positive Rate"},
                "x_range": [0, 1.05],
                "y_range": [0, 1.05],
            },
            kwargs,
        ),
    )


def precision_recall_curve(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    *,
    df: pd.DataFrame | None = None,
    no_skill: dict[str, Any] | Literal[False] | None = None,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Plot precision-recall curves, one per classifier, sorted by F1 score
    (descending, predicting positive for probabilities >= 0.5).

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **series_kwargs}}
              with matterviz series overrides like color or line_style
        df (pd.DataFrame | None): Optional DataFrame containing targets and
            probs_positive columns
        no_skill (dict[str, Any] | False): matterviz RefLine overrides (e.g. style,
            annotation, merged one level deep) for the no-skill baselines (one
            horizontal line per distinct positive-class prevalence), or False to
            hide them.
        **kwargs: Passed to ScatterPlotWidget, e.g. x_axis, y_axis, styles.

    Raises:
        ValueError: If a classifier's targets have no positives.

    Returns:
        ScatterPlotWidget: PR curves with point metadata holding thresholds and F1.
    """
    baselines: dict[float, list[str]] = {}

    def build_series(
        name: str, tgt: np.ndarray, probs: np.ndarray
    ) -> tuple[dict[str, Any], float]:
        import sklearn.metrics as skm

        # drop thresholds that don't change the plotted step curve (like roc_curve
        # does by default): 200k samples otherwise sync ~200k points of metadata
        prec_curve, recall_curve, thresholds = skm.precision_recall_curve(
            tgt, probs, drop_intermediate=True
        )
        baselines.setdefault(float(np.mean(tgt == 1)), []).append(name)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_curve = 2 * (prec_curve * recall_curve) / (prec_curve + recall_curve)
        f1_curve = np.nan_to_num(f1_curve)  # 0/0 where precision = recall = 0
        f1_score = float(skm.f1_score(tgt, (probs >= 0.5).astype(float)))
        # the last PR point (recall 0, precision 1) has no threshold
        thresholds = np.append(thresholds, np.nan)
        label = f"{name} · F1={f1_score:.2f}" if name else f"F1={f1_score:.2f}"
        series = {
            "x": recall_curve,
            "y": prec_curve,
            "label": label,
            "metadata": _point_metadata(threshold=thresholds, f1=f1_curve),
        }
        return series, f1_score

    # recall is 0/0 without positives, while all-positive targets give a valid curve
    series = _classifier_series(
        targets, probs_positive, df, build_series=build_series, required_classes=(1,)
    )
    ref_lines = []
    if no_skill is not False:
        for prevalence, names in baselines.items():
            label = (
                f"No skill ({', '.join(names)})" if len(baselines) > 1 else "No skill"
            )
            ref_lines.append(
                with_defaults(
                    {
                        "type": "horizontal",
                        "y": prevalence,
                        "style": {"color": "gray", "dash": "6 4"},
                        "label": label,
                        "annotation": {
                            "text": label,
                            "position": "end",
                            "color": "gray",
                        },
                    },
                    no_skill or {},
                )
            )
    return ScatterPlotWidget(
        series=series,
        **with_defaults(
            {
                "ref_lines": ref_lines,
                "x_axis": {"label": "Recall"},
                "y_axis": {"label": "Precision"},
            },
            kwargs,
        ),
    )
