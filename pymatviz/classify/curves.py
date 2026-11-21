"""ROC and precision-recall curves of binary classifiers."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pymatviz.utils.data import df_to_arrays
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


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
        if isinstance(probs_positive, Mapping):
            raise TypeError(
                f"when passing a DataFrame, probs_positive must be a column name "
                f"(str) or array, not dict. Got {type(probs_positive).__name__}. "
                f"Pass df=None to use dict of predictions."
            )
        targets, probs_positive = df_to_arrays(df, targets, probs_positive)

    curves_dict: dict[str, dict[str, Any]]
    if isinstance(probs_positive, Mapping):
        # convert array values to dicts if needed. both casts needed since ty keeps
        # ArrayLike & Mapping/dict intersections from the isinstance checks
        probs_dict = cast("Mapping[str, ArrayLike | dict[str, Any]]", probs_positive)
        curves_dict = {
            name: (
                cast("dict[str, Any]", probs)
                if isinstance(probs, dict)
                else {"probs_positive": probs}
            )
            for name, probs in probs_dict.items()
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
        ValueError: If targets hold more than 2 classes or a classifier's targets
            lack any of required_classes.
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
