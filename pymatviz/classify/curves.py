"""Plotly-based classification metrics visualization."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.metrics as skm
from numpy.typing import ArrayLike

from pymatviz.process_data import df_to_arrays
from pymatviz.utils.plotting import PLOTLY_LINE_STYLES


type Predictions = ArrayLike | str | Mapping[str, ArrayLike | dict[str, Any]]


def _standardize_input(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
) -> tuple[ArrayLike, dict[str, dict[str, Any]]]:
    """Standardize input into tuple of (targets, {name: {probs_positive,
    **trace_kwargs}}).

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **trace_kwargs}}
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


def _classifier_curves_fig(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: pd.DataFrame | None,
    *,
    build_trace: Callable[[str, np.ndarray, np.ndarray], dict[str, Any]],
    meta_key: str,
    kwargs: dict[str, Any],
) -> go.Figure:
    """Figure with one line per classifier, sorted by a meta metric (descending).

    Args:
        targets (ArrayLike | str): Ground truth labels (see roc_curve).
        probs_positive (Predictions): Per-classifier probabilities (see roc_curve).
        df (pd.DataFrame | None): Optional DataFrame holding both columns.
        build_trace (Callable): Given (name, targets_no_nan, probs_no_nan), returns
            the trace kwargs dict (x, y, name, hovertemplate, customdata, meta).
        meta_key (str): Key in each trace's meta dict to sort traces by.
        kwargs (dict): Extra keywords merged into every trace's add_scatter call.
    """
    targets, curves_dict = _standardize_input(targets, probs_positive, df)
    targets = np.asarray(targets)
    fig = go.Figure()
    for idx, (name, curve_cfg) in enumerate(curves_dict.items()):
        trace_kwargs = dict(curve_cfg)  # don't mutate caller-supplied dicts
        curve_probs = np.asarray(trace_kwargs.pop("probs_positive"))
        no_nan = ~np.isnan(targets) & ~np.isnan(curve_probs)
        trace_defaults = build_trace(name, targets[no_nan], curve_probs[no_nan])
        line_style = dict(
            width=2, dash=PLOTLY_LINE_STYLES[idx % len(PLOTLY_LINE_STYLES)]
        )
        fig.add_scatter(
            **trace_defaults | dict(line=line_style) | kwargs | trace_kwargs
        )

    # default to -inf so traces missing the metric sort last
    fig.data = sorted(
        fig.data, key=lambda tr: tr.meta.get(meta_key, float("-inf")), reverse=True
    )
    return fig


def roc_curve(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: pd.DataFrame | None = None,
    *,
    no_skill: dict[str, Any] | Literal[False] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot the receiver operating characteristic (ROC) curve using Plotly.

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **trace_kwargs}}
        df (pd.DataFrame | None): Optional DataFrame containing targets and
            probs_positive columns
        no_skill (dict[str, Any] | False): Options for the no-skill diagonal or
            False to hide it. Keys: "xs"/"ys" (line points), "line" (plotly line
            style) and "annotation" (plotly annotation dict labeling the line).
        **kwargs: Additional keywords passed to fig.add_scatter()

    Returns:
        Figure: Plotly figure object
    """

    def hover_template(trace_name: str) -> str:
        return (
            f"<b>{trace_name}</b><br>"
            "FPR: %{x:.3f}<br>"
            "TPR: %{y:.3f}<br>"
            "Threshold: %{customdata.threshold:.3f}<br>"
        )

    def build_trace(name: str, tgt: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
        fpr, tpr, thresholds = skm.roc_curve(tgt, probs)
        roc_auc = skm.roc_auc_score(tgt, probs)
        display_name = f"{name} · AUC={roc_auc:.2f}" if name else f"AUC={roc_auc:.2f}"
        return {
            "x": fpr,
            "y": tpr,
            "name": display_name,
            "hovertemplate": hover_template(display_name),
            "customdata": [dict(threshold=thr) for thr in thresholds],
            "meta": dict(roc_auc=roc_auc),
        }

    fig = _classifier_curves_fig(
        targets,
        probs_positive,
        df,
        build_trace=build_trace,
        meta_key="roc_auc",
        kwargs=kwargs,
    )

    if no_skill is not False:
        no_skill = dict(no_skill or {})
        # default to 100 points so whole line is hoverable, not just end points
        fig.add_scatter(
            x=no_skill.pop("xs", np.linspace(0, 1, 100)),
            y=no_skill.pop("ys", np.linspace(0, 1, 100)),
            line=dict(dash="dash", color="gray") | no_skill.pop("line", {}),
            showlegend=False,
            hovertemplate=hover_template("No skill"),
        )
        anno = dict(text="No skill", font=dict(color="gray")) | no_skill.pop(
            "annotation", {}
        )
        fig.add_annotation(**anno, xref="x", yref="y")

    fig.layout.legend.update(yanchor="bottom", y=0, xanchor="right", x=0.99)
    fig.layout.update(
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    return fig


def precision_recall_curve(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: pd.DataFrame | None = None,
    *,
    no_skill: dict[str, Any] | Literal[False] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot the precision-recall curve using Plotly.

    Args:
        targets (ArrayLike | str): Ground truth binary labels
        probs_positive (Predictions): Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **trace_kwargs}}
        df (pd.DataFrame | None): Optional DataFrame containing targets and
            probs_positive columns
        no_skill (dict[str, Any] | False): Options for no-skill baselines computed
            from each classifier's valid rows, or False to hide them. Identical
            baselines share a line. Keys: "line" (plotly line style) and
            "annotation" (plotly annotation dict labeling the line).
        **kwargs: Additional keywords passed to fig.add_scatter()

    Returns:
        Figure: Plotly figure object
    """
    baselines: dict[float, list[str]] = {}

    def build_trace(name: str, tgt: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
        prec_curve, recall_curve, thresholds = skm.precision_recall_curve(tgt, probs)
        baselines.setdefault(float(np.mean(tgt == 1)), []).append(name)
        # f1 scores for each threshold
        f1_curve = 2 * (prec_curve * recall_curve) / (prec_curve + recall_curve)
        f1_curve = np.nan_to_num(f1_curve)  # Handle division by zero
        f1_score = skm.f1_score(tgt, np.round(probs))
        # append final value since threshold has N-1 elements
        thresholds = [*thresholds, 1.0]
        display_name = f"{name} · F1={f1_score:.2f}" if name else f"F1={f1_score:.2f}"
        return {
            "x": recall_curve,
            "y": prec_curve,
            "name": display_name,
            "hovertemplate": (
                f"<b>{display_name}</b><br>"
                "Recall: %{x:.3f}<br>"
                "Prec: %{y:.3f}<br>"
                "F1: %{customdata.f1:.3f}<br>"
                "Threshold: %{customdata.threshold:.3f}<br>"
                "<extra></extra>"
            ),
            "customdata": [
                dict(threshold=thr, f1=f1)
                for thr, f1 in zip(thresholds, f1_curve, strict=True)
            ],
            "meta": dict(f1_score=f1_score),
        }

    fig = _classifier_curves_fig(
        targets,
        probs_positive,
        df,
        build_trace=build_trace,
        meta_key="f1_score",
        kwargs=kwargs,
    )

    if no_skill is not False:
        no_skill = dict(no_skill or {})
        no_skill_line = no_skill.pop("line", {})
        no_skill_anno = no_skill.pop("annotation", {})
        for prevalence, names in baselines.items():
            label = (
                f"No skill ({', '.join(names)})" if len(baselines) > 1 else "No skill"
            )
            fig.add_hline(
                y=prevalence,
                line=dict(dash="dash", color="gray") | no_skill_line,
                showlegend=False,
                annotation=dict(text=label, font=dict(color="gray")) | no_skill_anno,
            )

    fig.layout.legend.update(yanchor="bottom", y=0, xanchor="left", x=0)
    fig.layout.update(xaxis_title="Recall", yaxis_title="Precision")

    return fig
