"""Plotly-based classification metrics visualization."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.metrics as skm
from numpy.typing import ArrayLike

from pymatviz.process_data import df_to_arrays
from pymatviz.utils.plotting import PLOTLY_LINE_STYLES


Predictions: TypeAlias = ArrayLike | str | Mapping[str, ArrayLike | dict[str, Any]]


def _standardize_input(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
    *,
    strict: bool = False,
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
        strict (bool): If True, check that probabilities are in [0, 1].

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

    if strict:
        for trace_dict in curves_dict.values():
            curve_probs = np.asarray(trace_dict["probs_positive"])
            curve_probs_no_nan = curve_probs[~np.isnan(curve_probs)]
            min_prob, max_prob = curve_probs_no_nan.min(), curve_probs_no_nan.max()
            if not (0 <= min_prob <= max_prob <= 1):
                raise ValueError(
                    f"Probabilities must be in [0, 1], got range {(min_prob, max_prob)}"
                )

    return targets, curves_dict


def _add_classifier_curve_traces(
    fig: go.Figure,
    targets: np.ndarray,
    curves_dict: dict[str, dict[str, Any]],
    *,
    build_trace: Callable[[str, int, np.ndarray, np.ndarray], dict[str, Any]],
    meta_key: str,
    kwargs: dict[str, Any],
) -> None:
    """Add one scatter trace per classifier and sort traces by a meta metric.

    Args:
        fig (go.Figure): Figure to add the per-classifier traces to.
        targets (np.ndarray): Ground truth binary labels.
        curves_dict (dict): Map of name to {"probs_positive", **trace_kwargs}.
        build_trace (Callable): Given (name, idx, targets_no_nan, probs_no_nan), returns
            the trace kwargs dict (x, y, name, line, hovertemplate, customdata, meta).
        meta_key (str): Key in each trace's meta dict to sort traces by (descending).
        kwargs (dict): Extra keywords merged into every trace's add_scatter call.
    """
    for idx, (name, curve_cfg) in enumerate(curves_dict.items()):
        trace_kwargs = dict(curve_cfg)  # don't mutate caller-supplied dicts
        curve_probs = np.asarray(trace_kwargs.pop("probs_positive"))
        no_nan = ~np.isnan(targets) & ~np.isnan(curve_probs)
        trace_defaults = build_trace(name, idx, targets[no_nan], curve_probs[no_nan])
        fig.add_scatter(**trace_defaults | kwargs | trace_kwargs)

    # default to -inf so traces missing the metric sort last
    fig.data = sorted(
        fig.data, key=lambda tr: tr.meta.get(meta_key, float("-inf")), reverse=True
    )


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
        no_skill (dict[str, Any] | False): Options for no-skill baseline
            or False to hide it. Commonly needed keys:
            - show_legend: bool = True
            - annotation: dict = None (plotly annotation dict to label the line)
            All other keys are passed to fig.add_scatter()
        **kwargs: Additional keywords passed to fig.add_scatter()

    Returns:
        Figure: Plotly figure object
    """
    fig = go.Figure()
    targets, curves_dict = _standardize_input(targets, probs_positive, df)
    targets = np.asarray(targets)

    def hover_template(trace_name: str) -> str:
        return (
            f"<b>{trace_name}</b><br>"
            "FPR: %{x:.3f}<br>"
            "TPR: %{y:.3f}<br>"
            "Threshold: %{customdata.threshold:.3f}<br>"
        )

    def build_trace(
        name: str, idx: int, tgt: np.ndarray, probs: np.ndarray
    ) -> dict[str, Any]:
        fpr, tpr, thresholds = skm.roc_curve(tgt, probs)
        roc_auc = skm.roc_auc_score(tgt, probs)
        display_name = f"{name} · AUC={roc_auc:.2f}" if name else f"AUC={roc_auc:.2f}"
        return {
            "x": fpr,
            "y": tpr,
            "name": display_name,
            "line": dict(
                width=2, dash=PLOTLY_LINE_STYLES[idx % len(PLOTLY_LINE_STYLES)]
            ),
            "hovertemplate": hover_template(display_name),
            "customdata": [dict(threshold=thr) for thr in thresholds],
            "meta": dict(roc_auc=roc_auc),
        }

    _add_classifier_curve_traces(
        fig,
        targets,
        curves_dict,
        build_trace=build_trace,
        meta_key="roc_auc",
        kwargs=kwargs,
    )

    # Add no-skill baseline line
    if no_skill is not False:
        no_skill = dict(no_skill or {})

        # default to 100 points so whole line is hoverable, not just end points
        xs = no_skill.pop("xs", np.linspace(0, 1, 100))
        ys = no_skill.pop("ys", np.linspace(0, 1, 100))

        # Extract line options if provided
        line_kwargs = no_skill.pop("line", {})

        fig.add_scatter(
            x=xs,
            y=ys,
            line=dict(dash="dash", color="gray") | line_kwargs,
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
        no_skill (dict[str, Any] | False): Options for no-skill baseline
            or False to hide it. Commonly needed keys:
            - show_legend: bool = True
            - annotation: dict = None (plotly annotation dict to label the line)
            All other keys are passed to fig.add_scatter()
        **kwargs: Additional keywords passed to fig.add_scatter()

    Returns:
        Figure: Plotly figure object
    """
    fig = go.Figure()
    targets, curves_dict = _standardize_input(targets, probs_positive, df)
    targets = np.asarray(targets)

    def build_trace(
        name: str, idx: int, tgt: np.ndarray, probs: np.ndarray
    ) -> dict[str, Any]:
        prec_curve, recall_curve, thresholds = skm.precision_recall_curve(tgt, probs)
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
            "line": dict(
                width=2, dash=PLOTLY_LINE_STYLES[idx % len(PLOTLY_LINE_STYLES)]
            ),
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

    _add_classifier_curve_traces(
        fig,
        targets,
        curves_dict,
        build_trace=build_trace,
        meta_key="f1_score",
        kwargs=kwargs,
    )

    # Add no-skill baseline if not explicitly disabled
    if no_skill is not False:
        no_skill = dict(no_skill or {})
        no_skill_line = no_skill.pop("line", {})
        no_skill_anno = no_skill.pop("annotation", {})
        fig.add_hline(
            y=0.5,
            line=dict(dash="dash", color="gray") | no_skill_line,
            showlegend=False,
            annotation=dict(text="No skill", font=dict(color="gray")) | no_skill_anno,
        )

    fig.layout.legend.update(yanchor="bottom", y=0, xanchor="left", x=0)
    fig.layout.update(xaxis_title="Recall", yaxis_title="Precision")

    return fig
