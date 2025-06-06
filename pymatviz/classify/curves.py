"""Plotly-based classification metrics visualization."""

from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.metrics as skm
from numpy.typing import ArrayLike

from pymatviz.process_data import df_to_arrays


Predictions: TypeAlias = ArrayLike | str | dict[str, ArrayLike | dict[str, Any]]
PLOTLY_LINE_STYLES = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")


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
        targets, probs_positive = df_to_arrays(df, targets, probs_positive)  # type: ignore[arg-type]

    if isinstance(probs_positive, dict):
        # Convert array values to dicts if needed
        curves_dict = {
            name: (probs if isinstance(probs, dict) else {"probs_positive": probs})
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


def roc_curve_plotly(
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

    for idx, (name, trace_kwargs) in enumerate(curves_dict.items()):
        # Extract required data and optional trace kwargs
        curve_probs = np.asarray(trace_kwargs.pop("probs_positive"))

        no_nan = ~np.isnan(targets) & ~np.isnan(curve_probs)
        fpr, tpr, thresholds = skm.roc_curve(targets[no_nan], curve_probs[no_nan])
        roc_auc = skm.roc_auc_score(targets[no_nan], curve_probs[no_nan])

        roc_str = f"AUC={roc_auc:.2f}"
        display_name = f"{name} · {roc_str}" if name else roc_str
        trace_defaults = {
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
        fig.add_scatter(**trace_defaults | kwargs | trace_kwargs)

    # Sort traces by AUC (descending)
    fig.data = sorted(fig.data, key=lambda tr: tr.meta.get("roc_auc"), reverse=True)

    # Add no-skill baseline line
    if no_skill is not False:
        # Set up hover template for no-skill line
        no_skill = no_skill or {}
        no_skill.setdefault("hovertemplate", hover_template("No skill"))

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
    fig.layout.update(xaxis_range=[0, 1.05], yaxis_range=[0, 1.05])
    fig.layout.update(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )

    return fig


def precision_recall_curve_plotly(
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

    for idx, (name, trace_kwargs) in enumerate(curves_dict.items()):
        # Extract required data and optional trace kwargs
        curve_probs = np.asarray(trace_kwargs.pop("probs_positive"))
        no_nan = ~np.isnan(targets) & ~np.isnan(curve_probs)
        prec_curve, recall_curve, thresholds = skm.precision_recall_curve(
            targets[no_nan], curve_probs[no_nan]
        )
        # f1 scores for each threshold
        f1_curve = 2 * (prec_curve * recall_curve) / (prec_curve + recall_curve)
        f1_curve = np.nan_to_num(f1_curve)  # Handle division by zero
        f1_score = skm.f1_score(targets[no_nan], np.round(curve_probs[no_nan]))

        # append final value since threshold has N-1 elements
        thresholds = [*thresholds, 1.0]

        metrics_str = f"F1={f1_score:.2f}"
        display_name = f"{name} · {metrics_str}" if name else metrics_str
        trace_defaults = {
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
        fig.add_scatter(**trace_defaults | kwargs | trace_kwargs)

    # Sort traces by F1 score (descending)
    fig.data = sorted(fig.data, key=lambda tr: tr.meta.get("f1_score"), reverse=True)

    # Add no-skill baseline if not explicitly disabled
    if no_skill is not False:
        no_skill = no_skill or {}
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
