"""Plotly-based classification metrics visualization."""

from typing import Any, TypeAlias

import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as skm
from numpy.typing import ArrayLike

from pymatviz.utils import df_to_arrays


Predictions: TypeAlias = ArrayLike | str | dict[str, ArrayLike | dict[str, Any]]
PLOTLY_LINE_STYLES = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")


def _standardize_input(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
) -> tuple[ArrayLike, dict[str, dict[str, Any]]]:
    """Standardize input into tuple of (targets, {name: {probs_positive,
    **trace_kwargs}}).

    Handles three input formats for probs_positive:
    1. Basic: array of probabilities
    2. dict of arrays: {"name": probabilities}
    3. dict of dicts: {"name": {"probs_positive": np.array, **trace_kwargs}}
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

    for trace_dict in curves_dict.values():
        curve_probs = np.asarray(trace_dict["probs_positive"])
        min_prob, max_prob = curve_probs.min(), curve_probs.max()
        if not (0 <= min_prob <= max_prob <= 1):
            raise ValueError(
                f"Probabilities must be in [0, 1], got range {(min_prob, max_prob)}"
            )

    return targets, curves_dict


def roc_curve_plotly(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot the receiver operating characteristic (ROC) curve using Plotly.

    Args:
        targets: Ground truth binary labels
        probs_positive: Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **trace_kwargs}}
        df: Optional DataFrame containing targets and probs_positive columns
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
        fpr, tpr, _ = skm.roc_curve(targets[no_nan], curve_probs[no_nan])
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
            "hovertemplate": (
                f"<b>{display_name}</b><br>"
                "FPR: %{x:.3f}<br>"
                "TPR: %{y:.3f}<br>"
                "<extra></extra>"
            ),
            "meta": dict(roc_auc=roc_auc),
        }
        fig.add_scatter(**trace_defaults | kwargs | trace_kwargs)

    # Sort traces by AUC (descending)
    fig.data = sorted(fig.data, key=lambda tr: tr.meta.get("roc_auc"), reverse=True)

    # Random baseline (has 100 points so whole line is hoverable, not just end points)
    rand_baseline = dict(color="gray", width=2, dash="dash")
    fig.add_scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        name="Random",
        line=rand_baseline,
        hovertemplate=(
            "<b>Random</b><br>"
            "FPR: %{x:.3f}<br>"
            "TPR: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    )

    fig.layout.legend.update(yanchor="bottom", y=0, xanchor="right", x=0.99)
    fig.layout.update(xaxis_range=[0, 1.05], yaxis_range=[0, 1.05])
    fig.layout.update(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )

    return fig


def precision_recall_curve_plotly(
    targets: ArrayLike | str,
    probs_positive: Predictions,
    df: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot the precision-recall curve using Plotly.

    Args:
        targets: Ground truth binary labels
        probs_positive: Either:
            - Predicted probabilities for positive class, or
            - dict of form {"name": probabilities}, or
            - dict of form {"name": {"probs_positive": np.array, **trace_kwargs}}
        df: Optional DataFrame containing targets and probs_positive columns
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
        precision, recall, _ = skm.precision_recall_curve(
            targets[no_nan], curve_probs[no_nan]
        )
        f1_score = skm.f1_score(targets[no_nan], np.round(curve_probs[no_nan]))

        metrics_str = f"F1={f1_score:.2f}"
        display_name = f"{name} · {metrics_str}" if name else metrics_str
        trace_defaults = {
            "x": recall,
            "y": precision,
            "name": display_name,
            "line": dict(
                width=2, dash=PLOTLY_LINE_STYLES[idx % len(PLOTLY_LINE_STYLES)]
            ),
            "hovertemplate": (
                f"<b>{display_name}</b><br>"
                "Recall: %{x:.3f}<br>"
                "Prec: %{y:.3f}<br>"
                "F1: {f1_score:.3f}<br>"
                "<extra></extra>"
            ),
            "meta": dict(f1_score=f1_score),
        }
        fig.add_scatter(**trace_defaults | kwargs | trace_kwargs)

    # Sort traces by F1 score (descending)
    fig.data = sorted(fig.data, key=lambda tr: tr.meta.get("f1_score"), reverse=True)

    # No-skill baseline (has 100 points so whole line is hoverable, not just end points)
    no_skill_line = dict(color="gray", width=2, dash="dash")
    fig.add_scatter(
        x=np.linspace(0, 1, 100),
        y=np.full_like(np.linspace(0, 1, 100), 0.5),
        name="No skill",
        line=no_skill_line,
        hovertemplate=(
            "<b>No skill</b><br>"
            "Recall: %{x:.3f}<br>"
            "Prec: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    )

    fig.layout.legend.update(yanchor="bottom", y=0, xanchor="left", x=0)
    fig.layout.update(xaxis_title="Recall", yaxis_title="Precision")

    return fig
