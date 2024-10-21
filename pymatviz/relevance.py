"""Plots for evaluating classifier performance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import sklearn.metrics as skm

from pymatviz.utils import df_to_arrays


if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike


def roc_curve(
    targets: ArrayLike | str,
    proba_pos: ArrayLike | str,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
) -> tuple[float, plt.Axes]:
    """Plot the receiver operating characteristic curve of a binary classifier given
    target labels and predicted probabilities for the positive class.

    Args:
        targets (array): Ground truth targets.
        proba_pos (array): predicted probabilities for the positive class.
        df (pd.DataFrame, optional): DataFrame with targets and proba_pos columns.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        tuple[float, ax]: The classifier's ROC-AUC and the plot's matplotlib Axes.
    """
    targets, proba_pos = df_to_arrays(df, targets, proba_pos)
    ax = ax or plt.gca()

    # get the metrics
    false_pos_rate, true_pos_rate, _ = skm.roc_curve(targets, proba_pos)
    roc_auc = skm.roc_auc_score(targets, proba_pos)

    ax.plot(false_pos_rate, true_pos_rate, "b", label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1.1], [0, 1.1], "r--", label="random")
    ax.legend(loc="lower right", frameon=False)

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05))
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")

    return roc_auc, ax


def precision_recall_curve(
    targets: ArrayLike | str,
    proba_pos: ArrayLike | str,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
) -> tuple[float, plt.Axes]:
    """Plot the precision recall curve of a binary classifier.

    Args:
        targets (array): Ground truth targets.
        proba_pos (array): predicted probabilities for the positive class.
        df (pd.DataFrame, optional): DataFrame with targets and proba_pos columns.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        tuple[float, ax]: The classifier's precision score and the matplotlib Axes.
    """
    targets, proba_pos = df_to_arrays(df, targets, proba_pos)
    ax = ax or plt.gca()

    # get the metrics
    precision, recall, _ = skm.precision_recall_curve(targets, proba_pos)

    # proba_pos.round() converts class probabilities to integer class labels
    prec_score = skm.precision_score(targets, proba_pos.round())  # type: ignore[union-attr]

    ax.plot(recall, precision, color="blue", label=f"precision = {prec_score:.2f}")

    ax.plot([0, 1], [0.5, 0.5], "r--", label="No skill")
    ax.legend(loc="lower left", frameon=False)

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05))

    return prec_score, ax
