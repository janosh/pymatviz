from typing import Tuple

import matplotlib.pyplot as plt
import sklearn.metrics as skm
from matplotlib.axes import Axes
from numpy import ndarray as Array


def roc_curve(targets: Array, proba_pos: Array, ax: Axes = None) -> Tuple[float, Axes]:
    """Plot the receiver operating characteristic curve of a binary
    classifier given target labels and predicted probabilities for
    the positive class.

    Args:
        targets (Array): Ground truth targets.
        proba_pos (Array): predicted probabilities for the positive class.

    Returns:
        float: The classifier's ROC area under the curve.
    """
    if ax is None:
        ax = plt.gca()

    # get the metrics
    fpr, tpr, _ = skm.roc_curve(targets, proba_pos)
    roc_auc = skm.roc_auc_score(targets, proba_pos)

    ax.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1.1], [0, 1.1], "r--", label="random")
    ax.legend(loc="lower right", frameon=False)

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05))
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")

    return float(roc_auc), ax


def precision_recall_curve(
    targets: Array, proba_pos: Array, ax: Axes = None
) -> Tuple[float, Axes]:
    """Plot the precision recall curve of a binary classifier.

    Args:
        targets (Array): Ground truth targets.
        proba_pos (Array): predicted probabilities for the positive class.

    Returns:
        float: The classifier's precision score.
    """
    if ax is None:
        ax = plt.gca()

    # get the metrics
    precision, recall, _ = skm.precision_recall_curve(targets, proba_pos)
    # round: convert probas to preds
    prec = skm.precision_score(targets, proba_pos.round())

    ax.plot(recall, precision, "b", label=f"precision = {prec:.2f}")
    # plt.plot([0, 1], [0, 0], "r--", label="random")
    ax.legend(loc="lower left", frameon=False)

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision Recall Curve")

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05))

    return float(prec), ax
