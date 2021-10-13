from typing import Tuple

import matplotlib.pyplot as plt
import sklearn.metrics as skm
from matplotlib.axes import Axes

from ml_matrics.utils import NumArray


def roc_curve(
    targets: NumArray, proba_pos: NumArray, ax: Axes = None
) -> Tuple[float, Axes]:
    """Plot the receiver operating characteristic curve of a binary
    classifier given target labels and predicted probabilities for
    the positive class.

    Args:
        targets (array): Ground truth targets.
        proba_pos (array): predicted probabilities for the positive class.

    Returns:
        tuple[float, plt.Axes]: The classifier's ROC-AUC and the plt.Axes object.
    """
    if ax is None:
        ax = plt.gca()

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
    targets: NumArray, proba_pos: NumArray, ax: Axes = None
) -> Tuple[float, Axes]:
    """Plot the precision recall curve of a binary classifier.

    Args:
        targets (array): Ground truth targets.
        proba_pos (array): predicted probabilities for the positive class.

    Returns:
        tuple[float, plt.Axes]: The classifier's precision score and the plt.Axes object
    """
    if ax is None:
        ax = plt.gca()

    # get the metrics
    precision, recall, _ = skm.precision_recall_curve(targets, proba_pos)

    # proba_pos.round() converts class probabilities to integer class labels
    prec_score = skm.precision_score(targets, proba_pos.round())

    ax.plot(recall, precision, color="blue", label=f"precision = {prec_score:.2f}")

    ax.plot([0, 1], [0.5, 0.5], "r--", label="No skill")
    ax.legend(loc="lower left", frameon=False)

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision Recall Curve")

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05))

    return prec_score, ax
