import matplotlib.pyplot as plt
import sklearn.metrics as skm
from numpy import ndarray as array


def roc_curve(targets: array, proba_pos: array, ax=None) -> float:

    if ax is None:
        ax = plt.gca()

    # get the metrics
    fpr, tpr, _ = skm.roc_curve(targets, proba_pos)
    roc_auc = skm.roc_auc_score(targets, proba_pos)

    ax.set_title("Receiver Operating Characteristic")
    ax.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1.1], [0, 1.1], "r--", label="random")
    ax.legend(loc="lower right", frameon=False)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")

    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, 1.05))

    return roc_auc


def precision_recall_curve(targets: array, proba_pos: array, ax=None) -> float:

    if ax is None:
        ax = plt.gca()

    # get the metrics
    precision, recall, _ = skm.precision_recall_curve(targets, proba_pos)
    # round: convert probas to preds
    prec = skm.precision_score(targets, proba_pos.round())

    ax.set_title("Precision Recall Curve")
    ax.plot(recall, precision, "b", label=f"precision = {prec:.2f}")
    # plt.plot([0, 1], [0, 0], "r--", label="random")
    ax.legend(loc="lower left", frameon=False)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")

    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, 1.05))

    return prec
