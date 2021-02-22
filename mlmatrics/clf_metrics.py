import matplotlib.pyplot as plt
import sklearn.metrics as skm
from numpy import ndarray as array


def roc_curve(targets: array, proba_pos: array) -> float:
    fpr, tpr, _ = skm.roc_curve(targets, proba_pos)
    roc_auc = skm.roc_auc_score(targets, proba_pos)

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--", label="random")
    plt.legend(loc="lower right")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    return roc_auc


def precision_recall_curve(targets: array, proba_pos: array) -> float:

    precision, recall, _ = skm.precision_recall_curve(targets, proba_pos)
    # round: convert probas to preds
    prec = skm.precision_score(targets, proba_pos.round())

    plt.title("Precision Recall curve for positive label (1: superconductor)")
    plt.plot(recall, precision, "b", label=f"precision = {prec:.2f}")
    plt.plot([0, 1], [0, 0], "r--", label="random")
    plt.legend(loc="lower left")
    plt.ylabel("Precision")
    plt.xlabel("Recall")

    return prec
