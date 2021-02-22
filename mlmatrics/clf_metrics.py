import matplotlib.pyplot as plt
from numpy import ndarray as array
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve


def roc_curve(targets: array, proba_pos: array) -> float:
    fpr, tpr, _ = sklearn_roc_curve(targets, proba_pos)
    roc_auc = roc_auc_score(targets, proba_pos)

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--", label="random")
    plt.legend(loc="lower right")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    return roc_auc


def precision_recall_curve(targets: array, proba_pos: array) -> float:

    precision, recall, _ = sklearn_precision_recall_curve(targets, proba_pos)
    prec = precision_score(targets, proba_pos.round())  # round: convert probas to preds

    plt.title("Precision Recall curve for positive label (1: superconductor)")
    plt.plot(recall, precision, "b", label=f"precision = {prec:.2f}")
    plt.plot([0, 1], [0, 0], "r--", label="random")
    plt.legend(loc="lower left")
    plt.ylabel("Precision")
    plt.xlabel("Recall")

    return prec
