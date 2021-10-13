import sys
from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from ml_matrics.utils import NumArray


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


def regression_metrics(
    y_true: NumArray, y_preds: NumArray, verbose: bool = False
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Print a common selection of regression metrics.

    TODO make robust by finding the common axis

    Args:
        y_true (array): Regression targets.
        y_preds (array): Model predictions.
        verbose (bool, optional): Whether to print metrics. Defaults to False.

    Returns:
        Tuple[float]:
    """
    y_preds = np.atleast_2d(y_preds)
    y_true = y_true.ravel()
    n_ens = y_preds.shape[0]

    assert y_preds.shape[1] == y_true.shape[0]

    res = y_true - y_preds
    mae = np.abs(res).mean(axis=1)
    mse = np.square(res).mean(axis=1)
    rmse = np.sqrt(mse)
    r2 = r2_score(
        np.repeat(y_true[:, None], n_ens, axis=1),
        y_preds.T,
        multioutput="raw_values",
    )

    if n_ens == 1:
        if verbose:
            print(f"MAE: {mae[0]:.4f}")
            print(f"RMSE: {rmse[0]:.4f}")
            print(f"R2 Score: {r2[0]:.4f}")

        return {"r2": r2, "mae": mae, "rmse": rmse}
    else:
        r2_avg = np.mean(r2)
        r2_std = np.std(r2)

        mae_avg = np.mean(mae)
        mae_std = np.std(mae) / np.sqrt(mae.shape[0])

        rmse_avg = np.mean(rmse)
        rmse_std = np.std(rmse) / np.sqrt(rmse.shape[0])

        if verbose:
            print("\nSingle-Model Performance Metrics:")
            print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
            print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")
            print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = np.mean(y_preds, axis=0)

        mae_ens = np.abs(y_true - y_ens).mean()
        mse_ens = np.square(y_true - y_ens).mean()
        rmse_ens = np.sqrt(mse_ens)

        r2_ens = r2_score(y_true, y_ens)

        if verbose:
            print("\nEnsemble Performance Metrics:")
            print(f"MAE  : {mae_ens:.4f}")
            print(f"RMSE : {rmse_ens:.4f}")
            print(f"R2 Score : {r2_ens:.4f} ")

        return {
            "single": {
                "mae": mae_avg,
                "mae_std": mae_std,
                "rmse": rmse_avg,
                "rmse_std": rmse_std,
                "r2": r2_avg,
                "r2_std": r2_std,
            },
            "ensemble": {
                "mae": mae_ens,
                "rmse": rmse_ens,
                "r2": r2_ens,
            },
        }


def classification_metrics(
    target: NumArray,
    logits: NumArray,
    average: Literal["micro", "macro", "samples", "weighted"] = "micro",
    verbose: bool = False,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Print out metrics for a classification task.

    TODO make less janky, first index is for ensembles, second data, third classes.
    always calculate metrics in the multi-class setting. How to convert binary labels
    to multi-task automatically?

    Args:
        target (array): categorical encoding of the tasks
        logits (array): logits predicted by the model
        average ("micro" | "macro" | "samples" | "weighted"): If None, the scores for
            each class are returned. Else this determines the type of data averaging
            performed on the data. Defaults to 'micro' calculates metrics globally by
            considering each element of the label indicator matrix as a label.
        verbose (bool, optional): Whether to print metrics. Defaults to False.
    """
    if len(logits.shape) != 3:
        raise ValueError(
            "please insure that the logits are of the form (n_ens, n_data, n_classes)"
        )

    if logits.shape[2] == 1:
        logits = np.concatenate((logits, 1 - logits), axis=2)

    acc = np.zeros(len(logits))
    roc_auc = np.zeros(len(logits))
    precision = np.zeros(len(logits))
    recall = np.zeros(len(logits))
    fscore = np.zeros(len(logits))

    for idx, y_logit in enumerate(logits):

        target_ohe = np.zeros_like(y_logit)
        target_ohe[np.arange(target.size), target.astype(int)] = 1

        acc[idx] = accuracy_score(target, np.argmax(y_logit, axis=1))
        roc_auc[idx] = roc_auc_score(target_ohe, y_logit, average=average)
        precision[idx], recall[idx], fscore[idx] = precision_recall_fscore_support(
            target, np.argmax(y_logit, axis=1), average=average
        )[:3]

    if len(logits) == 1:
        if verbose:
            print("\nModel Performance Metrics:")
            print(f"Accuracy : {acc[0]:.4f} ")
            print(f"ROC-AUC  : {roc_auc[0]:.4f}")
            print(f"Precision : {precision[0]:.4f}")
            print(f"Recall    : {recall[0]:.4f}")
            print(f"F-score   : {fscore[0]:.4f}")

        return {
            "acc": acc[0],
            "roc_auc": roc_auc[0],
            "precision": precision[0],
            "recall": recall[0],
            "f1": fscore[0],
        }
    else:
        acc_avg = np.mean(acc)
        acc_std = np.std(acc) / np.sqrt(acc.shape[0])

        roc_auc_avg = np.mean(roc_auc)
        roc_auc_std = np.std(roc_auc) / np.sqrt(roc_auc.shape[0])

        prec_avg = np.mean(precision)
        prec_std = np.std(precision) / np.sqrt(precision.shape[0])

        recall_avg = np.mean(recall)
        recall_std = np.std(recall) / np.sqrt(recall.shape[0])

        fscore_avg = np.mean(fscore)
        fscore_std = np.std(fscore) / np.sqrt(fscore.shape[0])

        if verbose:
            print("\nModel Performance Metrics:")
            print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
            print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
            print(f"Precision : {prec_avg:.4f} +/- {prec_std:.4f}")
            print(f"Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
            print(f"F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(logits, axis=0)

        target_ohe = np.zeros_like(ens_logits)
        target_ohe[np.arange(target.size), target.astype(int)] = 1

        ens_acc = accuracy_score(target, np.argmax(ens_logits, axis=1))
        ens_roc_auc = roc_auc_score(target_ohe, ens_logits, average=average)
        ens_prec, ens_recall, ens_fscore = precision_recall_fscore_support(
            target, np.argmax(ens_logits, axis=1), average=average
        )[:3]

        if verbose:
            print("\nEnsemble Performance Metrics:")
            print(f"Accuracy : {ens_acc:.4f} ")
            print(f"ROC-AUC  : {ens_roc_auc:.4f}")
            print(f"Precision : {ens_prec:.4f}")
            print(f"Recall    : {ens_recall:.4f}")
            print(f"F-score   : {ens_fscore:.4f}")

        return {
            "single": {
                "acc": acc_avg,
                "roc_auc": roc_auc_avg,
                "precision": prec_avg,
                "recall": recall_avg,
                "f1": fscore_avg,
                "acc_std": acc_std,
                "roc_auc_std": roc_auc_std,
                "precision_std": prec_std,
                "recall_std": recall_std,
                "f1_std": fscore_std,
            },
            "ensemble": {
                "acc": ens_acc,
                "roc_auc": ens_roc_auc,
                "precision": ens_prec,
                "recall": ens_recall,
                "f1": ens_fscore,
            },
        }
