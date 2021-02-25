from typing import Dict

import numpy as np
from numpy import ndarray as Array
from sklearn.metrics import r2_score


def regression_metrics(
    y_true: Array, y_preds: Array, verbose: bool = False
) -> Dict[str, float]:
    """Print a common selection of regression metrics

    TODO make robust by finding the common axis

    Args:
        y_true (Array): Regression targets.
        y_preds (Array): Model predictions.
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
