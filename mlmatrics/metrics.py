import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(y_preds, y_true, verbose=False):
    """Print a common selection of regression metrics

    TODO make robust by finding the common axis

    Args:
        y_preds ([type]): [description]
        y_true ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.
    """
    y_preds = np.atleast_2d(y_preds)
    y_true = y_true.ravel()
    n_ens = y_preds.shape[0]

    assert y_preds.shape[1] == y_true.shape[0]

    res = y_true - y_preds
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = r2_score(
        np.repeat(y_true[:, np.newaxis], n_ens, axis=1),
        y_preds.T,
        multioutput="raw_values",
    )

    if n_ens == 1:
        if verbose:
            print("\nModel Performance Metrics:")
            print("R2 Score: {:.4f} ".format(r2[0]))
            print("MAE: {:.4f}".format(mae[0]))
            print("RMSE: {:.4f}".format(rmse[0]))
    else:
        r2_avg = np.mean(r2)
        r2_std = np.std(r2)

        mae_avg = np.mean(mae)
        mae_std = np.std(mae)/np.sqrt(mae.shape[0])

        rmse_avg = np.mean(rmse)
        rmse_std = np.std(rmse)/np.sqrt(rmse.shape[0])

        if verbose:
            print("\nModel Performance Metrics:")
            print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
            print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
            print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = np.mean(y_preds, axis=0)

        mae_ens = np.abs(y_true - y_ens).mean()
        mse_ens = np.square(y_true - y_ens).mean()
        rmse_ens = np.sqrt(mse_ens)

        r2_ens = r2_score(y_true, y_ens)

        if verbose:
            print("\nEnsemble Performance Metrics:")
            print(f"R2 Score : {r2_ens:.4f} ")
            print(f"MAE  : {mae_ens:.4f}")
            print(f"RMSE : {rmse_ens:.4f}")
