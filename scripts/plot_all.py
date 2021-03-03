# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlmatrics import (
    ROOT,
    cum_err,
    cum_res,
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    err_decay,
    hist_elemental_prevalence,
    precision_recall_curve,
    ptable_elemental_prevalence,
    ptable_elemental_ratio,
    qq_gaussian,
    residual_hist,
    residual_vs_actual,
    roc_curve,
    scatter_with_err_bar,
)

plt.rcParams.update({"font.size": 20})
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.major.width"] = 2.5
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 2.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.major.width"] = 2.5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["ytick.minor.width"] = 2.5
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.figsize"] = (8, 7)


# %%
y_binary, y_proba, y_clf = pd.read_csv(f"{ROOT}/data/rand_clf.csv").to_numpy().T


# na_filter=False so sodium amide (NaN) is not parsed as 'not a number'
df_roost_ens = pd.read_csv(f"{ROOT}/data/ex-ensemble-roost.csv", na_filter=False)

y_true = df_roost_ens.target

pred_cols = [col for col in df_roost_ens.columns if "pred" in col]
y_preds = df_roost_ens[pred_cols].T
y_pred = np.mean(y_preds, axis=0)

ale_cols = [col for col in df_roost_ens.columns if "ale" in col]
y_ales = df_roost_ens[ale_cols].T

y_var_ale = np.square(y_ales).mean(axis=0)
y_var_epi = y_preds.var(axis=0)

y_var = y_var_ale + y_var_epi
y_std = np.sqrt(y_var)


def savefig(filename: str) -> None:
    plt.savefig(f"{ROOT}/assets/{filename}.svg", bbox_inches="tight")
    plt.close()


# %% Parity Plots
density_scatter(y_pred, y_true)
savefig("density_scatter")


density_scatter_with_hist(y_pred, y_true)
savefig("density_scatter_with_hist")


density_hexbin(y_pred, y_true)
savefig("density_scatter_hex")


density_hexbin_with_hist(y_pred, y_true)
savefig("density_scatter_hex_with_hist")


scatter_with_err_bar(y_pred, y_true, yerr=y_std)
savefig("scatter_with_err_bar")


residual_vs_actual(y_true, y_pred)
savefig("residual_vs_actual")


# %% Elemental Plots
mp_formulas = pd.read_csv(f"{ROOT}/data/mp-n_elements<2.csv").formula
roost_formulas = pd.read_csv(f"{ROOT}/data/ex-ensemble-roost.csv").composition


ptable_elemental_prevalence(mp_formulas)
savefig("ptable_elemental_prevalence")


ptable_elemental_prevalence(mp_formulas, log_scale=True)
savefig("ptable_elemental_prevalence_log")


ptable_elemental_ratio(mp_formulas, roost_formulas)
savefig("ptable_elemental_ratio")


ptable_elemental_ratio(mp_formulas, roost_formulas, log_scale=True)
savefig("ptable_elemental_ratio_log")


hist_elemental_prevalence(mp_formulas, keep_top=15)
savefig("hist_elemental_prevalence")


hist_elemental_prevalence(mp_formulas, keep_top=20, log_scale=True, bar_values="count")
savefig("hist_elemental_prevalence_log_count")


# %% Quantile/Calibration Plots
qq_gaussian(y_pred, y_true, y_std)
savefig("normal_prob_plot")


qq_gaussian(y_pred, y_true, {"overconfident": y_std, "underconfident": 1.5 * y_std})
savefig("normal_prob_plot_multiple")


# %% Cumulative Plots
cum_err(y_pred, y_true)
savefig("cumulative_error")


cum_res(y_pred, y_true)
savefig("cumulative_residual")


# %% Rankign Plots
err_decay(y_true, y_pred, y_std)
savefig("err_decay")


err_decay(
    y_true,
    y_pred,
    {"better": y_std, "worse": y_std + 0.2 * np.random.randn(*y_std.shape)},
)
savefig("err_decay_multiple")


# %% Relevance Plots
roc_curve(y_binary, y_proba)
savefig("roc_curve")


precision_recall_curve(y_binary, y_proba)
savefig("precision_recall_curve")


# %% Histogram Plots
residual_hist(y_true, y_pred)
savefig("residual_hist")
