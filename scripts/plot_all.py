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
    qq_gaussian,
    roc_curve,
)
from mlmatrics.parity import err_scatter

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


# Load example dataset
df_roost_ens = pd.read_csv(
    f"{ROOT}/data/ex-ensemble-roost.csv", comment="#", na_filter=False
)

y_true = df_roost_ens.target

pred_cols = [col for col in df_roost_ens.columns if "pred" in col]
y_preds = df_roost_ens[pred_cols].T
y_pred = np.mean(y_preds, axis=0)

ale_cols = [col for col in df_roost_ens.columns if "ale" in col]
y_ales = df_roost_ens[ale_cols].T
y_ale = np.mean(np.square(y_ales), axis=0)

y_epi = np.var(y_preds, axis=0, ddof=0)

y_var = y_ale + y_epi
y_std = np.sqrt(y_var)


def savefig(filename: str) -> None:
    plt.savefig(f"{ROOT}/assets/{filename}.svg", bbox_inches="tight")
    plt.close()


# %%
density_scatter(y_pred, y_true)
savefig("density_scatter")


density_scatter_with_hist(y_pred, y_true)
savefig("density_scatter_with_hist")


density_hexbin(y_pred, y_true)
savefig("density_scatter_hex")


density_hexbin_with_hist(y_pred, y_true)
savefig("density_scatter_hex_with_hist")


err_scatter(y_pred, y_true, yerr=y_std)
savefig("err_scatter")


# %%
mp_formulas = pd.read_csv(f"{ROOT}/data/mp-n_elements<2.csv").formula


ptable_elemental_prevalence(mp_formulas)
savefig("ptable_elemental_prevalence")


ptable_elemental_prevalence(mp_formulas, log_scale=True)
savefig("ptable_elemental_prevalence_log")


hist_elemental_prevalence(mp_formulas, keep_top=15)
savefig("hist_elemental_prevalence")


hist_elemental_prevalence(mp_formulas, keep_top=20, log_scale=True, bar_values="count")
savefig("hist_elemental_prevalence_log_count")


# %%
qq_gaussian(y_pred, y_true, y_std)
savefig("normal_prob_plot")


qq_gaussian(y_pred, y_true, {"overconfident": y_std, "underconfident": 1.5 * y_std})
savefig("normal_prob_plot_multiple")


# %%
cum_err(y_pred, y_true)
savefig("cumulative_error")


cum_res(y_pred, y_true)
savefig("cumulative_residual")


# %%
err_decay(y_true, y_pred, y_std)
savefig("err_decay")


err_decay(
    y_true,
    y_pred,
    {"better": y_std, "worse": y_std + 0.2 * np.random.randn(*y_std.shape)},
)
savefig("err_decay_multiple")


# %%
roc_curve(y_binary, y_proba)
savefig("roc_curve")


precision_recall_curve(y_binary, y_proba)
savefig("precision_recall_curve")
