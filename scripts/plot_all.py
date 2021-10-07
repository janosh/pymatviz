# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_matrics.correlation import marchenko_pastur
from ml_matrics.cumulative import cum_err, cum_res
from ml_matrics.elements import (
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_ratio,
)
from ml_matrics.histograms import (
    residual_hist,
    spacegroup_hist,
    true_pred_hist,
)
from ml_matrics.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from ml_matrics.quantile import qq_gaussian
from ml_matrics.ranking import err_decay
from ml_matrics.relevance import precision_recall_curve, roc_curve
from ml_matrics.utils import ROOT


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

pred_cols = [col for col in df_roost_ens if "pred" in col]
y_preds = df_roost_ens[pred_cols].to_numpy()
y_pred = y_preds.mean(axis=1)

ale_cols = [col for col in df_roost_ens if "ale" in col]  # aleatoric uncertainties
y_ales = df_roost_ens[ale_cols].to_numpy()

y_var_ale = np.square(y_ales).mean(axis=1)
y_var_epi = y_preds.var(axis=1)

y_std = np.sqrt(y_var_ale + y_var_epi)


def savefig(filename: str) -> None:
    """Save current matplotlib figure as SVG.

    Args:
        filename (str): Name of SVG file (w/o extension).
    """
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


ptable_heatmap(mp_formulas)
savefig("ptable_heatmap")

ptable_heatmap(mp_formulas, log=True)
savefig("ptable_heatmap_log")

ptable_heatmap_ratio(mp_formulas, roost_formulas)
savefig("ptable_heatmap_ratio")

ptable_heatmap_ratio(roost_formulas, mp_formulas)
savefig("ptable_heatmap_ratio_inverse")

hist_elemental_prevalence(mp_formulas, keep_top=15, voffset=1)
savefig("hist_elemental_prevalence")

hist_elemental_prevalence(
    mp_formulas, keep_top=20, log=True, bar_values="count", voffset=1
)
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


# %% Ranking Plots
err_decay(y_true, y_pred, y_std)
savefig("err_decay")

eps = 0.2 * np.random.randn(*y_std.shape)

err_decay(y_true, y_pred, {"better": y_std, "worse": y_std + eps})
savefig("err_decay_multiple")


# %% Relevance Plots
roc_curve(y_binary, y_proba)
savefig("roc_curve")


precision_recall_curve(y_binary, y_proba)
savefig("precision_recall_curve")


# %% Histogram Plots
residual_hist(y_true, y_pred)
savefig("residual_hist")

true_pred_hist(y_true, y_pred, y_std)
savefig("true_pred_hist")

phonons = pd.read_csv(f"{ROOT}/data/matbench-phonons.csv")
spacegroup_hist(phonons.sg_number)
savefig("spacegroup_hist")

spacegroup_hist(phonons.sg_number, show_counts=False)
savefig("spacegroup_hist_no_counts")


# %% Correlation Plots
rand_wide_mat = pd.read_csv(f"{ROOT}/data/rand_wide_matrix.csv", header=None).to_numpy()
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols
corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])
marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
savefig("marchenko_pastur")

rand_wide_mat = pd.read_csv(f"{ROOT}/data/rand_wide_matrix.csv", header=None).to_numpy()
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols
corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])
marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
savefig("marchenko_pastur_significant_eval")

rand_tall_mat = pd.read_csv(f"{ROOT}/data/rand_tall_matrix.csv", header=None).to_numpy()
n_rows, n_cols = rand_tall_mat.shape
corr_mat_rank_deficient = np.corrcoef(rand_tall_mat)
marchenko_pastur(corr_mat_rank_deficient, gamma=n_cols / n_rows)
savefig("marchenko_pastur_rank_deficient")
