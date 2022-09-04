# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.ext.matproj import MPRester

from pymatviz.correlation import marchenko_pastur
from pymatviz.cumulative import cumulative_error, cumulative_residual
from pymatviz.histograms import (
    hist_elemental_prevalence,
    residual_hist,
    spacegroup_hist,
    true_pred_hist,
)
from pymatviz.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.ptable import ptable_heatmap, ptable_heatmap_plotly, ptable_heatmap_ratio
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.structure_viz import plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import df_ptable, save_and_compress_svg


# %%
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# random classification data
np.random.seed(42)
rand_clf_size = 100
y_binary = np.random.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(
    y_binary - 0.1 * np.random.normal(scale=5, size=rand_clf_size), 0.2, 0.9
)


# random regression data
rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = y_true + np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)


df_steels = load_dataset("matbench_steels")
df_expt_gap = load_dataset("matbench_expt_gap")
df_phonons = load_dataset("matbench_phonons")

df_phonons[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in df_phonons.structure
]


# %% Parity Plots
density_scatter(y_pred, y_true)
save_and_compress_svg("density-scatter")


density_scatter_with_hist(y_pred, y_true)
save_and_compress_svg("density-scatter-with-hist")


density_hexbin(y_pred, y_true)
save_and_compress_svg("density-scatter-hex")


density_hexbin_with_hist(y_pred, y_true)
save_and_compress_svg("density-scatter-hex-with-hist")


scatter_with_err_bar(y_pred, y_true, yerr=y_std)
save_and_compress_svg("scatter-with-err-bar")


residual_vs_actual(y_true, y_pred)
save_and_compress_svg("residual-vs-actual")


# %% Elemental Plots
ptable_heatmap(df_expt_gap.composition, log=True)
title = (
    f"Elements in Matbench Experimental Band Gap ({len(df_expt_gap):,} compositions)"
)
plt.suptitle(title, y=0.96)
save_and_compress_svg("ptable-heatmap")

ptable_heatmap(df_ptable.atomic_mass)
plt.suptitle("Atomic Mass Heatmap", y=0.96)
save_and_compress_svg("ptable-heatmap-atomic-mass")

ptable_heatmap(df_expt_gap.composition, heat_mode="percent", exclude_elements=["O"])
title = "Elements in Matbench Experimental Band Gap (percent)"
plt.suptitle(title, y=0.96)
save_and_compress_svg("ptable-heatmap-percent")

ptable_heatmap_ratio(df_expt_gap.composition, df_steels.composition, log=True)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
plt.suptitle(title, y=0.96)
save_and_compress_svg("ptable-heatmap-ratio")

hist_elemental_prevalence(df_expt_gap.composition, keep_top=15, v_offset=1)
save_and_compress_svg("hist-elemental-prevalence")


# %% Plotly interactive periodic table heatmap
fig = ptable_heatmap_plotly(
    df_ptable.atomic_mass,
    hover_props=["atomic_mass", "atomic_number"],
    hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
)
fig.update_layout(
    title=dict(text="<b>Atomic mass heatmap</b>", x=0.4, y=0.94, font_size=20)
)
fig.show()
save_and_compress_svg("ptable-heatmap-plotly-more-hover-data", fig)

fig = ptable_heatmap_plotly(df_expt_gap.composition, heat_mode="percent")
title = "Elements in Matbench Experimental Bandgap"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_and_compress_svg("ptable-heatmap-plotly-percent-labels", fig)


# %% Uncertainty Plots
qq_gaussian(y_pred, y_true, y_std)
save_and_compress_svg("normal-prob-plot")


qq_gaussian(y_pred, y_true, {"overconfident": y_std, "underconfident": 1.5 * y_std})
save_and_compress_svg("normal-prob-plot-multiple")


error_decay_with_uncert(y_true, y_pred, y_std)
save_and_compress_svg("error-decay-with-uncert")

eps = 0.2 * np.random.randn(*y_std.shape)

error_decay_with_uncert(y_true, y_pred, {"better": y_std, "worse": y_std + eps})
save_and_compress_svg("error-decay-with-uncert-multiple")


# %% Cumulative Plots
cumulative_error(y_pred, y_true)
save_and_compress_svg("cumulative-error")


cumulative_residual(y_pred, y_true)
save_and_compress_svg("cumulative-residual")


# %% Relevance Plots
roc_curve(y_binary, y_proba)
save_and_compress_svg("roc-curve")


precision_recall_curve(y_binary, y_proba)
save_and_compress_svg("precision-recall-curve")


# %% Histogram Plots
residual_hist(y_true, y_pred)
save_and_compress_svg("residual-hist")

true_pred_hist(y_true, y_pred, y_std)
save_and_compress_svg("true-pred-hist")


# %%
spacegroup_hist(df_phonons.spg_num)
save_and_compress_svg("spg-num-hist")

spacegroup_hist(df_phonons.spg_symbol)
save_and_compress_svg("spg-symbol-hist")


# %% Sunburst Plots
fig = spacegroup_sunburst(df_phonons.spg_num, show_counts="percent")
title = "Matbench Phonons Spacegroup Sunburst"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18))
save_and_compress_svg("spg-num-sunburst", fig)

fig = spacegroup_sunburst(df_phonons.spg_symbol, show_counts="percent")
title = "Matbench Phonons Spacegroup Symbols Sunburst"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18))
save_and_compress_svg("spg-symbol-sunburst", fig)


# %% Correlation Plots
# plot eigenvalue distribution of a pure-noise correlation matrix
# i.e. the correlation matrix contains no significant correlations
# beyond the spurious correlation that occurs randomly
n_rows, n_cols = 500, 1000
rand_wide_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat = np.corrcoef(rand_wide_mat)

marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg("marchenko-pastur")

# plot eigenvalue distribution of a correlation matrix with significant
# (i.e. non-noise) eigenvalue
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols

corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])

marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg("marchenko-pastur-significant-eval")

# plot eigenvalue distribution of a rank-deficient correlation matrix
n_rows, n_cols = 600, 500
rand_tall_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat_rank_deficient = np.corrcoef(rand_tall_mat)

marchenko_pastur(corr_mat_rank_deficient, gamma=n_cols / n_rows)
save_and_compress_svg("marchenko-pastur-rank-deficient")


# %%
df_phonons = load_dataset("matbench_phonons")

n_rows, n_cols = 3, 4
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_rows, 3 * n_cols))

for struct, ax in zip(df_phonons.structure.head(n_rows * n_cols), axs.flat):
    ax = plot_structure_2d(struct, ax=ax)
    spg_symbol, _ = struct.get_space_group_info()
    formula = struct.composition.reduced_formula
    ax.set_title(f"{formula} ({spg_symbol})", fontweight="bold")

save_and_compress_svg("matbench-phonons-structures-2d")


# %% plot some disordered structures in 2d
mp_ids = ["mp-19017", "mp-12712"]
disordered_structs = [
    MPRester().get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    for mp_id in mp_ids
]


# %%
for struct, mp_id in zip(disordered_structs, mp_ids):
    for site in struct:  # disorder structures in-place
        if "Fe" in site.species:
            site.species = {"Fe": 0.4, "C": 0.4, "Mn": 0.2}
        elif "Zr" in site.species:
            site.species = {"Zr": 0.5, "Hf": 0.5}

    ax = plot_structure_2d(struct)
    formula = struct.composition.reduced_formula
    _, spacegroup = struct.get_space_group_info()

    anno_text = f"{formula}\ndisordered {mp_id} with {spacegroup = }"
    href = f"https://materialsproject.org/materials/{mp_id}"
    ax.text(0.5, 1, anno_text, url=href, ha="center", transform=ax.transAxes)

    ax.figure.set_size_inches(8, 8)

    save_and_compress_svg(f"struct-2d-{mp_id}-{formula}-disordered")
    plt.show()


# %% Sankey diagram of random integers
df = pd.DataFrame(np.random.randint(1, 6, size=(100, 2)), columns=["col_a", "col_b"])
fig = sankey_from_2_df_cols(df, df.columns, labels_with_counts="percent")
rand_int_title = "Two sets of 100 random integers from 1 to 5"
fig.update_layout(title=dict(text=rand_int_title, x=0.5, y=0.87))
code_anno = dict(
    x=0.5,
    y=-0.2,
    text="<span style='font-family: monospace;'>df = pd.DataFrame("
    "np.random.randint(1, 6, size=(100, 2)), columns=['col_a','col_b'])<br>"
    "fig = sankey_from_2_df_cols(df, df.columns)</span>",
    font_size=12,
    showarrow=False,
)
fig.add_annotation(code_anno)
save_and_compress_svg("sankey-from-2-df-cols-randints", fig)
