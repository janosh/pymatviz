# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm

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
    struct.get_space_group_info() for struct in tqdm(df_phonons.structure)
]


# %% Parity Plots
ax = density_scatter(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter")


ax = density_scatter_with_hist(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter-with-hist")


ax = density_hexbin(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter-hex")


ax = density_hexbin_with_hist(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter-hex-with-hist")


ax = scatter_with_err_bar(y_pred, y_true, yerr=y_std)
save_and_compress_svg(ax, "scatter-with-err-bar")


ax = residual_vs_actual(y_true, y_pred)
save_and_compress_svg(ax, "residual-vs-actual")


# %% Elemental Plots
ax = ptable_heatmap(df_expt_gap.composition, log=True)
title = (
    f"Elements in Matbench Experimental Band Gap ({len(df_expt_gap):,} compositions)"
)
plt.set_title(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap")

ax = ptable_heatmap(df_ptable.atomic_mass)
plt.set_title("Atomic Mass Heatmap", y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-atomic-mass")

ax = ptable_heatmap(
    df_expt_gap.composition, heat_mode="percent", exclude_elements=["O"]
)
title = "Elements in Matbench Experimental Band Gap (percent)"
plt.set_title(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-percent")

ax = ptable_heatmap_ratio(df_expt_gap.composition, df_steels.composition, log=True)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
plt.set_title(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-ratio")


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
save_and_compress_svg(fig, "ptable-heatmap-plotly-more-hover-data")

fig = ptable_heatmap_plotly(df_expt_gap.composition, heat_mode="percent")
title = "Elements in Matbench Experimental Bandgap"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-percent-labels")

fig = ptable_heatmap_plotly(df_expt_gap.composition, log=True, colorscale="viridis")
title = "Elements in Matbench Experimental Bandgap (log scale)"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-log")


# %% Uncertainty Plots
ax = qq_gaussian(y_pred, y_true, y_std)
save_and_compress_svg(ax, "normal-prob-plot")


ax = qq_gaussian(
    y_pred, y_true, {"over-confident": y_std, "under-confident": 1.5 * y_std}
)
save_and_compress_svg(ax, "normal-prob-plot-multiple")


ax = error_decay_with_uncert(y_true, y_pred, y_std)
save_and_compress_svg(ax, "error-decay-with-uncert")

eps = 0.2 * np.random.randn(*y_std.shape)

ax = error_decay_with_uncert(y_true, y_pred, {"better": y_std, "worse": y_std + eps})
save_and_compress_svg(ax, "error-decay-with-uncert-multiple")


# %% Cumulative Plots
ax = cumulative_error(y_pred, y_true)
save_and_compress_svg(ax, "cumulative-error")


ax = cumulative_residual(y_pred, y_true)
save_and_compress_svg(ax, "cumulative-residual")


# %% Relevance Plots
ax = roc_curve(y_binary, y_proba)
save_and_compress_svg(ax, "roc-curve")


ax = precision_recall_curve(y_binary, y_proba)
save_and_compress_svg(ax, "precision-recall-curve")


# %% Histogram Plots
ax = residual_hist(y_true, y_pred)
save_and_compress_svg(ax, "residual-hist")

ax = true_pred_hist(y_true, y_pred, y_std)
save_and_compress_svg(ax, "true-pred-hist")

ax = hist_elemental_prevalence(df_expt_gap.composition, keep_top=15, v_offset=1)
save_and_compress_svg(ax, "hist-elemental-prevalence")


# %%
ax = spacegroup_hist(df_phonons.spg_num)
save_and_compress_svg(ax, "spg-num-hist")

ax = spacegroup_hist(df_phonons.spg_symbol)
save_and_compress_svg(ax, "spg-symbol-hist")


# %% Sunburst Plots
fig = spacegroup_sunburst(df_phonons.spg_num, show_counts="percent")
title = "Matbench Phonons Spacegroup Sunburst"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18))
save_and_compress_svg(fig, "spg-num-sunburst")

fig = spacegroup_sunburst(df_phonons.spg_symbol, show_counts="percent")
title = "Matbench Phonons Spacegroup Symbols Sunburst"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18))
save_and_compress_svg(fig, "spg-symbol-sunburst")


# %% Correlation Plots
# plot eigenvalue distribution of a pure-noise correlation matrix
# i.e. the correlation matrix contains no significant correlations
# beyond the spurious correlation that occurs randomly
n_rows, n_cols = 500, 1000
rand_wide_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat = np.corrcoef(rand_wide_mat)

ax = marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur")

# plot eigenvalue distribution of a correlation matrix with significant
# (i.e. non-noise) eigenvalue
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols

corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])

ax = marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur-significant-eval")

# plot eigenvalue distribution of a rank-deficient correlation matrix
n_rows, n_cols = 600, 500
rand_tall_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat_rank_deficient = np.corrcoef(rand_tall_mat)

ax = marchenko_pastur(corr_mat_rank_deficient, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur-rank-deficient")


# %%
n_rows, n_cols = 3, 4
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
title = f"{len(axs.flat)} Matbench Phonons Structures"
fig.suptitle(title, fontweight="bold", fontsize=20)

for row, ax in zip(df_phonons.itertuples(), axs.flat):
    idx, struct, *_, spg_num = row
    plot_structure_2d(
        struct,
        ax=ax,
        show_bonds=True,
        bond_kwargs=dict(facecolor="gray", linewidth=2, linestyle="dotted"),
    )
    sub_title = f"{idx + 1}. {struct.formula} ({spg_num})"
    ax.set_title(sub_title, fontweight="bold")

fig.show()
save_and_compress_svg(fig, "matbench-phonons-structures-2d")


# %% plot some disordered structures in 2d
disordered_structs = {
    mp_id: MPRester().get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    for mp_id in ["mp-19017", "mp-12712"]
}


# %%
for mp_id, struct in disordered_structs.items():
    for site in struct:  # disorder structures in-place
        if "Fe" in site.species:
            site.species = {"Fe": 0.4, "C": 0.4, "Mn": 0.2}
        elif "Zr" in site.species:
            site.species = {"Zr": 0.5, "Hf": 0.5}

    ax = plot_structure_2d(struct)
    _, spacegroup = struct.get_space_group_info()

    formula = struct.formula.replace(" ", "")
    text = f"{formula}\ndisordered {mp_id}, {spacegroup = }"
    href = f"https://materialsproject.org/materials/{mp_id}"
    ax.text(
        0.5, 1, text, url=href, ha="center", transform=ax.transAxes, fontweight="bold"
    )

    ax.figure.set_size_inches(8, 8)

    save_and_compress_svg(ax, f"struct-2d-{mp_id}-{formula}-disordered")
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
save_and_compress_svg(fig, "sankey-from-2-df-cols-randints")
