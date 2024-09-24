# %%
from __future__ import annotations

import numpy as np
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")
df_expt_gap = load_dataset("matbench_expt_gap")

df_phonons[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_phonons[Key.structure], desc="Getting spacegroups")
]

# Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
y_true = np_rng.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size)


# %% Histogram Plots
ax = pmv.elements_hist(
    df_expt_gap[Key.composition], keep_top=15, v_offset=200, rotation=0, fontsize=12
)
pmv.io.save_and_compress_svg(ax, "elements-hist")


# %% Spacegroup histograms
for backend in pmv.BACKENDS:
    fig = pmv.spacegroup_bar(df_phonons[Key.spg_num], backend=backend)
    pmv.io.save_and_compress_svg(fig, f"spg-num-hist-{backend}")

    fig = pmv.spacegroup_bar(df_phonons[Key.spg_symbol], backend=backend)
    pmv.io.save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")


# %% plot 2 Gaussians and their cumulative distribution functions
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)

fig = pmv.histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=100)
for idx in range(len(fig.data)):
    pmv.powerups.add_ecdf_line(fig, trace_idx=idx)
fig.show()

pmv.io.save_and_compress_svg(fig, "histogram-ecdf")
