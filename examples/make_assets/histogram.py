# %%
import numpy as np
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz.enums import Key
from pymatviz.histograms import elements_hist, histogram, spacegroup_hist
from pymatviz.io import save_and_compress_svg
from pymatviz.powerups import add_ecdf_line
from pymatviz.templates import set_plotly_template
from pymatviz.utils import BACKENDS


set_plotly_template("pymatviz_white")


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
ax = elements_hist(
    df_expt_gap[Key.composition], keep_top=15, v_offset=200, rotation=0, fontsize=12
)
save_and_compress_svg(ax, "elements-hist")


# %% Spacegroup histograms
for backend in BACKENDS:
    fig = spacegroup_hist(df_phonons[Key.spg_num], backend=backend)
    save_and_compress_svg(fig, f"spg-num-hist-{backend}")

    fig = spacegroup_hist(df_phonons[Key.spg_symbol], backend=backend)
    save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")


# %% plot 2 Gaussians and their cumulative distribution functions
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)

fig = histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=100)
for idx in range(len(fig.data)):
    add_ecdf_line(fig, trace_idx=idx)
fig.show()

save_and_compress_svg(fig, "plot-histogram-ecdf")
