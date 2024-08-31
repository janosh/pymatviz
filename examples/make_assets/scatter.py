# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.datasets import make_blobs

import pymatviz as pmv


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
y_true = np_rng.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size)


# %% density scatter plotly
df_tips = px.data.tips().rename(columns={"total_bill": "X", "tip": "Y", "size": "Z"})
fig = pmv.density_scatter_plotly(df_tips, x="X", y="Y", size="Z", facet_col="smoker")
fig.show()
pmv.io.save_and_compress_svg(fig, "density-scatter-plotly")


# %% density scatter plotly blobs
xs, ys = make_blobs(n_samples=100_000, centers=3, n_features=2, random_state=42)

x_col, y_col, target_col = "feature1", "feature2", "target"
df_blobs = pd.DataFrame(dict(zip([x_col, y_col], xs.T, strict=True)) | {target_col: ys})

fig = pmv.density_scatter_plotly(df=df_blobs, x=x_col, y=y_col)
fig.show()
pmv.io.save_and_compress_svg(fig, "density-scatter-plotly-blobs")


# %% density scatter
ax = pmv.density_scatter(y_pred, y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter")


# %% density scatter with hist
ax = pmv.density_scatter_with_hist(y_pred, y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter-with-hist")


# %% density hexbin
ax = pmv.density_hexbin(
    y_pred, y_true, best_fit_line={"annotate_params": {"loc": "lower center"}}
)
pmv.io.save_and_compress_svg(ax, "density-scatter-hex")


# %% density hexbin with hist
ax = pmv.density_hexbin_with_hist(
    y_pred, y_true, best_fit_line={"annotate_params": {"loc": "lower center"}}
)
pmv.io.save_and_compress_svg(ax, "density-scatter-hex-with-hist")


# %% scatter with error bar
ax = pmv.scatter_with_err_bar(y_pred, y_true, yerr=y_std)
pmv.io.save_and_compress_svg(ax, "scatter-with-err-bar")


# %% residual vs actual
ax = pmv.residual_vs_actual(y_true, y_pred)
pmv.io.save_and_compress_svg(ax, "residual-vs-actual")
