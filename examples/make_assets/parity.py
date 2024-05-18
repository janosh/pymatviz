# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio

from pymatviz.io import save_and_compress_svg
from pymatviz.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True

px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


# Random regression data
rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)


# %% Parity Plots
ax = density_scatter(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter")


ax = density_scatter_with_hist(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter-with-hist")


ax = density_hexbin(
    y_pred, y_true, best_fit_line={"annotate_params": {"loc": "lower center"}}
)
save_and_compress_svg(ax, "density-scatter-hex")


ax = density_hexbin_with_hist(
    y_pred, y_true, best_fit_line={"annotate_params": {"loc": "lower center"}}
)
save_and_compress_svg(ax, "density-scatter-hex-with-hist")


ax = scatter_with_err_bar(y_pred, y_true, yerr=y_std)
save_and_compress_svg(ax, "scatter-with-err-bar")


ax = residual_vs_actual(y_true, y_pred)
save_and_compress_svg(ax, "residual-vs-actual")
