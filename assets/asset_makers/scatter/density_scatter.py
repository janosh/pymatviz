# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio

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
y_std = abs((y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size))


# %% density scatter
ax = pmv.density_scatter(y_pred, y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter")
