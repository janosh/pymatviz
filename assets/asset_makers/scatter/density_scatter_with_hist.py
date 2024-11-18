# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import pymatviz as pmv

from ._random_regression_data import y_pred, y_true


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# %% density scatter with hist
ax = pmv.density_scatter_with_hist(y_pred, y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter-with-hist")
