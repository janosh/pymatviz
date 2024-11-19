# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.data import get_regression_data


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

dummy_data = get_regression_data()


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# %% residual vs actual
ax = pmv.residual_vs_actual(dummy_data.y_true, dummy_data.y_pred)
pmv.io.save_and_compress_svg(ax, "residual-vs-actual")
