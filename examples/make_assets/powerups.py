# %%
from __future__ import annotations

import numpy as np

from pymatviz.histograms import plot_histogram
from pymatviz.powerups import add_ecdf_line
from pymatviz.templates import set_plotly_template


set_plotly_template("pymatviz_white")


# %% Configure matplotlib and load test data
# Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)


# %%
fig = plot_histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=200)
for idx in range(len(fig.data)):
    add_ecdf_line(fig, trace_idx=idx)
fig.show()
