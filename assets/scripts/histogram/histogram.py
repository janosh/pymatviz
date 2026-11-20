"""Histogram examples."""

# %%
import numpy as np

import pymatviz as pmv


# %% Two overlaid Gaussians binned on shared edges
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)

widget = pmv.histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=100)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/histogram.svg", width=600)


# %% Same data as probability densities on a log y-axis
pmv.histogram(
    {"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=50, density=True, log_y=True
).show()
