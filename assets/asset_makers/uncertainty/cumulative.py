# %%
import numpy as np

import pymatviz as pmv


# %% Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
y_true = np_rng.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size)


# %% Cumulative Plots
ax = pmv.cumulative_error(y_pred - y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-error")


ax = pmv.cumulative_residual(y_pred - y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-residual")
