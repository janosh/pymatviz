# %%
import numpy as np

import pymatviz as pmv


# %% Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
y_true = np_rng.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size)


# %% Uncertainty Plots
ax = pmv.error_decay_with_uncert(y_true, y_pred, y_std)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert")

eps = 0.2 * np_rng.standard_normal(*y_std.shape)

ax = pmv.error_decay_with_uncert(
    y_true, y_pred, {"better": y_std, "worse": y_std + eps}
)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert-multiple")
