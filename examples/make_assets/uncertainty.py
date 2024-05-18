# %%
import numpy as np

from pymatviz.cumulative import cumulative_error, cumulative_residual
from pymatviz.io import save_and_compress_svg
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian


# %% Random regression data
rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)


# %% Uncertainty Plots
ax = qq_gaussian(y_pred, y_true, y_std, identity_line={"line_kwds": {"color": "red"}})
save_and_compress_svg(ax, "normal-prob-plot")


ax = qq_gaussian(
    y_pred, y_true, {"over-confident": y_std, "under-confident": 1.5 * y_std}
)
save_and_compress_svg(ax, "normal-prob-plot-multiple")


ax = error_decay_with_uncert(y_true, y_pred, y_std)
save_and_compress_svg(ax, "error-decay-with-uncert")

eps = 0.2 * np.random.randn(*y_std.shape)

ax = error_decay_with_uncert(y_true, y_pred, {"better": y_std, "worse": y_std + eps})
save_and_compress_svg(ax, "error-decay-with-uncert-multiple")


# %% Cumulative Plots
ax = cumulative_error(y_pred, y_true)
save_and_compress_svg(ax, "cumulative-error")


ax = cumulative_residual(y_pred, y_true)
save_and_compress_svg(ax, "cumulative-residual")
