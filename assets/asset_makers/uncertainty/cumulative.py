# %%
import pymatviz as pmv

from ._random_regression_data import y_pred, y_true


# %% Cumulative Plots
ax = pmv.cumulative_error(y_pred - y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-error")


ax = pmv.cumulative_residual(y_pred - y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-residual")
