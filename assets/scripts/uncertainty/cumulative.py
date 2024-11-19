# %%
import pymatviz as pmv
from pymatviz.test.data import get_regression_data


dummy_data = get_regression_data()


# %% Cumulative Plots
ax = pmv.cumulative_error(dummy_data.y_pred - dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-error")


ax = pmv.cumulative_residual(dummy_data.y_pred - dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-residual")
