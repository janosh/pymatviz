# %%
import pymatviz as pmv


dummy_data = pmv.data.regression()


# %% Cumulative Plots
ax = pmv.cumulative_error(dummy_data.y_pred - dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-error")


ax = pmv.cumulative_residual(dummy_data.y_pred - dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "cumulative-residual")
