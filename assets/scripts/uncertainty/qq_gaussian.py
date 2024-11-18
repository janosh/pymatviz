# %%
import pymatviz as pmv
from pymatviz.data.regression import y_pred, y_std, y_true


# %% Uncertainty Plots
ax = pmv.qq_gaussian(
    y_pred, y_true, y_std, identity_line={"line_kwargs": {"color": "red"}}
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot")


ax = pmv.qq_gaussian(
    y_pred, y_true, {"over-confident": y_std, "under-confident": 1.5 * y_std}
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot-multiple")
