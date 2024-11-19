# %%
import pymatviz as pmv
from pymatviz.test.data import get_regression_data


dummy_data = get_regression_data()


# %% Uncertainty Plots
ax = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    dummy_data.y_std,
    identity_line={"line_kwargs": {"color": "red"}},
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot")


ax = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    {"over-confident": dummy_data.y_std, "under-confident": 1.5 * dummy_data.y_std},
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot-multiple")
