"""Normal probability plots."""

# %%
import pymatviz as pmv


dummy_data = pmv.data.regression()


# %% Uncertainty Plots
ax = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    dummy_data.y_std,
    identity_line={"line_kwargs": {"color": "red"}},
)
pmv.io.save_and_compress_svg(ax, "qq-gaussian")


ax = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    {"over-confident": dummy_data.y_std, "under-confident": 1.5 * dummy_data.y_std},
)
pmv.io.save_and_compress_svg(ax, "qq-gaussian-multiple")
