"""Normal probability plots."""

# %%
import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")
dummy_data = pmv.data.regression()


# %% Uncertainty Plots
fig = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    dummy_data.y_std,
    identity_line={"line_kwargs": {"color": "red"}},
)
fig.show()
pmv.io.save_and_compress_svg(fig, "qq-gaussian")


fig = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    {"overconfident": dummy_data.y_std, "less overconfident": 1.5 * dummy_data.y_std},
)
fig.show()
pmv.io.save_and_compress_svg(fig, "qq-gaussian-multiple")
