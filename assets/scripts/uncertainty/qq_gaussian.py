"""Normal probability plots."""

# %%
import pymatviz as pmv


dummy_data = pmv.data.regression()


# %% Q-Q plot of a single uncertainty estimate
widget = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    dummy_data.y_std,
    identity_line={"color": "red"},
)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/qq-gaussian.svg", width=500)


# %% Compare two uncertainty estimates
widget = pmv.qq_gaussian(
    dummy_data.y_pred,
    dummy_data.y_true,
    {"overconfident": dummy_data.y_std, "less overconfident": 1.5 * dummy_data.y_std},
)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/qq-gaussian-multiple.svg", width=500)
