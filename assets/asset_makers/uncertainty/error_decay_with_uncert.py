# %%
import pymatviz as pmv

from ._random_regression_data import np_rng, y_pred, y_std, y_true


# %% Uncertainty Plots
ax = pmv.error_decay_with_uncert(y_true, y_pred, y_std)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert")

eps = 0.2 * np_rng.standard_normal(*y_std.shape)

ax = pmv.error_decay_with_uncert(
    y_true, y_pred, {"better": y_std, "worse": y_std + eps}
)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert-multiple")
