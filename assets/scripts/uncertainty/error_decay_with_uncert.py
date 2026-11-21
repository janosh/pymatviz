"""Examples of error decay with uncertainty."""

# %%
import numpy as np

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)
dummy_data = pmv.data.regression()


# %% Error decay with a single uncertainty estimate
widget = pmv.error_decay_with_uncert(
    dummy_data.y_true, dummy_data.y_pred, dummy_data.y_std
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/error-decay-with-uncert.svg",
    width=500,
    transparent=True,
)


# %% Compare two uncertainty estimates
eps = 0.2 * np_rng.standard_normal(*dummy_data.y_std.shape)
widget = pmv.error_decay_with_uncert(
    dummy_data.y_true,
    dummy_data.y_pred,
    {"better": dummy_data.y_std, "worse": dummy_data.y_std + eps},
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/error-decay-with-uncert-multiple.svg",
    width=500,
    transparent=True,
)
