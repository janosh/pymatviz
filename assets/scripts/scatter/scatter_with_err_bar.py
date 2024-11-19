# %%
from __future__ import annotations

import pymatviz as pmv


dummy_data = pmv.data.regression()
pmv.utils.apply_matplotlib_template()


# %% scatter with error bar
ax = pmv.scatter_with_err_bar(
    dummy_data.y_pred, dummy_data.y_true, yerr=dummy_data.y_std
)
pmv.io.save_and_compress_svg(ax, "scatter-with-err-bar")
