"""Matplotlib density hexbin examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


y_true, y_pred, _y_std = pmv.data.regression()


# %% density hexbin
best_fit_line = {"annotate_params": {"loc": "lower center"}}
ax = pmv.density_hexbin(y_true, y_pred, best_fit_line=best_fit_line, gridsize=40)
# pmv.io.save_and_compress_svg(ax, "density-scatter-hex")


# %% density hexbin with hist
ax = pmv.density_hexbin_with_hist(
    y_true, y_pred, best_fit_line={"annotate_params": {"loc": "lower center"}}
)
# pmv.io.save_and_compress_svg(ax, "density-scatter-hex-with-hist")
