"""Matplotlib density scatter examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


y_true, y_pred, _y_std = pmv.data.regression()


# %% density scatter
ax = pmv.density_scatter(y_true, y_pred)
pmv.io.save_and_compress_svg(ax, "density-scatter")
"""Density scatter with histogram examples."""

# %% density scatter with hist
ax = pmv.density_scatter_with_hist(y_true, y_pred)
pmv.io.save_and_compress_svg(ax, "density-scatter-with-hist")
