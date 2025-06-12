"""Plotly density scatter examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")
y_true, y_pred, _y_std = pmv.data.regression()


# %% density scatter
fig = pmv.density_scatter(y_true, y_pred)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter")


# %% density scatter with hist
fig = pmv.density_scatter_with_hist(y_true, y_pred)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter-with-hist")
