"""Plotly density hexbin examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")
y_true, y_pred, _y_std = pmv.data.regression()


# %% density hexbin
fig = pmv.density_hexbin(
    y_true, y_pred, best_fit_line={"annotate_params": {}}, gridsize=40
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter-hex")


# %% density hexbin with hist
fig = pmv.density_hexbin_with_hist(
    y_true, y_pred, best_fit_line={"annotate_params": {}}
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter-hex-with-hist")
