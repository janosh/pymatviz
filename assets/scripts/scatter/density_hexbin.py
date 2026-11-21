"""Binned density parity plot examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


y_true, y_pred, _y_std = pmv.data.regression()


# %% Binned density parity plot, bins recomputed on zoom
widget = pmv.density_hexbin(y_true, y_pred, best_fit_line=True, bin_px=8)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/density-hexbin.svg", width=500, transparent=True
)


# %% With marginal histograms
widget = pmv.density_hexbin(
    y_true, y_pred, marginal_bins=100, best_fit_line=True, bin_px=8
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/density-hexbin-with-hist.svg",
    width=500,
    transparent=True,
)
