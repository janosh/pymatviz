# %%
from __future__ import annotations

import pymatviz as pmv


dummy_data = pmv.data.regression()
pmv.utils.apply_matplotlib_template()


# %% density hexbin
ax = pmv.density_hexbin(
    dummy_data.y_pred,
    dummy_data.y_true,
    best_fit_line={"annotate_params": {"loc": "lower center"}},
    gridsize=40,
)
pmv.io.save_and_compress_svg(ax, "density-scatter-hex")
