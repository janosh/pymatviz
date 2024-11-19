# %%
from __future__ import annotations

import pymatviz as pmv


dummy_data = pmv.data.regression()
pmv.utils.apply_matplotlib_template()


# %% density scatter
ax = pmv.density_scatter(dummy_data.y_pred, dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter")
