"""Residual vs actual examples."""

# %%
from __future__ import annotations

import pymatviz as pmv


dummy_data = pmv.data.regression()
pmv.utils.apply_matplotlib_template()


# %% residual vs actual
ax = pmv.residual_vs_actual(dummy_data.y_true, dummy_data.y_pred)
pmv.io.save_and_compress_svg(ax, "residual-vs-actual")
