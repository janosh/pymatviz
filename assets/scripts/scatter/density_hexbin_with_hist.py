# %%
from __future__ import annotations

import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.config import config_matplotlib


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

dummy_data = pmv.data.regression()
# Configure matplotlib
config_matplotlib()


# %% density hexbin with hist
ax = pmv.density_hexbin_with_hist(
    dummy_data.y_pred,
    dummy_data.y_true,
    best_fit_line={"annotate_params": {"loc": "lower center"}},
)
pmv.io.save_and_compress_svg(ax, "density-scatter-hex-with-hist")
