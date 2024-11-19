# %%
from __future__ import annotations

import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.config import config_matplotlib
from pymatviz.test.data import get_regression_data


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

dummy_data = get_regression_data()

# Configure matplotlib
config_matplotlib()


# %% residual vs actual
ax = pmv.residual_vs_actual(dummy_data.y_true, dummy_data.y_pred)
pmv.io.save_and_compress_svg(ax, "residual-vs-actual")
