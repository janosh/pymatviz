# %%
from __future__ import annotations

import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.config import config_matplotlib


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


# Configure matplotlib
config_matplotlib()


# %% density scatter plotly
df_tips = px.data.tips().rename(columns={"total_bill": "X", "tip": "Y", "size": "Z"})
fig = pmv.density_scatter_plotly(df_tips, x="X", y="Y", size="Z", facet_col="smoker")
fig.show()
pmv.io.save_and_compress_svg(fig, "density-scatter-plotly")
