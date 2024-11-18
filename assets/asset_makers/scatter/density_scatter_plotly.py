# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import pymatviz as pmv


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# %% density scatter plotly
df_tips = px.data.tips().rename(columns={"total_bill": "X", "tip": "Y", "size": "Z"})
fig = pmv.density_scatter_plotly(df_tips, x="X", y="Y", size="Z", facet_col="smoker")
fig.show()
pmv.io.save_and_compress_svg(fig, "density-scatter-plotly")
