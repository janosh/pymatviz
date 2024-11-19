# %%
from __future__ import annotations

import plotly.express as px

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")


# %% density scatter plotly
df_tips = px.data.tips().rename(columns={"total_bill": "X", "tip": "Y", "size": "Z"})
fig = pmv.density_scatter_plotly(df_tips, x="X", y="Y", size="Z", facet_col="smoker")
fig.show()
pmv.io.save_and_compress_svg(fig, "density-scatter-plotly")
