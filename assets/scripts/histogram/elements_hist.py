"""Elements histogram examples."""

# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %% Histogram Plots
df_expt_gap = load_dataset("matbench_expt_gap")

fig = pmv.elements_hist(
    df_expt_gap[Key.composition], keep_top=15, show_values="percent"
)
fig.show()
pmv.io.save_and_compress_svg(fig, "elements-hist")
