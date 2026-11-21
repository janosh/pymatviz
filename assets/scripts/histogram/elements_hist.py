"""Elements histogram examples."""

# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %% Histogram Plots
df_expt_gap = load_dataset("matbench_expt_gap")

widget = pmv.elements_hist(
    df_expt_gap[Key.composition], max_elements=15, show_values="percent"
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/elements-hist.svg", width=700, transparent=True
)
