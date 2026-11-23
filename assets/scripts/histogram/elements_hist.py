"""Elements histogram examples."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

from matminer.datasets import load_dataset

import pymatviz as pmv


# %% Histogram Plots
df_expt_gap = load_dataset("matbench_expt_gap")

widget = pmv.elements_hist(
    df_expt_gap["composition"], max_elements=15, show_values="percent"
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/elements-hist.svg", width=700, transparent=True
)
