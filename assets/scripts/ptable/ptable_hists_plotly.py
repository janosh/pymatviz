"""Periodic table histogram plotly examples."""

# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}


# %% Example 1: Basic histogram with colorbar
fig = pmv.ptable_hists_plotly(
    data_dict,
    bins=30,
    colorbar=dict(title="Element Distributions"),  # type: ignore[arg-type]
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "ptable-hists-plotly")


# %% Example 2: horizontal colorbar and custom annotations
fig = pmv.ptable_hists_plotly(
    data_dict,  # type: ignore[arg-type]
    bins=30,
    colorbar=dict(title="Element Distributions", orientation="h"),
    color_elem_strategy="background",
    # symbol_kwargs=dict(x=0.25, y=0.8),
    # add atomic numbers to top right of each element tile
    annotations={
        elem.symbol: dict(text=str(idx + 1), font_color="white", y=0.95)
        for idx, elem in enumerate(Element)
    },
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "ptable-hists-plotly-with-annotations")
