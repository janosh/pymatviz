"""Periodic table of per-element histograms."""

# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}


# %% Histograms on shared bins, bars colored by bin position
widget = pmv.ptable_hists(data_dict, bins=30, color_bar={"title": "Values"})
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/ptable-hists.svg", width=1000, transparent=True
)


# %% Log-scaled bar heights, fixed x range and each element's mean as tile label
widget = pmv.ptable_hists(
    data_dict,
    bins=20,
    x_range=(-4, 4),
    log_y=True,
    color_scale="interpolatePlasma",
    labels=lambda vals: f"{vals.mean():.2f}",
)
widget.show()
