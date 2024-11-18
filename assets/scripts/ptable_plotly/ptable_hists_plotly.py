# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)


# %%
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}

fig = pmv.ptable_hists_plotly(
    data_dict, bins=30, colorbar=dict(title="Element Distributions")
)
fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-hists-plotly")
