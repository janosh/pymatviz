# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


# %%
np_rng = np.random.default_rng(seed=0)


# %% Line plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.linspace(0, 10, 10),
        np.sin(2 * np.pi * np.linspace(0, 10, 10)) + np_rng.normal(0, 0.2, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_lines(
    data_dict,
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
    anno_text={elem.symbol: str(idx - 1) for idx, elem in enumerate(Element)},
    anno_pos=(0.25, 0.2),
    anno_kwargs={"fontsize": 6},
)
pmv.io.save_and_compress_svg(fig, "ptable-lines")
