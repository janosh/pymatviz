# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


# %%
np_rng = np.random.default_rng(seed=0)


# %% Scatter plots laid out as a periodic table
data_dict = {  # random parity data with y = x + noise
    elem.symbol: [
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_scatters(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=10),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
    anno_text={elem.symbol: str(idx - 1) for idx, elem in enumerate(Element)},
    anno_pos=(0.75, 0.2),
    anno_kwargs={"fontsize": 10},
)
pmv.io.save_and_compress_svg(fig, "ptable-scatters-parity")


# %% 2nd pmv.ptable_scatters example
data_dict = {  # random parabola data with y = x^2 + noise
    elem.symbol: [
        np.arange(10),
        (np.arange(10) - 4) ** 2 + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 10, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_scatters(
    data_dict,
    colormap="inferno",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=8, alpha=1),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
    color_elem_strategy="off",
)
pmv.io.save_and_compress_svg(fig, "ptable-scatters-parabola")
