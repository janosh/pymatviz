"""Periodic table of per-element scatter and line plots."""

# /// script
# dependencies = [
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
import numpy as np
from pymatgen.core import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)

# sinusoidal waves with random frequencies, phases and noise
xs = np.linspace(0, 10, 20)
rand_sine_data = {}
for elem in Element:
    freq = np_rng.uniform(0.5, 2)
    phase = np_rng.uniform(0, 2 * np.pi)
    noise = np_rng.normal(0, 0.2, len(xs))
    rand_sine_data[elem.symbol] = (xs, np.sin(freq * xs + phase) + noise)

# random parity data with y = x + noise, third entry colors the markers
rand_parity_data = {
    elem.symbol: (
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 3, 10),
    )
    for elem in Element
}


# %% Markers colored by a third value, atomic numbers as tile labels
widget = pmv.ptable_scatter(
    rand_parity_data,
    mode="markers",
    color_bar={"title": "Color value"},
    labels={elem.symbol: str(elem.Z) for elem in Element},
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/ptable-scatter-markers.svg",
    width=1000,
    transparent=True,
)


# %% Lines on a shared y range
widget = pmv.ptable_scatter(
    rand_sine_data, mode="line", y_range=(-1.5, 1.5), color="crimson"
)
widget.show()


# %% Two named series per element, each name gets its own color
widget = pmv.ptable_scatter(
    {
        sym: {"sin": (xs, ys), "cos": (xs, np.cos(xs))}
        for sym, (xs, ys) in list(rand_sine_data.items())[:54]
    },
    mode="line+markers",
)
widget.show()
