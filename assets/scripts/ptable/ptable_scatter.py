"""Periodic table of per-element scatter and line plots."""

# %%
import numpy as np

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)
atomic_numbers = pmv.utils.df_ptable["z"]

# sinusoidal waves with random frequencies, phases and noise
xs = np.linspace(0, 10, 20)
rand_sine_data = {}
for symbol in atomic_numbers.index:
    freq = np_rng.uniform(0.5, 2)
    phase = np_rng.uniform(0, 2 * np.pi)
    noise = np_rng.normal(0, 0.2, len(xs))
    rand_sine_data[symbol] = (xs, np.sin(freq * xs + phase) + noise)

# random parity data with y = x + noise, third entry colors the markers
rand_parity_data = {
    symbol: (
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 3, 10),
    )
    for symbol in atomic_numbers.index
}


# %% Markers colored by a third value, atomic numbers as tile labels
widget = pmv.ptable_scatter(
    rand_parity_data,
    mode="markers",
    color_bar={"title": "Color value"},
    labels={symbol: str(atomic_num) for symbol, atomic_num in atomic_numbers.items()},
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
