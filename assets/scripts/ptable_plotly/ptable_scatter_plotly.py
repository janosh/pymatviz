# %%
import numpy as np
from pymatgen.core import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)

# Generate some example data - sinusoidal waves with different frequencies and noise
rand_sine_data: dict[str, tuple[list[float], list[float]]] = {}
xs = np.linspace(0, 10, 20)
for elem in Element:
    freq = np_rng.uniform(0.5, 2.0)
    phase = np_rng.uniform(0, 2 * np.pi)
    noise = np_rng.normal(0, 0.2, len(xs))
    ys = np.sin(freq * xs + phase) + noise
    rand_sine_data[elem.symbol] = xs, ys


rand_parity_data = {  # random parity data with y = x + noise
    elem.symbol: [
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 3, 10),
    ]
    for elem in Element
}


# %% Plot different modes
for mode, line_kwargs, marker_kwargs, symbol_kwargs, elem_data_dict in [
    (
        "markers",
        dict(color="blue"),
        dict(size=4),
        dict(x=0, y=0.7, xanchor="left", yanchor="bottom"),
        rand_parity_data,
    ),
    ("lines", dict(color="red", width=1.5), None, None, rand_sine_data),
    (
        "lines+markers",
        dict(color="blue"),
        dict(color="white", size=4),
        dict(x=0.5, y=1, xanchor="center", yanchor="middle"),
        rand_sine_data,
    ),
]:
    fig = pmv.ptable_scatter_plotly(
        elem_data_dict,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        line_kwargs=line_kwargs,  # type: ignore[arg-type]
        color_elem_strategy="symbol",
        scale=1.2,
        marker_kwargs=marker_kwargs,
        symbol_kwargs=symbol_kwargs,
    )

    title = f"<b>Periodic Table Scatter Plots</b><br>{mode=}"
    fig.layout.title.update(text=title, x=0.4, y=0.85, font_size=20)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"ptable-scatter-plotly-{mode.replace('+', '-')}")
