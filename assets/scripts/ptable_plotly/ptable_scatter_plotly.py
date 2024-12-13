# %%
import numpy as np
from pymatgen.core import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)

# Generate some example data - sinusoidal waves with different frequencies and noise
data_dict: dict[str, tuple[list[float], list[float]]] = {}
xs = np.linspace(0, 10, 30)
for elem in Element:
    freq = np_rng.uniform(0.5, 2.0)
    phase = np_rng.uniform(0, 2 * np.pi)
    noise = np_rng.normal(0, 0.2, len(xs))
    ys = np.sin(freq * xs + phase) + noise
    data_dict[elem.symbol] = (xs, ys)


# %% Plot different modes
for mode, color, width in [
    ("markers", "blue", None),
    ("lines", "red", 1.5),
    ("lines+markers", "green", 3),
]:
    fig = pmv.ptable_scatter_plotly(
        data_dict,
        mode=mode,  # type: ignore[arg-type]
        line_kwargs=dict(color=color, width=width) if width else dict(color=color),
        color_elem_strategy="symbol",
        scale=1.2,
    )

    mode_title = mode.replace("+", " + ").title()
    title = f"<b>Periodic Table {mode_title} Plots</b><br>mode='{mode}'"
    fig.layout.title.update(text=title, x=0.4, y=0.8, font_size=20)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"ptable-scatter-plotly-{mode.replace('+','-')}")
